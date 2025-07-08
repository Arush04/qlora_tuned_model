import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, OffloadPolicy
from transformers import (
    AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from datasets import load_dataset

def create_prompt(sample):
    prompt = (
        "### Context:\n"
        f"{sample['Context']}\n\n"
        "### Response:\n"
        f"{sample['Response']}"
    )
    return {"text": prompt}

def tokenize_function(batch, tokenizer, max_length=128):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt"
    )

def collate_fn(batch):
    input_ids = torch.tensor([item["input_ids"] for item in batch], dtype=torch.long)
    attention_mask = torch.tensor([item["attention_mask"] for item in batch], dtype=torch.long)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": input_ids.clone()
    }


def main():
    rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", rank=rank)
    torch.manual_seed(0)
    gradient_accumulation_steps = 2
    total_loss = 0
    base_model = "meta-llama/Llama-3.1-8B-Instruct"

    # ---- Load model without quantization for FSDP2 compatibility ----
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_cache=False,
    )
    model.gradient_checkpointing_enable()
    # ---- LoRA Configuration ----
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Added more target modules
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model = model.to(torch.bfloat16)
    for param in model.parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if "lora" in name.lower():
            param.requires_grad = True

    # ---- FSDP2 sharding ----
    world_size = dist.get_world_size()
    device_mesh = init_device_mesh("cuda", (world_size,))
    fsdp_kwargs = {
        "mp_policy": MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
        ),
        "offload_policy": OffloadPolicy()
    }
    
    for module in model.modules():
        if isinstance(module, LlamaDecoderLayer):
            fully_shard(module, mesh=device_mesh, **fsdp_kwargs)
    fully_shard(model, mesh=device_mesh, **fsdp_kwargs)

    model.to(device)

    # ---- Tokenizer ----
    if rank == 0:
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
    dist.barrier()
    tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token

    # ---- Load and split dataset ----
    dataset = load_dataset("Amod/mental_health_counseling_conversations")
    train_test_split = dataset["train"].train_test_split(test_size=0.3, seed=42)
    test_validation_split = train_test_split["test"].train_test_split(test_size=1/3, seed=42)

    dataset_train = train_test_split["train"]
    dataset_validation = test_validation_split["train"]
    dataset_test = test_validation_split["test"]

    if rank == 0:
        print("training dataset ", len(dataset_train))
        print("validation dataset ", len(dataset_validation))
        print("test dataset ", len(dataset_test))

    # ---- Tokenize datasets ----
    dataset_train = dataset_train.map(create_prompt)
    dataset_validation = dataset_validation.map(create_prompt)
    dataset_test = dataset_test.map(create_prompt)

    def tokenize_batch(batch):
        return tokenize_function(batch, tokenizer, max_length=128)

    dataset_train = dataset_train.map(tokenize_batch, batched=True, remove_columns=['Context', 'Response'])
    dataset_validation = dataset_validation.map(tokenize_batch, batched=True, remove_columns=['Context', 'Response'])
    dataset_test = dataset_test.map(tokenize_batch, batched=True, remove_columns=['Context', 'Response'])

    # ---- DataLoader ----
    batch_size = 1
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # ---- Training Loop ----
    model.train()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=2e-3)  # Reduced learning rate
    
    if rank == 0:
        print(f"Number of trainable parameters: {len(trainable_params)}")
        total_params = sum(p.numel() for p in trainable_params)
        print(f"Total trainable parameters: {total_params}")
    
    for step, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss / gradient_accumulation_steps
        loss.backward()
        if (step + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            if rank == 0:
                print(f"Step {step+1}, Loss: {loss.item() * gradient_accumulation_steps}")
        
        total_loss += loss.item()
        torch.cuda.empty_cache()
        if step > 20:  # For demonstration, stop after 20 steps
            break

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
