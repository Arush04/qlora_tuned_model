import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy
from transformers import (
    AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
)
from datasets import load_dataset

def tokenize_function(batch, tokenizer, max_length=128):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt"
    )

def collate_fn(batch):
    input_ids = torch.stack([item["input_ids"].squeeze(0) for item in batch])
    attention_mask = torch.stack([item["attention_mask"].squeeze(0) for item in batch])
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

    base_model = "meta-llama/Llama-3.1-8B-Instruct"

    # ---- BitsAndBytesConfig for FSDP2 compatibility ----
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_storage=torch.bfloat16,  # critical for FSDP2!
    )

    # ---- Load quantized model ----
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,  # ensure consistency
        trust_remote_code=True,
        use_auth_token=True
    )

    # ---- FSDP2 sharding ----
    world_size = dist.get_world_size()
    device_mesh = init_device_mesh("cuda", (world_size,))
    fsdp_kwargs = {
        "mp_policy": MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )
    }
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer
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
    def tokenize_batch(batch):
        return tokenize_function(batch, tokenizer, max_length=128)

    dataset_train = dataset_train.map(tokenize_batch, batched=True, remove_columns=["text"])
    dataset_validation = dataset_validation.map(tokenize_batch, batched=True, remove_columns=["text"])
    dataset_test = dataset_test.map(tokenize_batch, batched=True, remove_columns=["text"])

    # ---- DataLoader ----
    batch_size = 2
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # ---- Training Loop ----
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    for step, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
        if step % 10 == 0 and rank == 0:
            print(f"Step {step}, Loss: {loss.item()}")
        if step == 20:  # For demonstration, stop after 20 steps
            break

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
