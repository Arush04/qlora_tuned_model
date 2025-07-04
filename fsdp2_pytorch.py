import os
import torch
import torch.distributed as dist

from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader

def tokenize_function(examples, tokenizer, max_length=128):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt"
    )

def main():
    # 1. Distributed initialization
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"

    # 2. Model config and instantiation on meta device
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    config = AutoConfig.from_pretrained(model_name)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)

    # 3. Device mesh for FSDP2
    device_mesh = init_device_mesh("cuda", (world_size,))

    # 4. Mixed precision policy (bfloat16)
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        output_dtype=torch.bfloat16,
    )

    # 5. Shard all transformer blocks and then the root model
    for module in model.modules():
        if isinstance(module, LlamaDecoderLayer):
            fully_shard(module, mesh=device_mesh, mp_policy=mp_policy, reshard_after_forward=True)
    fully_shard(model, mesh=device_mesh, mp_policy=mp_policy, reshard_after_forward=True)

    # 6. Materialize model on CUDA device
    model.to_empty(device=device)

    # 7. Load DCP checkpoint
    import torch.distributed.checkpoint as dcp
    model_state_dict = model.state_dict()
    dcp.load(state_dict=model_state_dict, checkpoint_id="my_llama3_weights_dcp")
    model.load_state_dict(model_state_dict, strict=False)

    # 8. Prepare for QLoRA
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora_config = LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # 9. Tokenizer and dataset (only download tokenizer on rank 0)
    if local_rank == 0:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    dist.barrier()
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, padding_side="left", add_eos_token=True, add_bos_token=True, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    # 10. Load and preprocess dataset (using wikitext-2 for demonstration)
    raw_datasets = load_dataset("wikitext", "wikitext-2-raw-v1")
    tokenized_datasets = raw_datasets.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=["text"],
    )

    train_dataset = tokenized_datasets["train"]

    # 11. DataLoader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=2,  # Adjust based on GPU memory
        shuffle=True,
        collate_fn=lambda x: {
            "input_ids": torch.stack([f["input_ids"].squeeze(0) for f in x]),
            "attention_mask": torch.stack([f["attention_mask"].squeeze(0) for f in x]),
        }
    )

    # 12. Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

    # 13. Training loop
    model.train()
    num_epochs = 1
    for epoch in range(num_epochs):
        for step, batch in enumerate(train_dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = input_ids.clone()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % 10 == 0 and local_rank == 0:
                print(f"Epoch {epoch} Step {step} Loss: {loss.item()}")

    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
