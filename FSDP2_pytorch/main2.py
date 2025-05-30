import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.optim.lr_scheduler import StepLR
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from datasets import load_dataset
import functools
from helper_functions import *

# --- Helper: Preprocessing ---
class HFTextDataset(Dataset):
    def __init__(self, hf_dataset, input_key="input_ids", label_key="labels"):
        self.hf_dataset = hf_dataset
        self.input_key = input_key
        self.label_key = label_key

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        return {
            "input_ids": torch.tensor(item[self.input_key], dtype=torch.long),
            "labels": torch.tensor(item[self.label_key], dtype=torch.long)
        }


# --- Training and Validation ---
def train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=None):
    model.train()
    local_rank = int(os.environ['LOCAL_RANK'])
    fsdp_loss = torch.zeros(2).to(local_rank)
    for batch in train_loader:
        x = batch["input_ids"].to(local_rank)
        y = batch["labels"].to(local_rank)
        optimizer.zero_grad()
        outputs = model(input_ids=x, labels=y)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        fsdp_loss[0] += loss.item() * x.size(0)
        fsdp_loss[1] += x.size(0)
    dist.all_reduce(fsdp_loss, op=dist.ReduceOp.SUM)
    train_loss = fsdp_loss[0] / fsdp_loss[1]
    if rank == 0:
        print(f"Train Loss: {train_loss:.4f}")
    return train_loss

def validation(model, rank, world_size, val_loader):
    model.eval()
    local_rank = int(os.environ['LOCAL_RANK'])
    fsdp_loss = torch.zeros(2).to(local_rank)
    with torch.no_grad():
        for batch in val_loader:
            x = batch["input_ids"].to(local_rank)
            y = batch["labels"].to(local_rank)
            outputs = model(input_ids=x, labels=y)
            loss = outputs.loss
            fsdp_loss[0] += loss.item() * x.size(0)
            fsdp_loss[1] += x.size(0)
    dist.all_reduce(fsdp_loss, op=dist.ReduceOp.SUM)
    val_loss = fsdp_loss[0] / fsdp_loss[1]
    if rank == 0:
        print(f"Validation Loss: {val_loss:.4f}")
    return val_loss

# --- Main FSDP2 logic ---
def fsdp_main():
    class Args:
        batch_size = 1
        test_batch_size = 1
        epochs = 10
        lr = 2e-4
        gamma = 0.7
        no_cuda = False
        seed = 42
        track_memory = True
        run_validation = True
        save_model = False

    args = Args()

    # Distributed setup
    local_rank = int(os.environ['LOCAL_RANK'])
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank
    )

    # Dataset loading and splitting
    dataset = load_dataset("Amod/mental_health_counseling_conversations")
    train_test_split = dataset["train"].train_test_split(test_size=0.3, seed=args.seed)
    test_validation_split = train_test_split["test"].train_test_split(test_size=1/3, seed=args.seed)
    dataset_train = train_test_split["train"]
    dataset_validation = test_validation_split["train"]
    dataset_test = test_validation_split["test"]

    print("training dataset ", len(dataset_train))
    print("validation dataset ", len(dataset_validation))
    print("test dataset ", len(dataset_test))

    # Quantization config
    compute_dtype = getattr(torch, "float16")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )

    # Model and tokenizer
    base_model = "microsoft/phi-2"
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map={"": local_rank},
        quantization_config=bnb_config,
        trust_remote_code=True,
        use_auth_token=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        base_model, trust_remote_code=True, padding_side="left", add_eos_token=True, add_bos_token=True, use_fast=False
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Preprocessing
    max_length = 2048  # Or use your get_max_length(model) helper
    dataset_validation_pre = preprocess_dataset(tokenizer, max_length, args.seed, dataset_validation)

    train_dataset = HFTextDataset(dataset_train)
    val_dataset = HFTextDataset(dataset_validation)

    sampler1 = DistributedSampler(train_dataset, rank=rank, num_replicas=world_size, shuffle=True)
    sampler2 = DistributedSampler(val_dataset, rank=rank, num_replicas=world_size)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler1, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.test_batch_size, sampler=sampler2, num_workers=2, pin_memory=True)
    for batch in train_loader:
        print({k: v.dtype for k, v in batch.items()})
        break


    # LoRA config
    model = prepare_model_for_kbit_training(model)
    config = LoraConfig(
        r=32,
        lora_alpha=32,
        target_modules=[
            'q_proj',
            'k_proj',
            'v_proj',
            'dense'
        ],
        bias="none",
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )
    model.gradient_checkpointing_enable()
    model = get_peft_model(model, config)
    model.config.use_cache = False

    # FSDP wrapping
    model = FSDP(model, device_id=local_rank)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    best_val_loss = float("inf")
    curr_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss = train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=sampler1)
        if args.run_validation:
            curr_val_loss = validation(model, rank, world_size, val_loader)
        scheduler.step()
        if curr_val_loss < best_val_loss:
            best_val_loss = curr_val_loss
            if rank == 0:
                print(f"-->>>> New Val Loss Record: {best_val_loss}")

    dist.barrier()
    dist.destroy_process_group()

if __name__ == '__main__':
    fsdp_main()
