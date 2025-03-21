import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import get_peft_model, LoraConfig
from datasets import load_dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # Ensure only GPUs 0 and 1 are used

def cleanup():
    dist.destroy_process_group()

def setup(rank, world_size):
    """Initialize distributed training environment"""
    os.environ['MASTER_ADDR'] = '127.0.0.1'  # Use 127.0.0.1 instead of 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    dist.init_process_group(
        backend='nccl', 
        init_method='env://', 
        rank=rank,  # Use the rank passed to the function
        world_size=world_size
    )

def train_with_fsdp(rank, world_size, training_dataset, testing_dataset):
    """Main training function for each process"""
    torch.cuda.set_device(rank)  # Assign correct GPU
    setup(rank, world_size)

    base_model = "meta-llama/Llama-3.1-8B"

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map={"": rank},  # Explicitly assign GPU
    )

    tokenizer_model_name = "unsloth/meta-Llama-3.1-8B-Instruct-bnb-4bit"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        per_device_train_batch_size=1,  # Reduce memory usage
        gradient_accumulation_steps=8,
        max_steps=10,
        logging_steps=1,
        save_steps=5,
        learning_rate=2e-5,
        bf16=True,
        seed=3407,
        gradient_checkpointing=True,
        fsdp="full_shard auto_wrap",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=training_dataset,
        eval_dataset=testing_dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=1024,
    )

    trainer.train()

    if rank == 0:
        trainer.save_model("./outputs/final_model")

    cleanup()

if __name__ == "__main__":
    dataset = load_dataset("SmallDoge/SmallThoughts")
    split_dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)
    training_dataset = split_dataset["train"]
    testing_dataset = split_dataset["test"]

    world_size = torch.cuda.device_count()
    print(f"Detected {world_size} GPUs")

    mp.spawn(
        train_with_fsdp,
        args=(world_size, training_dataset, testing_dataset),
        nprocs=world_size,
        join=True
    )
