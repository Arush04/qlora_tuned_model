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
    print(f"RANK: {rank}") 
    dist.init_process_group(
        backend='nccl', 
        init_method='env://', 
        rank=rank,  # Use the rank passed to the function
        world_size=world_size
    )

def train_with_fsdp(rank, world_size, training_dataset, testing_dataset):
    """Main training function for each process"""
    torch.cuda.set_device(rank)  # Assign correct GPU per process
    
    print(f"Process {rank} is running on GPU {torch.cuda.current_device()}")
    setup(rank, world_size)
    
    # Load model on each process
    base_model = "meta-llama/Llama-3.1-8B"
    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )
    
    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map={"": rank},
        attn_implementation="sdpa",
        quantization_config=bnb_config,
    )
    print("model loaded")
    # Load tokenizer
    tokenizer_model_name = "unsloth/meta-Llama-3.1-8B-Instruct-bnb-4bit"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name)
    tokenizer.padding_side = "right"
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    # LoRA configuration
    lora_config = LoraConfig(
        r=16,                    # Rank
        lora_alpha=32,           # Alpha parameter for LoRA scaling
        target_modules=[
            "q_proj", 
            "k_proj", 
            "v_proj", 
            "o_proj",
            "gate_proj", 
            "up_proj", 
            "down_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # Apply LoRA adapters
    model = get_peft_model(model, lora_config)
    
    # FSDP configuration
    mixed_precision_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    print("mixed precision done")
    auto_wrap_policy = transformer_auto_wrap_policy(
        module=model, recurse=True, nonwrapped_numel=2e5
    )
    
    # Wrap model with FSDP
    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mixed_precision_policy,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
    )
    print("fspd done")
    
    # Configure saving/loading for FSDP
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"./outputs/{rank}",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=1,
        max_steps=10,
        logging_steps=1,
        save_steps=5,
        learning_rate=2e-5,
        fp16=False,
        bf16=True,
        seed=3407,
        data_seed=3407,
        gradient_checkpointing=True,
        save_total_limit=1,
        report_to="none",
        ddp_find_unused_parameters=False,
        fsdp="full_shard auto_wrap",
        fsdp_config={
            "fsdp_offload_params": False,
            "fsdp_backward_prefetch": "backward_pre",
            "fsdp_sync_module_states": True,
            "fsdp_state_dict_type": "full",
        },
    )
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=training_dataset,
        eval_dataset=testing_dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=1024,
    )
    print("going for training")
    # Train model
    trainer.train()
    
    # Save model on rank 0
    if rank == 0:
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            state_dict = model.state_dict()
            trainer.save_model("./outputs/final_model")
        
        # Also save adapter weights separately
        model.save_pretrained("./outputs/final_model_adapters")
    
    # Cleanup
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
