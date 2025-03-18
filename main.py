import os
import torch
import torch.distributed as dist
import transformers
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    BitsAndBytesConfig,
    set_seed
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer
import torch.multiprocessing as mp
from accelerate import init_empty_weights
import functools
from typing import Type, Dict, List, Optional, Union

# Initialize distributed environment
def setup():
    # Initialize the process group
    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup():
    dist.destroy_process_group()

def sft_main(args):
    # Set up distributed training environment
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    # Initialize distributed setup
    setup()
    
    # Set seeds for reproducibility
    set_seed(args.seed)
    
    if rank == 0:
        print(f"Starting training with {world_size} GPUs")
    
    # Load base model and tokenizer
    base_model = "meta-llama/Llama-3.1-8B"
    new_model = "Thoughts.AI"
    
    # Set up quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )
    
    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",  # This will be overridden by FSDP
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
    )
    
    # Prepare the model for k-bit training (needed for QLoRA)
    model = prepare_model_for_kbit_training(
        model, 
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    
    # Load tokenizer
    tokenizer_model_name = "unsloth/meta-Llama-3.1-8B-Instruct-bnb-4bit"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name)
    tokenizer.padding_side = "right"
    
    # Set padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Configure LoRA
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        modules_to_save=["embed_tokens", "lm_head"],
    )
    
    # Load SmallThoughts dataset
    dataset = load_dataset("SmallDoge/SmallThoughts")
    split_dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)
    training_dataset = split_dataset["train"]
    testing_dataset = split_dataset["test"]
    
    if rank == 0:
        print(f"Training dataset size: {len(training_dataset)}")
        print(f"Testing dataset size: {len(testing_dataset)}")
        
        # Print a sample to understand the structure
        print("\nSample data point:")
        for key in training_dataset[0]:
            print(f"{key}: {training_dataset[0][key][:100]}..." if isinstance(training_dataset[0][key], str) else f"{key}: {training_dataset[0][key]}")
    
    # Format instructions function - Adjust based on your dataset structure
    # Note: This assumes the dataset has "instruction" and "response" fields
    # If your dataset has different field names, adjust accordingly
    def format_instruction(example):
        # Check which fields are available in the dataset
        if "instruction" in example and "response" in example:
            # For instruction-response format
            formatted_text = f"""<|begin_of_text|><|user|>
{example["instruction"]}<|end_of_text|>

<|assistant|>
{example["response"]}<|end_of_text|>"""
        elif "prompt" in example and "completion" in example:
            # Alternative format
            formatted_text = f"""<|begin_of_text|><|user|>
{example["prompt"]}<|end_of_text|>

<|assistant|>
{example["completion"]}<|end_of_text|>"""
        elif "input" in example and "output" in example:
            # Another alternative format
            formatted_text = f"""<|begin_of_text|><|user|>
{example["input"]}<|end_of_text|>

<|assistant|>
{example["output"]}<|end_of_text|>"""
        else:
            # If the format doesn't match expected patterns, log a sample on first process
            if rank == 0 and "logged_sample" not in globals():
                print("Dataset format doesn't match expected patterns. Sample:", example)
                globals()["logged_sample"] = True
            
            # Use a default approach - assume first column is input, second is output
            keys = list(example.keys())
            if len(keys) >= 2:
                formatted_text = f"""<|begin_of_text|><|user|>
{example[keys[0]]}<|end_of_text|>

<|assistant|>
{example[keys[1]]}<|end_of_text|>"""
            else:
                # Fallback for single-field datasets
                formatted_text = f"""<|begin_of_text|><|user|>
Generate a thoughtful response<|end_of_text|>

<|assistant|>
{example[keys[0]]}<|end_of_text|>"""
        
        return {"formatted_text": formatted_text}
    
    # Apply formatting to the datasets
    train_dataset = training_dataset.map(format_instruction)
    val_dataset = testing_dataset.map(format_instruction)
    
    # Set up training arguments for FSDP
    output_dir = f"Llama-3.1-{new_model}"
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.test_batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=10,
        logging_dir="./logs",
        evaluation_strategy="epoch" if args.run_validation else "no",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True if args.run_validation else False,
        ddp_find_unused_parameters=False,
        
        # FSDP specific settings
        fsdp="full_shard auto_wrap",  # This enables FSDP2
        fsdp_transformer_layer_cls_to_wrap="LlamaDecoderLayer",  # Specify layers to wrap
        fsdp_config={
            "fsdp_transformer_layer_cls_to_wrap": "LlamaDecoderLayer",
            "fsdp_offload_params": "false",  # CPU offload disabled
            "fsdp_state_dict_type": "full",  # For easier saving/loading
            "fsdp_backward_prefetch": "backward_pre",  # Better memory efficiency
            "fsdp_sharding_strategy": 2,  # SHARD_GRAD_OP = 2 for Zero2 (what you asked for)
        },
        
        # Additional training settings
        bf16=True,  # Use bfloat16 for training
        bf16_full_eval=True,
        gradient_checkpointing=True,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=2,
        optim="adamw_torch_fused",  # Use fused optimizer when available
        lr_scheduler_type="cosine",
    )
    
    # Create the SFT Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset if args.run_validation else None,
        peft_config=peft_config,
        dataset_text_field="formatted_text",
        max_seq_length=512,
        packing=False,
    )
    
    # Train the model
    trainer.train()
    
    # Save the final model
    if args.save_model and rank == 0:
        trainer.save_model(f"{output_dir}-final")
        print(f"Model saved to {output_dir}-final")
    
    # Clean up distributed environment
    cleanup()

if __name__ == '__main__':
    import argparse
    import sys
    
    # Training settings
    parser = argparse.ArgumentParser(description='QLoRA FSDP Training for Llama 3.1')
    parser.add_argument('--batch-size', type=int, default=2, metavar='N',
                        help='input batch size for training (default: 2)')
    parser.add_argument('--test-batch-size', type=int, default=2, metavar='N',
                        help='input batch size for testing (default: 2)')
    parser.add_argument('--epochs', type=int, default=2, metavar='N',
                        help='number of epochs to train (default: 2)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--run_name', type=str, default="",
                        help='Optional name for the run')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--run_validation', action='store_false', default=True,
                        help='running the validation')
    parser.add_argument('--save-model', action='store_false', default=True,
                        help='For saving the current Model')
    
    # Use parse_known_args for Kaggle compatibility
    args, unknown = parser.parse_known_args()
    
    # Call the main function
    sft_main(args)
