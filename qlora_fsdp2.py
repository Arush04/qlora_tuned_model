from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    GenerationConfig
)
from tqdm import tqdm
from trl import SFTTrainer
import torch
import time
import pandas as pd
import numpy as np
from helper_functions import *
from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig

# setting up configs for FSDP2

fsdp_plugin = FullyShardedDataParallelPlugin(
    fsdp_version=2,
    state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
    optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
    cpu_offload=True,
    activation_checkpointing=True,
    
)
accelerator = Accelerator(fsdp_plugin=fsdp_plugin)

# Loading the dataset
dataset = load_dataset("Amod/mental_health_counseling_conversations")
train_test_split = dataset["train"].train_test_split(test_size=0.3, seed=42)
test_validation_split = train_test_split["test"].train_test_split(test_size=1/3, seed=42)

dataset_train = train_test_split["train"]
dataset_validation = test_validation_split["train"]
dataset_test = test_validation_split["test"]

print("training dataset ", len(dataset_train))
print("validation dataset ", len(dataset_validation))
print("test dataset ", len(dataset_test))

# Creating Quantization Configuration
compute_dtype = getattr(torch, "float16")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

# Loading the model and tokenizer
base_model = "microsoft/phi-2"
device_map = {"": 0}
model = AutoModelForCausalLM.from_pretrained(base_model, device_map=device_map,quantization_config=bnb_config,trust_remote_code=True,use_auth_token=True)
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, padding_side="left", add_eos_token=True, add_bos_token=True, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

max_length = get_max_length(model)
print(max_length)
seed =42
dataset_train_pre = preprocess_dataset(tokenizer, max_length,seed, dataset_train)
dataset_validation_pre = preprocess_dataset(tokenizer, max_length,seed, dataset_validation)

model = accelerator.prepare_model(model)

# Enable model parallelism if multiple GPUs are available
if torch.cuda.device_count() > 1:
    model.is_parallelizable = True
    model.model_parallel = True

# Training
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

peft_model = get_peft_model(model, config)


peft_training_args = TrainingArguments(
    output_dir = output_dir,
    warmup_steps=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    max_steps=1000,
    learning_rate=2e-4,
    optim="paged_adamw_8bit",
    logging_steps=25,
    logging_dir="./logs",
    save_strategy="steps",
    save_steps=25,
    evaluation_strategy="steps",
    eval_steps=100,
    do_eval=True,
    gradient_checkpointing=True,
    report_to="none",
    overwrite_output_dir = 'True',
    group_by_length=True,
)

peft_model.config.use_cache = False
peft_trainer = transformers.Trainer(
    model=peft_model,
    train_dataset=dataset_train_pre,
    eval_dataset=dataset_validation_pre,
    args=peft_training_args,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
peft_trainer.train()
