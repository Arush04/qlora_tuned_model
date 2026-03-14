import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,"\
    "roundup_power2_divisions:[32:256,64:128,256:64,>:32]"
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType
from torch.distributed.fsdp import fully_shard, FSDPModule, MixedPrecisionPolicy
from peft.tuners.lora import LoraLayer
import sys
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

def remove_patched_module(package_name):
    modules_to_delete = [
        name for name in sys.modules
        if name == package_name or name.startswith(package_name + ".")
    ]
    for name in modules_to_delete: del sys.modules[name]

def tokenize(x, tokenizer):
        return tokenizer(
            x["text"],
            truncation=True,
            max_length=2048,
            padding="max_length",
        )

def main():
    max_seq_length = 2048
    # torch.set_default_dtype(torch.float16)
    model_name = "unsloth/meta-Llama-3.1-8B-Instruct-bnb-4bit"
    dtype = torch.float16

    bnb_config = BitsAndBytesConfig(
        load_in_4bit              = True,
        bnb_4bit_use_double_quant = True,
        bnb_4bit_quant_type       = "nf4",
        bnb_4bit_compute_dtype    = dtype,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map = None,
        attn_implementation = "sdpa",
        quantization_config = bnb_config,
    )
    # freeze base weights
    for p in model.parameters():
        p.requires_grad = False
        
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "right"
        
    lora_config = LoraConfig(
        r = 64,
        lora_alpha = 128,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        lora_dropout = 0,
        bias = "none",
        task_type = TaskType.CAUSAL_LM,
    )

    # Get LoRA and setup model
    model = get_peft_model(model, lora_config)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.to(torch.bfloat16)
    # with torch.no_grad():
    #     for name, param in model.named_parameters():
    #         if ".lora_A." in name or ".lora_B." in name: param.requires_grad_(True)
    #         else: param.requires_grad_(False)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # FSDP sharding submodules and the root model
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
    )
    base_model = model.get_base_model()
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            fully_shard(module, mp_policy=mp_policy)
    # for layer in base_model.model.layers:
    #     fully_shard(layer, mp_policy=mp_policy)
    # fully_shard(model, mp_policy=mp_policy)

    url = "https://huggingface.co/datasets/laion/OIG/resolve/main/unified_chip2.jsonl"
    dataset = load_dataset("json", data_files = {"train" : url}, split = "train[:10%]")

    dataset = dataset.map(
        tokenize,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=dataset.column_names,
    )
    
    trainer = SFTTrainer(
        model = model,
        train_dataset = dataset,
        processing_class = tokenizer,
        args = SFTConfig(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 1,
            max_steps = 10,
            logging_steps = 1,
            output_dir = "outputs",
            seed = 3407,
            max_length = max_seq_length,
            fp16 = model.get_input_embeddings().weight.dtype == torch.float16,
            bf16 = model.get_input_embeddings().weight.dtype == torch.bfloat16,
            report_to = "none", # For W&B
            dataset_num_proc = 1,
        ),
    )
    trainer.train()

if __name__ == "__main__":
    try:
        dist.init_process_group("nccl")
        # remove_patched_module("trl")
        remove_patched_module("transformers")
        remove_patched_module("peft")
        # remove_patched_module("bitsandbytes")
        main()
    finally:
        if dist.is_initialized():
                dist.destroy_process_group()
