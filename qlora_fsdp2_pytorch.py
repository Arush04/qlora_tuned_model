import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,"\
    "roundup_power2_divisions:[32:256,64:128,256:64,>:32]"
import gc
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType
from torch.distributed.fsdp import fully_shard, FSDPModule, MixedPrecisionPolicy
from peft.tuners.lora import LoraLayer
import sys
from datasets import load_dataset
from torch.utils.data import DataLoader, DistributedSampler

def tokenize(x, tokenizer):
        return tokenizer(
            x["text"],
            truncation=True,
            max_length=1024,
            padding="max_length",
        )

def main(local_rank):
    max_seq_length = 1024
    torch.set_default_dtype(torch.float16)
    model_name = "unsloth/meta-Llama-3.1-8B-Instruct-bnb-4bit"
    dtype = torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map={"": local_rank},
        torch_dtype=dtype,
        attn_implementation = "sdpa",
        low_cpu_mem_usage=True,
    )

    # The pre-quantized model has bnb_4bit_compute_dtype=bfloat16 baked in.
    # T4 GPUs don't support bf16 natively — override to fp16.
    for module in model.modules():
        if hasattr(module, "compute_dtype"):
            module.compute_dtype = dtype
        
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
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

    if model.config.tie_word_embeddings:
        output_emb = model.get_output_embeddings()
        input_emb  = model.get_input_embeddings()
        output_emb.weight = torch.nn.Parameter(
            input_emb.weight.detach().clone()
        )
        model.config.tie_word_embeddings = False
    
    with torch.no_grad():
        for name, param in model.named_parameters():
            if ".lora_A." in name or ".lora_B." in name: param.requires_grad_(True)
            else: param.requires_grad_(False)
     
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()
    base_model = model.get_base_model()

    # FSDP sharding submodules and the root model
    mp_policy = MixedPrecisionPolicy(
        param_dtype=dtype,
        reduce_dtype=dtype,
        output_dtype=dtype,
    )

    # Only ignore non-float params (quantized int/uint8 weights) — FSDP2 can't shard those.
    # Float frozen params (e.g. embeddings) must be managed by FSDP2 to avoid device mismatches.
    non_float_params = {p for p in model.parameters() if not p.dtype.is_floating_point}

    for layer in base_model.model.layers:
        fully_shard(layer, mp_policy=mp_policy, ignored_params=non_float_params)

    fully_shard(model, mp_policy=mp_policy, ignored_params=non_float_params)

    url = "https://huggingface.co/datasets/laion/OIG/resolve/main/unified_chip2.jsonl"
    dataset = load_dataset("json", data_files = {"train" : url}, split = "train[:10%]")

    dataset = dataset.map(
        tokenize,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=dataset.column_names,
        num_proc=1,
    )

    dataset.set_format("torch")

    # Training hyperparams
    per_device_batch_size = 1
    gradient_accumulation_steps = 4
    warmup_steps = 1
    max_steps = 10
    lr = 5e-5

    sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(),
                                 rank=dist.get_rank(), shuffle=True, seed=3407)
    dataloader = DataLoader(dataset, batch_size=per_device_batch_size,
                            sampler=sampler, pin_memory=False)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )

    # Linear warmup then constant LR
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    gc.collect()
    torch.cuda.empty_cache()

    model.train()
    step = 0
    optimizer.zero_grad()

    while step < max_steps:
        for batch in dataloader:
            input_ids = batch["input_ids"].to(f"cuda:{local_rank}")
            attention_mask = batch["attention_mask"].to(f"cuda:{local_rank}")
            
            labels[attention_mask == 0] = -100  # ignore padding
            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                            labels=labels)
            loss = outputs.loss / gradient_accumulation_steps
            loss.backward()

            if (step + 1) % gradient_accumulation_steps == 0 or step == max_steps - 1:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            step += 1
            if local_rank == 0:
                print(f"step {step}/{max_steps}  loss={loss.item() * gradient_accumulation_steps:.4f}")

            if step >= max_steps:
                break

if __name__ == "__main__":
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    try:
        dist.init_process_group("nccl")
        main(local_rank)
    finally:
        if dist.is_initialized():
                dist.destroy_process_group()
