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
# from functools import partial

# def create_prompt_formats(sample):
#     """
#     Format various fields of the sample ('instruction','output')
#     Then concatenate them using two newline characters 
#     :param sample: Sample dictionnary
#     """
#     INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
#     INSTRUCTION_KEY = "### Instruct: Summarize the below conversation."
#     RESPONSE_KEY = "### Output:"
#     END_KEY = "### End"

#     blurb = f"\n{INTRO_BLURB}"
#     instruction = f"{INSTRUCTION_KEY}"
#     input_context = f"{sample['dialogue']}" if sample["dialogue"] else None
#     response = f"{RESPONSE_KEY}\n{sample['summary']}"
#     end = f"{END_KEY}"

#     parts = [part for part in [blurb, instruction, input_context, response, end] if part]

#     formatted_prompt = "\n\n".join(parts)
#     sample["text"] = formatted_prompt

#     return sample

# SOURCE https://github.com/databrickslabs/dolly/blob/master/training/trainer.py
# def get_max_length(model):
#     conf = model.config
#     max_length = None
#     for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
#         max_length = getattr(model.config, length_setting, None)
#         if max_length:
#             print(f"Found max lenth: {max_length}")
#             break
#     if not max_length:
#         max_length = 1024
#         print(f"Using default max length: {max_length}")
#     return max_length


# def preprocess_batch(batch, tokenizer, max_length):
#     """
#     Tokenizing a batch
#     """
#     return tokenizer(
#         batch["text"],
#         max_length=max_length,
#         truncation=True,
#     )

# SOURCE https://github.com/databrickslabs/dolly/blob/master/training/trainer.py
# def preprocess_dataset(tokenizer: AutoTokenizer, max_length: int,seed, dataset):
#     """Format & tokenize it so it is ready for training
#     :param tokenizer (AutoTokenizer): Model Tokenizer
#     :param max_length (int): Maximum number of tokens to emit from tokenizer
#     """

#     # Add prompt to each sample
#     print("Preprocessing dataset...")
#     dataset = dataset.map(prompt_helper_function)#, batched=True)

#     # Apply preprocessing to each batch of the dataset & and remove 'instruction', 'context', 'response', 'category' fields
#     _preprocessing_function = partial(preprocess_batch, max_length=max_length, tokenizer=tokenizer)
#     dataset = dataset.map(
#         _preprocessing_function,
#         batched=True,
#         remove_columns=['Context', 'Response'],
#     )

#     # Filter out samples that have input_ids exceeding max_length
#     dataset = dataset.filter(lambda sample: len(sample["input_ids"]) < max_length)

#     # Shuffle dataset
#     dataset = dataset.shuffle(seed=seed)

#     return dataset

def create_prompt(sample):
    prompt = (
        "### Context:\n"
        f"{sample['Context']}\n\n"
        "### Response:\n"
        f"{sample['Response']}"
    )
    return {"text": prompt}

def tokenize_function(batch, tokenizer, max_length=64):
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

    base_model = "unsloth/Meta-Llama-3.1-8B-Instruct"

    # ---- BitsAndBytesConfig for FSDP2 compatibility ----
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",  # or "fp4"
        bnb_4bit_compute_dtype=torch.float16,  # or bfloat16
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
            reduce_dtype=torch.float16,
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
    dataset_train = dataset_train.map(create_prompt)
    dataset_validation = dataset_validation.map(create_prompt)
    dataset_test = dataset_test.map(create_prompt)

    def tokenize_batch(batch):
        return tokenize_function(batch, tokenizer, max_length=64)

    dataset_train = dataset_train.map(tokenize_batch, batched=True, remove_columns=['Context', 'Response'])
    dataset_validation = dataset_validation.map(tokenize_batch, batched=True, remove_columns=['Context', 'Response'])
    dataset_test = dataset_test.map(tokenize_batch, batched=True, remove_columns=['Context', 'Response'])

    # ---- DataLoader ----
    batch_size = 1
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # ---- Training Loop ----
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
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
        torch.cuda.empty_cache()
        if step % 10 == 0 and rank == 0:
            print(f"Step {step}, Loss: {loss.item()}")
        if step == 20:  # For demonstration, stop after 20 steps
            break

    dist.destroy_process_group()

if __name__ == "__main__":
    main()