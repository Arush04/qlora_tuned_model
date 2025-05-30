from functools import partial
from transformers import AutoTokenizer


def prompt_helper_function(sample):
    INTRO = "Below is a mental health question. Write a supportive response that addresses the person's concerns"
    INSTRUCTION = "Instruct: Provide empathetic counseling for the following issue."
    QUESTION = "Question:"
    RESPONSE = "Response:"
    END = "END"
    
    blurb = f"{INTRO}"
    instruction = f"{INSTRUCTION}"
    question = f"{QUESTION}\n{sample['Context']}" if sample["Context"] else None
    response = f"{RESPONSE}\n{sample['Response']}"
    end = f"{END}"
    
    parts = [part for part in [blurb, instruction, question, response, end] if part]
    formatted_prompt = "\n\n".join(parts)
    sample["text"] = formatted_prompt
    return sample

def get_max_length(model):
    conf = model.config
    max_length = None
    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(model.config, length_setting, None)
        if max_length:
            print(f"Found max length: {max_length}")
            break
    if not max_length:
        max_length = 1024
        print(f"Using default max length: {max_length}")
    return max_length

def preprocess_batch(batch, tokenizer, max_length):
    """
    Tokenizing a batch with proper padding
    """
    return tokenizer(
        batch["text"],
        max_length=max_length,
        truncation=True,
        padding="max_length",  # Add padding to ensure consistent sequence lengths
        return_tensors="pt",   # Return PyTorch tensors
    )

def preprocess_dataset(tokenizer, max_length, seed, dataset):
    """Format & tokenize it so it is ready for training
    :param tokenizer (AutoTokenizer): Model Tokenizer
    :param max_length (int): Maximum number of tokens to emit from tokenizer
    """
    
    # Add prompt to each sample
    print("Preprocessing dataset...")
    dataset = dataset.map(prompt_helper_function)
    
    # Apply preprocessing to each batch of the dataset
    preprocessing_function = partial(preprocess_batch, max_length=max_length, tokenizer=tokenizer)
    dataset = dataset.map(
        preprocessing_function,
        batched=True,
        remove_columns=['Context', 'Response'],
    )
    
    # Add labels for the trainer (needed for causal language modeling)
    dataset = dataset.map(
        lambda examples: {"labels": examples["input_ids"].copy()},
        batched=True
    )
    
    # Filter out samples that have input_ids exceeding max_length
    dataset = dataset.filter(lambda sample: len(sample["input_ids"]) <= max_length)
    
    # Shuffle dataset
    dataset = dataset.shuffle(seed=seed)
    return dataset
