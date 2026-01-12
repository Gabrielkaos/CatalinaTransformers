from unidecode import unidecode
from datasets import load_dataset
import torch
from collections import OrderedDict
import tiktoken
from tqdm import tqdm


def tokenize_with_tiktoken(text, max_length=None):
    """
    Tokenizes the input text using a tokenizer similar to ChatGPT's tokenizer (tiktoken).

    Args:
        text (str): The input text to be tokenized.
        max_length (int, optional): Maximum number of tokens to return. If None, return all tokens.

    Returns:
        list[int]: List of token IDs.
        list[str]: List of corresponding token strings.
    """
    # Load the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")  # This encoding is similar to GPT-3.5/4.

    # Tokenize the text
    token_ids = tokenizer.encode(text)
    token_strings = [tokenizer.decode([token_id]) for token_id in token_ids]

    # Optionally truncate to max_length
    if max_length:
        token_ids = token_ids[:max_length]
        token_strings = token_strings[:max_length]

    return token_ids, token_strings


def format_simple(instruction, input_text, output):
    
    if input_text:
        formatted = (
            f"Instruction:\n{instruction}\n\n"
            f"Input:\n{input_text}\n\n"
            f"Response:\n{output}"
        )
    else:
        formatted = (
            f"Instruction:\n{instruction}\n\n"
            f"Response:\n{output}"
        )
    
    return formatted



def process_instruction_dataset_tiktoken(
    dataset_name,
    max_seq_len=1024,
    split="train"
):
    
    print(f"\n{'='*60}")
    print("Processing Instruction Dataset")
    print(f"{'='*60}\n")
    
    tokenizer = tiktoken.get_encoding("gpt2")
    
    sequences = []
    skipped = 0


    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split=split)
    print(f"\nProcessing {len(dataset)} examples...")
    for idx, data in enumerate(tqdm(dataset)):
        
        formatted_text = format_simple(
            data["instruction"],
            data["input"],
            data["output"]
        )
        
        input_tokens = tokenizer.encode(formatted_text)
        
        
        if len(input_tokens) >= max_seq_len:
            skipped += 1
            continue

        input_tokens = input_tokens + [50256] * (max_seq_len - len(input_tokens))
        
        sequences.append(input_tokens)

        if len(sequences) >= 15_000:
            break
        

        

    
    print(f"\n{'='*60}")
    print("PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total examples processed: {len(sequences)}")
    print(f"Examples skipped (too long): {skipped}")
    print(f"Max sequence length: {max_seq_len}")
    print(f"{'='*60}\n")
    
    
    x = torch.tensor(sequences,dtype=torch.int64)
    
    return x



if __name__ == "__main__":
    
    x = process_instruction_dataset_tiktoken(
        dataset_name="yahma/alpaca-cleaned",
        max_seq_len=256, 
        split="train"
    )
    
    file_name = "data.pth"
    torch.save({
        "x": x
    }, file_name)

    print(x.shape)
    
    print("Done")