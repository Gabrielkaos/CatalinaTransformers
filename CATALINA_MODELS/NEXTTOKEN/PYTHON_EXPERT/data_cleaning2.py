from unidecode import unidecode
from datasets import load_dataset
import torch
from collections import OrderedDict
import tiktoken
from tqdm import tqdm


def tokenize_with_tiktoken(text, max_length=None):
    
    tokenizer = tiktoken.get_encoding("cl100k_base")
    token_ids = tokenizer.encode(text)
    
    if max_length:
        token_ids = token_ids[:max_length]
    
    return token_ids


def parse_instruction_prompt(prompt_text):
    
    parts = {
        "instruction": "",
        "input": "",
        "output": ""
    }
    
    
    sections = prompt_text.split("###")
    
    for section in sections:
        section = section.strip()
        
        if section.startswith("Instruction:"):
            parts["instruction"] = section.replace("Instruction:", "").strip()
        elif section.startswith("Input:"):
            parts["input"] = section.replace("Input:", "").strip()
        elif section.startswith("Output:"):
            parts["output"] = section.replace("Output:", "").strip()
    
    return parts


def format_for_training(instruction, input_text, output):
    
    
    
    if input_text:
        formatted = (
            f"<|instruction|>{instruction}\n"
            f"<|input|>{input_text}\n"
            f"<|response|>{output}<|endoftext|>"
        )
    else:
        formatted = (
            f"<|instruction|>{instruction}\n"
            f"<|response|>{output}<|endoftext|>"
        )
    
    return formatted


def format_simple(instruction, input_text, output):
    
    if input_text:
        formatted = (
            f"Instruction: {instruction}\n"
            f"Input: {input_text}\n"
            f"Response: {output}\n\n"
        )
    else:
        formatted = (
            f"Instruction: {instruction}\n"
            f"Response: {output}\n\n"
        )
    
    return formatted


def format_chat_style(instruction, input_text, output):
    
    if input_text:
        formatted = (
            f"<|user|>\n{instruction}\n{input_text}<|end|>\n"
            f"<|assistant|>\n{output}<|end|>\n"
        )
    else:
        formatted = (
            f"<|user|>\n{instruction}<|end|>\n"
            f"<|assistant|>\n{output}<|end|>\n"
        )
    
    return formatted


def process_instruction_dataset_tiktoken(
    dataset_name="flytech/python-codes-25k",
    max_seq_len=1024,
    format_style="alpaca",  
    split="train"
):
    
    print(f"\n{'='*60}")
    print("Processing Instruction Dataset with TikToken")
    print(f"{'='*60}\n")
    
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split=split)
    
    tokenizer = tiktoken.get_encoding("cl100k_base")
    
    sequences = []
    skipped = 0
    
    print(f"\nProcessing {len(dataset)} examples...")
    for idx, data in enumerate(tqdm(dataset)):
        try:
            if format_style == "others":
                instruction = data["instruction"]
                output = data["output"]
                formatted_text = format_simple(
                    instruction,None,output
                )
            else:
                
                parsed = parse_instruction_prompt(data["prompt"])
                
                
                if format_style == "alpaca":
                    formatted_text = format_for_training(
                        parsed["instruction"],
                        parsed["input"],
                        parsed["output"]
                    )
                elif format_style == "simple":
                    formatted_text = format_simple(
                        parsed["instruction"],
                        parsed["input"],
                        parsed["output"]
                    )
                elif format_style == "chat":
                    formatted_text = format_chat_style(
                        parsed["instruction"],
                        parsed["input"],
                        parsed["output"]
                    )
                else:
                    raise ValueError(f"Unknown format_style: {format_style}")
            
            
            token_ids = tokenizer.encode(formatted_text)
            
            
            
            if len(token_ids) > max_seq_len:
                skipped += 1
                continue
            
            
            padded = token_ids + [tokenizer.eot_token] * (max_seq_len - len(token_ids))
            sequences.append(padded)
            
            
            if idx == 0:
                print(f"\n{'='*60}")
                print("EXAMPLE FORMATTED TEXT:")
                print(f"{'='*60}")
                print(formatted_text[:500])
                print(f"{'='*60}")
                print(f"Token count: {len(token_ids)}")
                print(f"{'='*60}\n")
        
        except Exception as e:
            print(f"Error processing example {idx}: {e}")
            skipped += 1
            continue
    
    print(f"\n{'='*60}")
    print("PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total examples processed: {len(sequences)}")
    print(f"Examples skipped (too long): {skipped}")
    print(f"Vocabulary size: {tokenizer.n_vocab}")
    print(f"Max sequence length: {max_seq_len}")
    print(f"{'='*60}\n")
    
    
    x = torch.tensor(sequences, dtype=torch.long)
    
    return x, tokenizer



if __name__ == "__main__":
    
    x, tokenizer = process_instruction_dataset_tiktoken(
        dataset_name="flytech/python-codes-25k",
        max_seq_len=256,
        format_style="others",  
        split="train"
    )
    
    
    torch.save({
        "x": x,
        "tokenizer_name": "cl100k_base",  
        "vocab_size": tokenizer.n_vocab,
        "format_style": "simple"
    }, "data_tiktoken.pth")
    
    

    
    
    data = torch.load("data_tiktoken.pth")
    print(f"\nTikToken Data:")
    print(f"  Sequences: {data['x'].shape}")
    print(f"  Vocab size: {data['vocab_size']}")
    print(f"  Sequence shape: {data['x'][0].shape}")
    
    
    tokenizer = tiktoken.get_encoding("cl100k_base")
    first_seq = data['x'][0]
    
    first_seq_clean = first_seq[first_seq != tokenizer.eot_token]
    decoded = tokenizer.decode(first_seq_clean.tolist())
    
    print(f"\nFirst sequence decoded:")
    print("-" * 60)
    print(decoded[:500])
    print("-" * 60)