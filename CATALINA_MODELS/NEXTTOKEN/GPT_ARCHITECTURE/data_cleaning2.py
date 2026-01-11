from unidecode import unidecode
from datasets import load_dataset
import torch
from collections import OrderedDict
import tiktoken
from tqdm import tqdm


def tokens_to_tensor(tokens_list, word_to_index, max_sequence_length):
    indexed_sequences = [
        [word_to_index.get(token, word_to_index["<PAD>"]) for token in sequence] for sequence in tokens_list
    ]

    padded_sequences = [
        sequence + [word_to_index["<PAD>"]] * (max_sequence_length - len(sequence)) for sequence in indexed_sequences
    ]
    return torch.tensor(padded_sequences, dtype=torch.int64)


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
    tokenizer = tiktoken.get_encoding("cl100k_base")  # This encoding is similar to GPT-3.5/4.

    # Tokenize the text
    token_ids = tokenizer.encode(text)
    token_strings = [tokenizer.decode([token_id]) for token_id in token_ids]

    # Optionally truncate to max_length
    if max_length:
        token_ids = token_ids[:max_length]
        token_strings = token_strings[:max_length]

    return token_ids, token_strings



def get_unique(list1):
    unique = OrderedDict()

    for line in list1:
        for word in line:
            unique[word] = None

    return list(unique.keys())

def split_string_with_special_characters(input_str):
    char_not = ["`", "~", "!", "@", "#", "$", "%", "^", "&", "*", "(", ")", "-", "_", "+", "=", ",", "<", ">", ".", "/",
                "?", "'", ";", '"', ":", "[", "]", "{", "}", "|", "\"", "\\"]
    words = []
    current_word = ''

    for char in input_str:
        
        if char.isspace():
            if current_word:
                words.append(current_word)
                current_word = ''
            words.append(char) 
        elif char in char_not:
            if current_word:
                words.append(current_word)
                current_word = ''
            words.append(char)
        else:
            current_word += char

    if current_word:
        words.append(current_word)

    return words


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
            f"Response: ```python\n{output}\n```\n\n"
        )
    else:
        formatted = (
            f"Instruction: {instruction}\n"
            f"Response: ```python\n{output}\n```\n\n"
        )
    
    return formatted

def format_simple_non_alpaca(instruction, input_text, output):
    
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
    
    
    
    sequences = []
    skipped = 0

    """
    #special for Python Codes
    """
    
    print(f"Loading dataset: Arjun-G-Ravi/Python-codes")
    dataset = load_dataset("Arjun-G-Ravi/Python-codes", split=split)
    print(f"\nProcessing {len(dataset)} examples...")
    for idx, data in enumerate(tqdm(dataset)):
        try:
            instruction = data["question"]
            output = data["code"]
            formatted_text = format_simple(
                instruction,None,output
            )
            
            # token_ids = tokenizer.encode(formatted_text)
            _,bag_of_words = tokenize_with_tiktoken(unidecode(formatted_text.strip()))
            copy_only = bag_of_words.copy()
            
            if len(bag_of_words) > (max_seq_len - 1):
                skipped += 1
                continue
            
            
            # padded = token_ids + [tokenizer.eot_token] * (max_seq_len - len(token_ids))
            bag_of_words = bag_of_words + ["<EOS>"] + ["<PAD>"] * (max_seq_len - len(bag_of_words))
            sequences.append(bag_of_words)
            
            
            if idx == 0:
                print(f"\n{'='*60}")
                print("EXAMPLE FORMATTED TEXT:")
                print(f"{'='*60}")
                print(formatted_text[:max_seq_len])
                print(f"{'='*60}")
                print(f"Token count: {len(copy_only)}")
                print(f"{'='*60}\n")
        
        except Exception as e:
            print(f"Error processing example {idx}: {e}")
            skipped += 1
            continue

    """
    """

    """
    #special for non alpaca mathqa filtered
    """
    
    print(f"Loading dataset: dtruong46me/mathqa-python")
    dataset = load_dataset("dtruong46me/mathqa-python",split=split)
    dataset1 = load_dataset("dtruong46me/mathqa-python",split="test")
    print(f"\nProcessing {len(dataset1)} examples...")
    for idx, data in enumerate(tqdm(dataset1)):
        try:
            instruction = data["text"]
            output = data["code"]
            formatted_text = format_simple(
                instruction,None,output
            )
            
            # token_ids = tokenizer.encode(formatted_text)
            _,bag_of_words = tokenize_with_tiktoken(unidecode(formatted_text.strip()))
            copy_only = bag_of_words.copy()
            
            if len(bag_of_words) > (max_seq_len - 1):
                skipped += 1
                continue
            
            
            # padded = token_ids + [tokenizer.eot_token] * (max_seq_len - len(token_ids))
            bag_of_words = bag_of_words + ["<EOS>"] + ["<PAD>"] * (max_seq_len - len(bag_of_words))
            sequences.append(bag_of_words)
            
            
            if idx == 0:
                print(f"\n{'='*60}")
                print("EXAMPLE FORMATTED TEXT:")
                print(f"{'='*60}")
                print(formatted_text[:max_seq_len])
                print(f"{'='*60}")
                print(f"Token count: {len(copy_only)}")
                print(f"{'='*60}\n")
        
        except Exception as e:
            print(f"Error processing example {idx}: {e}")
            skipped += 1
            continue
    print(f"\nProcessing {len(dataset)} examples...")
    for idx, data in enumerate(tqdm(dataset)):
        try:
            instruction = data["text"]
            output = data["code"]
            formatted_text = format_simple(
                instruction,None,output
            )
            
            # token_ids = tokenizer.encode(formatted_text)
            _,bag_of_words = tokenize_with_tiktoken(unidecode(formatted_text.strip()))
            copy_only = bag_of_words.copy()
            
            if len(bag_of_words) > (max_seq_len - 1):
                skipped += 1
                continue
            
            
            # padded = token_ids + [tokenizer.eot_token] * (max_seq_len - len(token_ids))
            bag_of_words = bag_of_words + ["<EOS>"] + ["<PAD>"] * (max_seq_len - len(bag_of_words))
            sequences.append(bag_of_words)
            
            
            if idx == 23:
                print(f"\n{'='*60}")
                print("EXAMPLE FORMATTED TEXT:")
                print(f"{'='*60}")
                print(formatted_text[:max_seq_len])
                print(f"{'='*60}")
                print(f"Token count: {len(copy_only)}")
                print(f"{'='*60}\n")
        
        except Exception as e:
            print(f"Error processing example {idx}: {e}")
            skipped += 1
            continue

    """
    """
#-----------------------------------------------------------------
    """
    #special for non alpaca 18k
    """
    
    print(f"Loading dataset: flytech/python-codes-25k")
    dataset = load_dataset("flytech/python-codes-25k", split=split)
    print(f"\nProcessing {len(dataset)} examples...")
    for idx, data in enumerate(tqdm(dataset)):
        try:
            instruction = data["instruction"]
            output = data["output"]
            formatted_text = format_simple_non_alpaca(
                instruction,None,output
            )
            
            # token_ids = tokenizer.encode(formatted_text)
            _,bag_of_words = tokenize_with_tiktoken(unidecode(formatted_text.strip()))
            copy_only = bag_of_words.copy()
            
            if len(bag_of_words) > (max_seq_len - 1):
                skipped += 1
                continue
            
            
            # padded = token_ids + [tokenizer.eot_token] * (max_seq_len - len(token_ids))
            bag_of_words = bag_of_words + ["<EOS>"] + ["<PAD>"] * (max_seq_len - len(bag_of_words))
            sequences.append(bag_of_words)
            
            
            if idx == 0:
                print(f"\n{'='*60}")
                print("EXAMPLE FORMATTED TEXT:")
                print(f"{'='*60}")
                print(formatted_text[:max_seq_len])
                print(f"{'='*60}")
                print(f"Token count: {len(copy_only)}")
                print(f"{'='*60}\n")
        
        except Exception as e:
            print(f"Error processing example {idx}: {e}")
            skipped += 1
            continue

    """
    """

    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split=split)
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
            
            
            # token_ids = tokenizer.encode(formatted_text)
            _,bag_of_words = tokenize_with_tiktoken(unidecode(formatted_text.strip()))
            copy_only = bag_of_words.copy()
            
            if len(bag_of_words) > (max_seq_len - 1):
                skipped += 1
                continue
            
            
            # padded = token_ids + [tokenizer.eot_token] * (max_seq_len - len(token_ids))
            bag_of_words = bag_of_words + ["<EOS>"] + ["<PAD>"] * (max_seq_len - len(bag_of_words))
            sequences.append(bag_of_words)
            
            
            if idx == 0:
                print(f"\n{'='*60}")
                print("EXAMPLE FORMATTED TEXT:")
                print(f"{'='*60}")
                print(formatted_text[:max_seq_len])
                print(f"{'='*60}")
                print(f"Token count: {len(copy_only)}")
                print(f"{'='*60}\n")
        
        except Exception as e:
            print(f"Error processing example {idx}: {e}")
            skipped += 1
            continue

    vocabulary = get_unique(sequences)
    tokenizer = {token: idx for idx, token in enumerate(vocabulary)}
    
    print(f"\n{'='*60}")
    print("PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total examples processed: {len(sequences)}")
    print(f"Examples skipped (too long): {skipped}")
    print(f"Vocabulary size: {len(vocabulary)}")
    print(f"Max sequence length: {max_seq_len}")
    print(f"{'='*60}\n")
    
    
    x = tokens_to_tensor(sequences, tokenizer, max_seq_len)
    
    return x, vocabulary, tokenizer



if __name__ == "__main__":
    
    x,vocab, tokenizer = process_instruction_dataset_tiktoken(
        dataset_name="iamtarun/python_code_instructions_18k_alpaca",
        max_seq_len=256,
        format_style="simple",  
        split="train"
    )
    
    file_name = "data_triple.pth"
    torch.save({
        "x": x,
        "tokenizer": tokenizer,  
        "vocab": vocab,
    }, file_name)
    
    

    
    
    data = torch.load(file_name)
    print(f"\nTikToken Data:")
    print(f"  Sequences: {data['x'].shape}")
    print(f"  Vocab size: {len(data['vocab'])}")
    print(f"  Sequence shape: {data['x'][2].shape}")
    
    
    tokenizer = data["tokenizer"]
    first_seq = data['x'][2]

    # first_seq = first_seq[first_seq != tokenizer["<PAD>"]]
    # tokenizer = tiktoken.get_encoding("cl100k_base")
    # decoded = tokenizer.decode(first_seq.tolist())
    
    de_tokenizer = {idx:char for char,idx in tokenizer.items()}
    decoded = "".join([de_tokenizer[i.item()] for i in first_seq])

    
    print(f"\nFirst sequence decoded:")
    print("-" * 60)
    print(decoded[:500])
    print("-" * 60)
