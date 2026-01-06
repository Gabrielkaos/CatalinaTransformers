from unidecode import unidecode
import torch
from tqdm import tqdm
import random


def tokens_to_tensor(tokens_list, word_to_index, max_sequence_length):
    indexed_sequences = [
        [word_to_index.get(token, word_to_index["<PAD>"]) for token in sequence] for sequence in tokens_list
    ]

    padded_sequences = [
        sequence + [word_to_index["<PAD>"]] * (max_sequence_length - len(sequence)) for sequence in indexed_sequences
    ]
    return torch.tensor(padded_sequences, dtype=torch.int64)



def process_instruction_dataset_character_level(
    max_seq_len=512
):
    
    print(f"\n{'='*60}")
    print("Processing Instruction Dataset Character Level")
    print(f"{'='*60}\n")
    
    
    
    sequences = []
    skipped = 0

 

    with open("dataset/data.txt","r") as f:
        full_text = f.read()
    print(f"\nProcessing...")
    i = 0
    while True:
        offset = random.randint(0,len(full_text)-max_seq_len)

        inputs = full_text[offset:offset+max_seq_len]
        

        bag_of_words = list(unidecode(inputs))

        if len(bag_of_words) > max_seq_len:
                skipped += 1
                continue
        
        sequences.append(bag_of_words)

        if len(sequences) >= 50_000:
             break
            
            
        if i == 0:
            print(f"\n{'='*60}")
            print("EXAMPLE TEXT:")
            print(f"{'='*60}")
            print(inputs)
            print(f"{'='*60}")
            print(f"Token count: {len(inputs)}")
            print(f"{'='*60}\n")
        i+=1

    
    vocabulary = sorted(list(set(full_text)))
    vocabulary.append("<PAD>")
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
    
    x,vocab, tokenizer = process_instruction_dataset_character_level(
        max_seq_len=512
    )
    
    file_name = "data.pth"
    torch.save({
        "x": x,
        "tokenizer": tokenizer,  
        "vocab": vocab,
    }, file_name)
    
    

    
    
    data = torch.load(file_name)
    print(f"  Sequences: {data['x'].shape}")
    print(f"  Vocab:{data["vocab"]}")
    print(f"  Vocab size: {len(data['vocab'])}")
    print(f"  Sequence shape: {data['x'][2].shape}")
    
    
    tokenizer = data["tokenizer"]
    first_seq = data['x'][2]

    
    de_tokenizer = {idx:char for char,idx in tokenizer.items()}
    decoded = "".join([de_tokenizer[i.item()] for i in first_seq])

    
    print(f"\nFirst sequence decoded:")
    print("-" * 60)
    print(decoded[:500])
    print("-" * 60)
