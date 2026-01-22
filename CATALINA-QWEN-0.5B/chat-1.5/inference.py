

import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer
)
from peft import PeftModel

def load_trained_model(model_dir, base_model_name):
    
    print(f"Loading trained model from {model_dir}...")
    
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True,
        offload_folder="offload"
    )
    
    model = PeftModel.from_pretrained(
        base_model, 
        model_dir,
        offload_folder="offload" 
    )
    
    print("Model loaded successfully!")
    return model, tokenizer


def generate_response_with_thinking(model, tokenizer, prompt, max_length=128):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    model.eval()
    input_ids = inputs["input_ids"]
    
    # Track generation state
    generated_tokens = []
    in_thinking = False
    current_text = ""
    
    with torch.no_grad():
        past_key_values = None
        current_input_ids = input_ids
        
        for _ in range(max_length):
            outputs = model(
                input_ids=current_input_ids,
                past_key_values=past_key_values,
                use_cache=True
            )
            
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]
            
            # # Apply temperature and top-p sampling
            # logits = logits / temp
            # sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            # cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            # sorted_indices_to_remove = cumulative_probs > top_p
            # sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            # sorted_indices_to_remove[..., 0] = 0
            # indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            # logits[indices_to_remove] = float('-inf')
            
            # probs = torch.softmax(logits, dim=-1)
            # next_token = torch.multinomial(probs, num_samples=1)

            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            generated_tokens.append(next_token[0].item())
            
            # Check for EOS
            if next_token[0].item() == tokenizer.eos_token_id:
                break
            
            # Decode incrementally to check for tags
            current_text = tokenizer.decode(generated_tokens, skip_special_tokens=False)
            
            # Check for thinking state changes
            if '<think>' in current_text and not in_thinking:
                in_thinking = True
                print("\n[Thinking]: ", end="", flush=True)
            
            if in_thinking:
                # Print only the new token
                token_text = tokenizer.decode([next_token[0].item()], skip_special_tokens=False)
                print(token_text, end="", flush=True)
                
                # if '</think>' in current_text:
                #     in_thinking = False
                #     print("\n\n[Response]: ", end="", flush=True)
            elif '<think>' not in current_text:
                # Print response tokens (outside of thinking)
                token_text = tokenizer.decode([next_token[0].item()], skip_special_tokens=False)
                print(token_text, end="", flush=True)
            
            current_input_ids = next_token
    
    print()  # Final newline
    
    # Extract just the response part (after </think>)
    full_response = tokenizer.decode(generated_tokens, skip_special_tokens=False)
    if '</think>' in full_response:
        response_part = full_response.split('</think>')[-1].strip()
        if '<|endoftext|>' in response_part:
            response_part = response_part.split('<|endoftext|>')[0].strip()
        return response_part
    
    return full_response

def generate_response(model, tokenizer, prompt, max_length=128, skip_special=False, sample=True
                      ,temp=0.7,top_p=0.9):
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    model.eval()
    input_ids = inputs["input_ids"]
    input_len = input_ids.shape[1]
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=sample,
            temperature=temp,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id, 
            repetition_penalty=1.1,
            
        )
    
    outputs = outputs[0][input_len:]
    response = tokenizer.decode(outputs, skip_special_tokens=skip_special)
    if not skip_special and tokenizer.eos_token in response:
        response = response.split(tokenizer.eos_token)[0]
    return response


# Main execution
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    model_dir = "./chat"
    base_model_name = "Qwen/Qwen2.5-1.5B"

    model, tokenizer = load_trained_model(model_dir, base_model_name)
    while True:
        prompt1 = input("Instruction:")
        if prompt1=="quit":
            break
        prompt = f"### Instruction:\n{prompt1}\n\n### Response:\n"
        
        # response = generate_response(model, tokenizer, prompt, skip_special=False, sample=False, max_length=8196)
        response = generate_response_with_thinking(model, tokenizer, prompt, max_length=8196)
        # print(f"\nOutput:\n{response}")
        print()

    # print(f"EOS token: {tokenizer.eos_token}")
    # print(f"EOS token ID: {tokenizer.eos_token_id}")
    # print(f"PAD token: {tokenizer.pad_token}")
    # print(f"PAD token ID: {tokenizer.pad_token_id}")
    
