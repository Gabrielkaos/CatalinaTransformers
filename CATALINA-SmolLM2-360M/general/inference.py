

import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer
)
from peft import PeftModel

def load_trained_model1(model_dir, base_model_name):
    
    print(f"Loading trained model from {model_dir}...")
    
    # Load tokenizer from fine-tuned model directory
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    # Load base model WITHOUT device_map initially
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        dtype=torch.float16,
        low_cpu_mem_usage=True  # Use this instead of device_map initially
    )
    
    # Resize embeddings before any device mapping
    base_model.resize_token_embeddings(len(tokenizer))
    
    # Set config values
    base_model.config.pad_token_id = tokenizer.pad_token_id
    base_model.config.eos_token_id = tokenizer.eos_token_id
    
    # NOW move to device (after resizing is complete)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = base_model.to(device)
    
    # Load the LoRA adapters
    model = PeftModel.from_pretrained(
        base_model, 
        model_dir
    )
    
    print("Model loaded successfully!")
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_length=128, skip_special=False, sample=True
                      ,temp=0.7,top_p=0.9):
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    model.eval()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id, 
            repetition_penalty=1.1,
            
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=skip_special)
    if not skip_special and tokenizer.eos_token in response:
        response = response.split(tokenizer.eos_token)[0]
    return response


# Main execution
if __name__ == "__main__":

    model_dir = "./chat"
    base_model_name = "HuggingFaceTB/SmolLM2-360M"

    model, tokenizer = load_trained_model1(model_dir, base_model_name)
    while True:
        prompt1 = input("Instruction:")
        if prompt1=="quit":
            break
        prompt2 = input("Input:")
        if not prompt2:
            prompt = f"### Instruction:\n{prompt1}\n\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{prompt1}\n\n### Input:\n{prompt2}\n\n### Response:\n"
        
        response = generate_response(model, tokenizer, prompt, skip_special=False, sample=False, max_length=512)
        print(f"\nOutput:\n{response.split("### Response:\n")[1]}")
        # print(response)
        print()

    # print(f"EOS token: {tokenizer.eos_token}")
    # print(f"EOS token ID: {tokenizer.eos_token_id}")
    # print(f"PAD token: {tokenizer.pad_token}")
    # print(f"PAD token ID: {tokenizer.pad_token_id}")
    
