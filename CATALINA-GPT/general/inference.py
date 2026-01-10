

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
        device_map="auto",
        offload_folder="offload"
    )
    
    
    model = PeftModel.from_pretrained(
        base_model, 
        model_dir,
        offload_folder="offload" 
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
            temperature=temp,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id, 
            repetition_penalty=1.1
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=skip_special)
    if not skip_special and tokenizer.eos_token in response:
        response = response.split(tokenizer.eos_token)[0]
    return response


# Main execution
if __name__ == "__main__":

    model_dir = "./feature_extract"
    base_model_name = "openai-community/gpt2"

    

    model, tokenizer = load_trained_model(model_dir, base_model_name)
    while True:
        context = input("Context:")
        if context=="quit":
            break
        question = input("Question:")
        
        prompt = (
            f"### Context:\n{context}\n\n"
            f"### Question:\n{question}\n\n"
            f"### Answer:\n"
        )
        
        response = generate_response(model, tokenizer, prompt, skip_special=False, sample=True, max_length=1024)
        # print(f"\nOutput:{response.split("### Answer:\n")[1]}")
        print(response)
        print()

    # print(f"EOS token: {tokenizer.eos_token}")
    # print(f"EOS token ID: {tokenizer.eos_token_id}")
    # print(f"PAD token: {tokenizer.pad_token}")
    # print(f"PAD token ID: {tokenizer.pad_token_id}")
    
