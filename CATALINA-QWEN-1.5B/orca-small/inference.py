

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

    model_dir = "./orca"
    base_model_name = "Qwen/Qwen2.5-1.5B"

    model, tokenizer = load_trained_model(model_dir, base_model_name)

    system = "Your name is Catalina. A helpful Large Language Model based on Qwen2.5 finetuned by Gab. You give detailed and long answers."

    while True:
        human = input("Prompt:")
        if human=="quit":
            break
        prompt = (
            f"### System:\n{system}\n\n"
            f"### Instruction:\n{human}\n\n"
            f"### Response:\n"
        )
        
        response = generate_response(model, tokenizer, prompt, skip_special=False, sample=False, max_length=8196)
        # response = generate_response_with_thinking(model, tokenizer, prompt, max_length=8196)
        print(f"\nOutput:\n{response}")
        print()

    # print(f"EOS token: {tokenizer.eos_token}")
    # print(f"EOS token ID: {tokenizer.eos_token_id}")
    # print(f"PAD token: {tokenizer.pad_token}")
    # print(f"PAD token ID: {tokenizer.pad_token_id}")
    
