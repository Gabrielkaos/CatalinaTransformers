import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


model_name = "Qwen/Qwen2.5-Coder-0.5B-Instruct"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None,
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name)


def generate_response(messages, max_new_tokens=2048, temperature=0.7, top_p=0.9):
    
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
   
    inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)
    
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
  
    generated_ids = [
        output_ids[len(inputs.input_ids[i]):] 
        for i, output_ids in enumerate(outputs)
    ]
    
   
    decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return decoded[0]



while True:
    prompt = input("User: ")
    if prompt.lower() in ["exit", "quit"]:
        break
    messages = [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": prompt}
    ]
    response = generate_response(messages)
    print(f"\n{response}\n")
