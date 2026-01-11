import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
warnings.filterwarnings("ignore")

MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"

# Load tokenizer & model (CPU only)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float32,
    device_map="cpu"
)

model.eval()

# Conversation memory


print("Catalina Chatbot based on SMolLM2 (type 'exit' to quit)\n")

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in {"exit", "quit"}:
        break
    
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant named Catalina. Answer directly and precisely with no emotion."},    
    ]

    my_message = {"role": "user", "content": user_input}
    messages.append(my_message)
    
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )

    attention_mask = torch.ones_like(input_ids)

    # Generate (NO temperature because do_sample=False)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=512,
            do_sample=False,           # deterministic
            pad_token_id=tokenizer.eos_token_id
        )

    # Get only the assistant's reply
    generated_ids = output_ids[0][input_ids.shape[-1]:]
    assistant_reply = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True
    ).strip()

    print(f"\nSmolLM:\n{assistant_reply}\n")

    # Save assistant reply to memory
    # messages.append({"role": "assistant", "content": assistant_reply})
