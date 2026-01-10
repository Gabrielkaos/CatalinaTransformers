
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from peft import PeftModel
from datasets import concatenate_datasets


def setup_model_for_finetuning(model_name):
    
    
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.padding_side = "right"
    
    
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
   
    lora_config = LoraConfig(
        r=16,  
        lora_alpha=32, 
        target_modules=[
            "q_proj",
            "k_proj", 
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ],  
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer


def prepare_math_dataset(tokenizer, max_len=512):
   
    quadratic = load_dataset("arnoudbuzing/quadratic-equation-training", split="train[:5000]") # Example data: Solve[96 + 64*u + 26*u^2 == -7, u] == {{u -> (-32 - I*Sqrt[1654])/26}, {u -> (-32 + I*Sqrt[1654])/26}}
    cubic = load_dataset("arnoudbuzing/cubic-equation-training", split="train[:7000]") # Example dataa: Solve[43 + 69*y + 17*y^2 - 9*y^3 == 15, y] == {{y -> 4}, {y -> (-19 - Sqrt[109])/18}, {y -> (-19 + Sqrt[109])/18}}
    linear = load_dataset("arnoudbuzing/linear-equation-training", split="train[:3000]") # Example data: Solve[-3 - 7*k == 48, k] == {{k -> -51/7}}

    combined = concatenate_datasets([linear, quadratic, cubic]).shuffle(32)
    
    def format_example(example):
       
        if "text" in example:
            example_text = example["text"]
            if "^2" in example_text:
                prompt = f"Topic: Quadratic\n{example_text}"
            elif "^3" in example_text:
                prompt = f"Topic: Cubic\n{example_text}"
            else:
                prompt = f"Topic: Linear\n{example_text}" 
        else:
            prompt = example.get("text", "")
        prompt+=tokenizer.eos_token
        print(prompt)
        exit()
        return {"text": prompt}
    
    dataset = combined.map(format_example)
    
  
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_len,
            padding=False
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    return tokenized_dataset


def train_model(model, tokenizer, dataset, output_dir="./chat_model"):

    
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=4,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=1.2e-4,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        warmup_steps=100,
        optim="paged_adamw_8bit",  
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    
    print("Starting training...")
    trainer.train()
    
    print(f"Saving model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    return model


def load_trained_model(model_dir="./chat_model", base_model_name="Qwen/Qwen2.5-0.5B"):
    
    print(f"Loading trained model from {model_dir}...")
    
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        dtype=torch.float16,
        device_map="auto",
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

def generate_response(model, tokenizer, prompt, max_length=128, skip_special=True, sample=True
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
            repetition_penalty=1.1,
            no_repeat_ngram_size=3
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=skip_special)
    if not skip_special and tokenizer.eos_token in response:
        response = response.split(tokenizer.eos_token)[0]
    return response

if __name__ == "__main__":
    print("Setting up model for fine-tuning...")

    model_dir = "./algebra"
    base_model_name = "Qwen/Qwen2.5-0.5B"

    # model, tokenizer = load_trained_model(model_dir, base_model_name)

    model, tokenizer = setup_model_for_finetuning(
        model_name=base_model_name
    )
    
    print("\nPreparing chat dataset...")
    dataset = prepare_math_dataset(tokenizer)

    print(dataset[0])
    
   
    # print("\nStarting fine-tuning...")
    # model = train_model(model, tokenizer, dataset,output_dir=model_dir)
    # print("\nDone training.")
    
    
    # print("\nTesting generation...")
    # test_prompt = "Solve for x, just generate the answer don't explain:\n2(x + 2) = 4\nAnswer:"
    # response = generate_response(model, tokenizer, test_prompt, max_length=128, sample=False)
    # print(f"\nPrompt: {test_prompt}")
    # print(f"Response: {response}")