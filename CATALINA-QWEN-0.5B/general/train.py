
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


def setup_model_for_chat_finetuning(model_name):
    
    
    
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


def prepare_chat_dataset(tokenizer, max_len=512, dataset_name="tatsu-lab/alpaca"):
   
    
    
    dataset = load_dataset(dataset_name, split="train")
    
    def format_chat(example):
       
        if "instruction" in example:
            
            instruction = example["instruction"]
            input_text = example.get("input", "")
            output = example["output"]
            
            if input_text:
                prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
            else:
                prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
        else:
            
            prompt = example.get("text", "")
        prompt+=tokenizer.eos_token
        return {"text": prompt}
    
    dataset = dataset.map(format_chat)
    
  
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_len
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    return tokenized_dataset


def train_chat_model(model, tokenizer, dataset, output_dir="./chat_model"):

    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=10,
        gradient_accumulation_steps=4,
        learning_rate=3e-5,
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



if __name__ == "__main__":
    print("Setting up model for chat fine-tuning...")

    model_dir = "./general_chat"
    base_model_name = "Qwen/Qwen2.5-0.5B"

    # model, tokenizer = load_trained_model(model_dir, base_model_name)

    
    
    model, tokenizer = setup_model_for_chat_finetuning(
        model_name=base_model_name
    )
    
    
    print("\nPreparing chat dataset...")
    dataset = prepare_chat_dataset(tokenizer)
    
   
    print("\nStarting fine-tuning...")
    model = train_chat_model(model, tokenizer, dataset,output_dir=model_dir)
    print("Done training.")
    
    
    # print("\nTesting generation...")
    # test_prompt = "### Instruction:\nWrite a short poem about coding.\n\n### Response:\n"
    # response = generate_response(model, tokenizer, test_prompt, max_length=50)
    # print(f"\nPrompt: {test_prompt}")
    # print(f"Response: {response}")