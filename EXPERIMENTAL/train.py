"""
PRACTICAL APPROACH: Use HuggingFace models with LoRA/QLoRA fine-tuning
This is much easier than converting weights and works on limited hardware
"""

import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import bitsandbytes as bnb


def setup_model_for_chat_finetuning(
    model_name="Qwen/Qwen2.5-1.5B",  # or "mistralai/Mistral-7B-v0.1"
    use_4bit=True  # Use 4-bit quantization for lower memory
):
    """
    Load a pretrained model and prepare it for efficient fine-tuning
    This works on GPUs with as little as 8-16GB VRAM
    """
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Load model with quantization
    if use_4bit:
        from transformers import BitsAndBytesConfig
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=16,  # Rank - higher = more parameters but better quality
        lora_alpha=32,  # Scaling factor
        target_modules=[
            "q_proj",
            "k_proj", 
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ],  # Modules to apply LoRA to
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer


def prepare_chat_dataset(tokenizer, dataset_name="tatsu-lab/alpaca"):
    """
    Prepare a chat/instruction dataset for training
    """
    
    # Load dataset (or use your own)
    dataset = load_dataset(dataset_name, split="train")  # Use subset for testing
    
    def format_chat(example):
        """Format as chat template"""
        if "instruction" in example:
            # Alpaca-style format
            instruction = example["instruction"]
            input_text = example.get("input", "")
            output = example["output"]
            
            if input_text:
                prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
            else:
                prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
        else:
            # Generic format
            prompt = example.get("text", "")
        
        return {"text": prompt}
    
    dataset = dataset.map(format_chat)
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=256,
            padding="max_length"
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    return tokenized_dataset


def train_chat_model(model, tokenizer, dataset, output_dir="./chat_model"):
    """
    Train the model on chat data
    """
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        warmup_steps=100,
        optim="paged_adamw_8bit",  # Memory efficient optimizer
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
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


def generate_response(model, tokenizer, prompt, max_length=256):
    """
    Generate a response using the fine-tuned model
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


# Main execution
if __name__ == "__main__":
    print("Setting up model for chat fine-tuning...")
    
    # 1. Load pretrained model with LoRA
    model, tokenizer = setup_model_for_chat_finetuning(
        model_name="Qwen/Qwen2.5-0.5B",  # Change to accessible model
        use_4bit=True
    )
    
    # 2. Prepare dataset
    print("\nPreparing chat dataset...")
    dataset = prepare_chat_dataset(tokenizer)
    
    # 3. Fine-tune
    print("\nStarting fine-tuning...")
    model = train_chat_model(model, tokenizer, dataset)
    
    # 4. Test
    # print("\nTesting generation...")
    # test_prompt = "### Instruction:\nWrite a short poem about coding.\n\n### Response:\n"
    # response = generate_response(model, tokenizer, test_prompt, max_length=50)
    # print(f"\nPrompt: {test_prompt}")
    # print(f"Response: {response}")