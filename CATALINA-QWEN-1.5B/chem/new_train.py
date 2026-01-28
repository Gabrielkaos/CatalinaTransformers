import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding
)
from transformers import default_data_collator
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from peft import PeftModel
from typing import List, Dict


class CausalLMDataCollator(DataCollatorWithPadding):
    def __call__(self, features):
        # 1. Extract labels
        labels = [f.pop("labels") for f in features]

        # 2. Pad input_ids / attention_mask ONLY
        batch = super().__call__(features)

        # 3. Pad labels manually with -100
        max_len = batch["input_ids"].shape[1]

        padded_labels = [
            lbl + [-100] * (max_len - len(lbl))
            for lbl in labels
        ]

        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch


def prepare_chat_dataset_kaggle(
    tokenizer,
    max_len=512,
    csv_path="/kaggle/input/chemistry-problem-solution-dataset/train.csv"
):
    # 1. Load CSV
    dataset = load_dataset(
        "csv",
        data_files=csv_path,
        split="train"
    )

    dataset = dataset.shuffle(seed=2222)

    def tokenize_and_mask(example):
        instruction = example["message_1"]
        output = example["message_2"]

        prompt = (
            f"### Question:\n{instruction}\n\n"
            f"### Response:\n"
        )

        full_text = prompt + output + tokenizer.eos_token

        tokenized = tokenizer(
            full_text,
            truncation=True,
            max_length=max_len,
            padding=False,
        )

        input_ids = tokenized["input_ids"]

        # Tokenize prompt alone to mask it
        prompt_ids = tokenizer(
            prompt,
            truncation=True,
            max_length=max_len,
            padding=False,
        )["input_ids"]

        labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids):]
        labels = labels[:len(input_ids)]

        return {
            "input_ids": input_ids,
            "attention_mask": tokenized["attention_mask"],
            "labels": labels,
        }

    dataset = dataset.map(
        tokenize_and_mask,
        remove_columns=dataset.column_names,
        num_proc=4,  # Kaggle usually prefers <=4
    )

    return dataset


def setup_model_for_chat_finetuning(model_name):
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    added=False

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        added=True
        print("Added pad token")
    tokenizer.padding_side = "right"
    
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        trust_remote_code=True
    )
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    if added:
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id
        if tokenizer.eos_token_id is not None:
            model.config.eos_token_id = tokenizer.eos_token_id

    
   
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
    print("Model loaded")
    print(tokenizer.pad_token)
    print(tokenizer.pad_token_id)
    print(tokenizer.eos_token)
    print(tokenizer.eos_token_id)
    
    return model, tokenizer

def prepare_chat_dataset1(tokenizer, max_len=512, dataset_name="oumi-ai/lmsys_chat_1m_clean_R1"):
    dataset = load_dataset(dataset_name, split="train")
    dataset = dataset.shuffle(seed=2222)

    def no_deepseek(example):
        instr = example["prompt"].lower()
        resp = example["response"].lower()
        return "deepseek" not in instr and "deepseek" not in resp

    dataset = dataset.filter(
        no_deepseek,
        num_proc=8
    )

    def tokenize_and_mask(example):
        instruction = example["prompt"]
        # input_text = example.get("input", "")
        output = example["response"]

        if "</think>" in output:
            output = output.split("</think>")[1].strip()

        prompt = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n"
        )

        

        full_text = prompt + output + tokenizer.eos_token

        tokenized = tokenizer(
            full_text,
            truncation=True,
            max_length=max_len,
            padding=False,
        )

        input_ids = tokenized["input_ids"]

        # Tokenize prompt alone to find cutoff
        prompt_ids = tokenizer(
            prompt,
            truncation=True,
            max_length=max_len,
            padding=False,
        )["input_ids"]

        labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids):]

        # Truncate labels to match input_ids length
        labels = labels[:len(input_ids)]

        return {
            "input_ids": input_ids,
            "attention_mask": tokenized["attention_mask"],
            "labels": labels,
        }

    dataset = dataset.map(
        tokenize_and_mask,
        remove_columns=dataset.column_names,
        num_proc=8,
    )

    return dataset


def train_chat_model(model, tokenizer, dataset, output_dir="./chat_model"):
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=2,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        fp16=True,
        log_level="info",
        report_to="none",
        logging_steps=5,
        save_strategy="steps",
        save_steps=100,
        warmup_steps=100,
        optim="paged_adamw_8bit",  
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        ddp_find_unused_parameters=False
    )
    
    data_collator = CausalLMDataCollator(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
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


if __name__ == "__main__":
    # print("Setting up model for chat fine-tuning...")

    model_dir = "./chem"
    base_model_name = "Qwen/Qwen2.5-1.5B"

    # model, tokenizer = load_trained_model(model_dir, base_model_name)
    
    model, tokenizer = setup_model_for_chat_finetuning(
        model_name=base_model_name
    )
    
    print("\nPreparing chat dataset...")
    dataset = prepare_chat_dataset_kaggle(tokenizer)

    # # see data
    # sample = dataset[0]
    # for tid, label in zip(sample["input_ids"], sample["labels"]):
    #     token = tokenizer.decode([tid])
    #     print(f"{token!r:15} -> {label}")
    # print(sample["attention_mask"])
        
   
    print("\nStarting fine-tuning...")
    model = train_chat_model(model, tokenizer, dataset,output_dir=model_dir)
    print("Done training.")
    
    
    # print("\nTesting generation...")
    # test_prompt = "### Instruction:\nWrite a short poem about coding.\n\n### Response:\n"
    # response = generate_response(model, tokenizer, test_prompt, max_length=50)
    # print(f"\nPrompt: {test_prompt}")
    # print(f"Response: {response}")