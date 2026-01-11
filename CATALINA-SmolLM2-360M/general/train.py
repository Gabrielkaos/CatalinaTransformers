
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
        device_map="auto"
    )
    
    if added:
        model.resize_token_embeddings(len(tokenizer))

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

   
    lora_config = LoraConfig(
        r=16,  
        lora_alpha=32, 
        target_modules=[
               "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
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


def prepare_chat_dataset1(tokenizer, max_len=512, dataset_name="yahma/alpaca-cleaned"):
    dataset = load_dataset(dataset_name, split="train[:15000]") # 50k rows


    def tokenize_and_mask(example):
        instruction = example["instruction"]
        input_text = example.get("input", "")
        output = example["output"]

        if input_text:
            prompt = (
                f"### Instruction:\n{instruction}\n\n"
                f"### Input:\n{input_text}\n\n"
                f"### Response:\n"
            )
        else:
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
        num_proc=4,
    )

    return dataset


def train_chat_model(model, tokenizer, dataset, output_dir="./chat_model"):
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=2,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=20,
        save_strategy="epoch",
        warmup_ratio=0.05,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        report_to="none",
    )
    # def data_collator_token_classification(features: List[Dict]):
    #     # Use tokenizer.pad to pad input_ids/attention_mask/etc into a batch
    #     batch = tokenizer.pad(
    #         features,
    #         padding="longest",
    #         return_tensors="pt",
    #     )

    #     # Handle labels: features may contain 'labels' as Python lists with -100 already set for prompt tokens
    #     if "labels" in features[0]:
    #         labels = [torch.tensor(f["labels"], dtype=torch.long) for f in features]
    #         labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    #         batch["labels"] = labels

    #     return batch

        
    # data_collator = DataCollatorForLanguageModeling(
    #     tokenizer=tokenizer,
    #     mlm=False,
    #     pad_to_multiple_of=8
    # )
    
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
            repetition_penalty=1.1,
            
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=skip_special)
    if not skip_special and tokenizer.eos_token in response:
        response = response.split(tokenizer.eos_token)[0]
    return response

if __name__ == "__main__":
    # print("Setting up model for chat fine-tuning...")

    model_dir = "./general_chat2"
    base_model_name = "HuggingFaceTB/SmolLM2-360M"

    # model, tokenizer = load_trained_model(model_dir, base_model_name)
    
    model, tokenizer = setup_model_for_chat_finetuning(
        model_name=base_model_name
    )
    
    
    print("\nPreparing chat dataset...")
    dataset = prepare_chat_dataset1(tokenizer)

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
    # test_prompt = ""
    # response = generate_response(model, tokenizer, test_prompt, max_length=50)
    # print(f"\nPrompt: {test_prompt}")
    # print(f"Response: {response}")