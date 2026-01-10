
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from transformers import default_data_collator
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from peft import PeftModel


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
    if tokenizer.pad_token is None:
        # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="auto"
    )

    # model.resize_token_embeddings(len(tokenizer))
    
   
    lora_config = LoraConfig(
        r=32,  
        lora_alpha=64, 
        target_modules=[
            "c_attn",
            "c_proj",    
            "c_fc"
        ],  
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer

def prepare_chat_dataset1(tokenizer, max_len=512, dataset_name="mou3az/Question-Answering-Generation-Choices"):
    dataset = load_dataset(dataset_name, split="train[:2000]")

    def filter_valid_examples(example):
        """Keep only examples with non-null, non-empty context, question, and answer"""
        context = example.get("context")
        question = example.get("question")
        answer = example.get("answer")
        
        # Check if any field is None or empty string
        if context is None or not str(context).strip():
            return False
        if question is None or not str(question).strip():
            return False
        if answer is None or not str(answer).strip():
            return False
        
        return True
    
    dataset = dataset.filter(filter_valid_examples, num_proc=6)

    def tokenize_and_mask(example):
        context = example["context"]
        question = example.get("question", "")
        answer = example["answer"]

        
        prompt = (
            f"### Context:\n{context}\n\n"
            f"### Question:\n{question}\n\n"
            f"### Answer:\n"
        )

        full_text = prompt + answer + tokenizer.eos_token

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
        num_proc=16
    )

    assert all(len(x["input_ids"]) == len(x["labels"]) for x in dataset.select(range(100)))


    return dataset


def train_chat_model(model, tokenizer, dataset, output_dir="./chat_model"):

    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=10,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        warmup_ratio=0.05,
        warmup_steps=100,
        optim="paged_adamw_8bit",  
    )
    
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
            repetition_penalty=1.1
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=skip_special)
    if not skip_special and tokenizer.eos_token in response:
        response = response.split(tokenizer.eos_token)[0]
    return response

if __name__ == "__main__":
    # print("Setting up model for chat fine-tuning...")

    model_dir = "./feature_extract"
    base_model_name = "openai-community/gpt2"

    # model, tokenizer = load_trained_model(model_dir, base_model_name)
    
    model, tokenizer = setup_model_for_chat_finetuning(
        model_name=base_model_name
    )
    
    print("\nPreparing chat dataset...")
    dataset = prepare_chat_dataset1(tokenizer, max_len=1024)

    # see data
    # sample = dataset[0]
    # for tid, label in zip(sample["input_ids"], sample["labels"]):
    #     token = tokenizer.decode([tid])
    #     print(f"{token!r:15} -> {label}")
        
   
    print("\nStarting fine-tuning...")
    model = train_chat_model(model, tokenizer, dataset,output_dir=model_dir)
    print("Done training.")
    
    # inputs = "10*10 = "
    # print("\nTesting generation...")
    # response = generate_response(model, tokenizer, inputs, max_length=128)
    # # print(f"\nPrompt: {test_prompt}")
    # print(f": {response}")