from transformers import AutoModelForCausalLM, AutoTokenizer
import re


model_name = "Qwen/Qwen2.5-Math-1.5B-Instruct"
device = "cuda";print(device)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

while True:
    prompt = input("\n:")
    if prompt in ["quit","exit"]:break

    # messages = [
    #     {
    #         "role": "system",
    #         "content": (
    #             "Solve the problem. "
    #             "Output ONLY:\n"
    #             "1) A brief solution (no prose)\n"
    #             "2) Final answer in \\boxed{}"
    #         )
    #     },
    #     {"role": "user", "content": prompt}
    # ]

    # # CoT
    messages = [
        {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
        {"role": "user", "content": prompt}
    ]

    # # TIR
    # messages = [
    #     {"role": "system", "content": "Please integrate natural language reasoning with programs to solve the problem above, and put your final answer within \\boxed{}."},
    #     {"role": "user", "content": prompt}
    # ]


    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    print("Generating")
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=1024,
        eos_token_id = tokenizer.eos_token_id,
        do_sample=False
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )[0]

    print("#" * 100)
    response = response.strip()
    print(response)
    print("#" * 100)
    print()