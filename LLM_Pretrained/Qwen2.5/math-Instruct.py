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

def clean_output(text):
        
    text = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"\1/\2", text)
    text = re.sub(r"\^\{\\frac\{(\d+)\{(\d+)\}\}\}", r"^(\1/\2)", text)


    
    text = re.sub(r"\\boxed\{([^}]+)\}", r"[\1]", text)

    
    text = re.sub(r"\\text\{([^}]+)\}", r"\1", text)
    text = re.sub(r"\\quad|\\qquad", " ", text)
    text = re.sub(r"\\sqrt\[(\d+)\]\{([^}]+)\}", r"(\2)^(1/\1)", text)
    text = re.sub(r"\\sqrt\{([^}]+)\}", r"sqrt(\1)", text)
    text = re.sub(r"\\times", "*", text)
    text = re.sub(r"\\pi", "π", text)
    text = re.sub(r"\\approx", "approximate~", text)
    text = re.sub(r"\\left|\\right", "", text)
    text = re.sub(r"\^\{([^}]+)\}", r"^(\1)", text)
    text = re.sub(r"1/2\s*sqrt", "1/(2*sqrt)", text)
    text = re.sub(r"([-\w^]+)/sqrt", r"(\1)/sqrt", text)
    text = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"(\1)/(\2)", text)

    text = re.sub(r"\\equiv", "≡", text)
    text = re.sub(r"\\pmod\{([^}]+)\}", r"(mod \1)", text)

    text = re.sub(r"\\mathbb\{([^}]+)\}", r"\1", text)

    text = re.sub(r"\\cdot", "*", text)
    text = re.sub(r"\\div", "/", text)
    text = re.sub(r"\\%", "%", text)
    text = re.sub(r"\\frac\{([^{}]+)\{([^}]+)\)\}", r"\1/\2", text)

    text = re.sub(r"-\\log", r"(-log)", text)
    text = re.sub(r"\\log", r"log", text)

    text = re.sub(r"\\setminus", "-", text)  # Set difference
    text = re.sub(r"\\cap", "∩", text)  # Intersection
    text = re.sub(r"\\cup", "∪", text)  # Union
    text = re.sub(r"\\emptyset", "∅", text)  # Empty set
    text = re.sub(r"\\subset", "⊂", text)  # Subset
    text = re.sub(r"\\subseteq", "⊆", text)  # Subset or equal
    text = re.sub(r"\\in", "∈", text)  # Element of
    text = re.sub(r"\\notin", "∉", text)  # Not element of
    
    text = text.replace(r"\(", "").replace(r"\)", "")
    text = text.replace(r"\[", "").replace(r"\]", "")

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    text = re.sub(r"\\begin\{pmatrix\}(.*?)\\end\{pmatrix\}", lambda m: "[" + m.group(1).replace("\\\\", "; ").strip() + "]", text, flags=re.DOTALL)
    text = re.sub(r"\\begin\{vmatrix\}(.*?)\\end\{vmatrix\}", lambda m: "|" + m.group(1).replace("\\\\", "; ").strip() + "|", text, flags=re.DOTALL)
    text = re.sub(r"\\begin\{cases\}(.*?)\\end\{cases\}", lambda m: m.group(1).replace("\\\\", "\n").strip(), text, flags=re.DOTALL)

    text = re.sub(r"\\mathbf\{([^}]+)\}", r"\1", text)

    text = re.sub(r"\\det\(([^)]+)\)", r"det(\1)", text)

    text = re.sub(r"\\vmatrix", "det", text)

    text = re.sub(r"\s*&\s*", " ", text)

    text = re.sub(r"\\\s*$", "", text)

    return text.strip()

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
    print(clean_output(response))
    print("#" * 100)
    print()