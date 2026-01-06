import tiktoken



tokenizer = tiktoken.get_encoding("cl100k_base")

tokens = [100256]

print(tokenizer.decode(tokens))
print(tokenizer.eot_token)   