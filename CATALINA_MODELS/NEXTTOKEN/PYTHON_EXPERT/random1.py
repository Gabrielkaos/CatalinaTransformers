import tiktoken



tokenizer = tiktoken.get_encoding("cl100k_base")

sentence = "Hello world I am Gabriel"

tokens = tokenizer.encode(sentence) + [tokenizer.eot_token]

print(tokens)

print(tokenizer.decode(tokens))