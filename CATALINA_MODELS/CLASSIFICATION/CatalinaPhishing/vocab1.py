import string

VOCAB = list(string.ascii_lowercase + string.digits + "/.-_?=&%:")
PAD = "<PAD>"
UNK = "<UNK>"

itos = [PAD, UNK] + VOCAB