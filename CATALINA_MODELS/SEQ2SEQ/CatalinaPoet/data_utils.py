import torch


def get_index(all_words1,char):
    try:
        return all_words1.index(char)
    except (IndexError, ValueError):
        return -1


# helper functions
def char_tensor(sentence, all_characters1):
    tensor = torch.zeros(len(sentence)).long()

    for i in range(len(sentence)):
        tensor[i] = get_index(all_characters1,sentence[i])

    return tensor
