


import torch



scores = torch.tensor([[1,2,3,4,5,6,7,8,9,10,10,10,10,10,10,10,10]],dtype=torch.float32)

pad_token = 10

mask = scores == pad_token


print(scores.masked_fill(mask,float("-inf")))