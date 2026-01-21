import torch


def rope(dim, pos, beta=10000):

    #beta ** (dim // 2)k / dim
    scale = beta ** (torch.arange(0,dim,2) / dim)
    # 1 / scale
    radians = 1.0 / scale
    pos = torch.arange(pos)

    out = torch.einsum("...i,j->...ij",pos, radians)

    cos, sin = torch.cos(out), torch.sin(out)

    out = torch.stack([cos, -sin, sin, cos], dim=-1)

    print(cos, sin)
    print(out.shape)
    print(out)




rope(4, 4)