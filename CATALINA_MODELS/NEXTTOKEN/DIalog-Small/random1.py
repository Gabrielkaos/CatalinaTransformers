

import torch



torch.save({"model_state":torch.load("best_model.pth",map_location=torch.device("cpu"))["model_state"]},"brain.pth")