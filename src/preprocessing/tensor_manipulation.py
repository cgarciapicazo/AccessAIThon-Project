import torch

def normalise_tensor(tensor):
    demo_tensor = torch.arange(2, 30, 2)
    demo_tensor = demo_tensor - demo_tensor[0]
    print(demo_tensor)


normalise_tensor("a")