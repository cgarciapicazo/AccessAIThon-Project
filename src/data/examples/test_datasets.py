import torch

def generate_static(num_of_labels ,num_of_records = 100, num_of_features = 84):
    X = torch.rand(num_of_records, num_of_features, dtype=torch.float32)
    y = torch.randint(0, num_of_labels, (num_of_records,), dtype=torch.int64)
    return X, y

# needs to generate an appropiate demo tensor in the formal [B, T, F]
### NOT IN USE YET ###
def generate_motion():
    X = []
    y = []
    return X, y
