import torch

def generate_example_dataset(num_of_labels ,num_of_records = 100, num_of_features = 84):
    X = torch.rand(num_of_records, num_of_features, dtype=torch.float32)
    y = torch.randint(0, num_of_labels, (num_of_records,), dtype=torch.int64)
    print(X)
    print(y)
    return X, y
