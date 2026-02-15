import torch
import os
from src.models.static_network import StaticSignClassifier

model_load_path = "src/models/saved_models/static_sign.pth"
data_load_path = "src/data/cache.pt"
model_weights = torch.load(model_load_path, weights_only=True)
data_info = torch.load(data_load_path)

CLASS_INDEX = data_info["classes"]
model = StaticSignClassifier(num_categories=len(CLASS_INDEX))
model.load_state_dict(model_weights)
model.eval()
print(CLASS_INDEX)