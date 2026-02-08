import torch.nn as nn

class StaticSignClassifier(nn.Module):
    '''
    Simple Classifier used for signs with no movement involved, such as much letters.
    Args:
        num_features - Number of features input into the neural network (default 21 points x 2 hands x 2 cords = 84)
        num_cateogries - The number of categories that the hand signal could represent
        dropout - Allows the node dropout rate to be configured
    '''
    def __init__(self,num_categories, num_features = 84, dropout = 0.3):
        super().__init__()
        # Sets up the neural network architecture
        self.model = nn.Sequential(nn.Linear(num_features, 128), nn.ReLU(), nn.Dropout(dropout),
                                   nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),
                                   nn.Linear(64, num_categories))
        
    # Pass the tensor through every layer
    def forward(self, x):
        return self.model(x)
        