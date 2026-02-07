import torch.nn as nn

class StaticSignClassifier(nn.Module):
    '''
    Simple Classifier used for signs with no movement involved, such as much letters.
    - Input: Takes in the number of features fed into the neural network and the number of categories that the
    hand signal could represent. Also allows the node dropout rate to be configured.
    '''
    def __init__(self, num_features, num_categories, dropout = 0.3):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(num_features, 128), nn.ReLU(), nn.Dropout(dropout),
                                   nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),
                                   nn.Linear(64, num_categories))
        
    def forward(self, x):
        return self.model(x)
        
