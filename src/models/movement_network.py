from typing import Any
import torch.nn as nn


### NOT IN USE YET, may be incorporated in the future ###


class MovementSignClassifier(nn.Module):
    '''
    Simple Classifier used for signs with movement involved, most signs.
    Args:
        num_features - Number of features input into the neural network (default 21 points x 2 hands x 2 cords = 84)
        num_cateogries - The number of categories that the hand signal could represent
        hidden_nodes -  Allows the number of nodes per hidden layer to be changed
                        If underfitting increase and if overfitting decrease
        kernel_size -   Edits the dimensions of the filter scanning the input to extract features
                        Increasing allows smoother motion but adds latency
    '''
    def __init__(self, num_categories, num_features = 84, hidden_nodes = 128, kernel_size = 5):

        super().__init__()
        # Sets up the neural network architecture
        self.model = nn.Sequential( nn.Conv1d(in_channels=num_features, out_channels=hidden_nodes,
                                              kernel_size=kernel_size, padding=2),
                                    nn.ReLU(inplace=True),
                                    nn.Conv1d(in_channels=hidden_nodes, out_channels=hidden_nodes,
                                              kernel_size=kernel_size, padding=2),
                                    nn.ReLU(inplace=True),
                                    nn.AdaptiveAvgPool1d(1))
        self.head = nn.Linear(hidden_nodes, num_categories)

    def forward(self, x):
        x = x.transpose(1, 2)
        h = self.model(x).squeeze(-1)
        return self.head(h)
        
