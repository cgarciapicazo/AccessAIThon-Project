'''
This file while smooth the probabilities given by the live video predictor in main
It uses exponential moving averages and its purpose is to prevent sudden cateogry
spikes and rather only decide on the category if it is predicted over several frames.
'''
import numpy as np

class probEMA:

    def __init__(self, num_categories, alpha = 0.2) -> None:
        self.alpha = alpha
        # Initial probabilities set to 0
        self.probabilities = np.zeros((num_categories,), dtype=np.float32)

    # When a new frame with probabilities is passes it will update the current probabilities using ema
    def update(self, new_probabilities):
        self.probabilities = (1.0 - self.alpha) * self.probabilities + self.alpha * new_probabilities
        return self.probabilities
    
    # source: https://en.wikipedia.org/wiki/Moving_average