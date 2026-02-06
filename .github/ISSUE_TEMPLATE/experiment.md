---
name: Experiment Tracking
about: Document ML experiments and results
title: '[EXPERIMENT] '
labels: experiment, ml
assignees: ''
---

## Experiment Overview
Brief description of the experiment and its goals.

## Hypothesis
What are you trying to test or improve?

## Methodology

### Model Architecture
Describe the model architecture used:
- Model type: [e.g., 3D CNN, Transformer, etc.]
- Key parameters:
- Number of layers:
- Hidden dimensions:
- Other relevant details:

### Dataset
- Dataset used:
- Size: [e.g., 1000 videos, 50K frames]
- Split: [e.g., 70% train, 15% val, 15% test]
- Preprocessing: [describe preprocessing steps]

### Hyperparameters
```yaml
learning_rate: 0.001
batch_size: 32
epochs: 50
optimizer: Adam
loss_function: CrossEntropyLoss
# Add other hyperparameters
```

### Training Configuration
- Hardware: [e.g., NVIDIA RTX 3090, 24GB RAM]
- Training time: [e.g., 4 hours]
- Framework: [e.g., PyTorch 2.0]

## Results

### Metrics
| Metric | Value |
|--------|-------|
| Accuracy | XX% |
| Precision | XX% |
| Recall | XX% |
| F1 Score | XX% |
| Loss | XX |

### Comparison with Baseline
How do these results compare to previous experiments or baseline?

### Visualizations
Add plots, confusion matrices, or other visualizations here.

## Analysis

### What Worked
- Observation 1
- Observation 2

### What Didn't Work
- Issue 1
- Issue 2

### Insights
Key insights gained from this experiment.

## Next Steps
What should be tried next based on these results?
- [ ] Next experiment idea 1
- [ ] Next experiment idea 2
- [ ] Areas to investigate further

## Reproducibility

### Code
- Branch/commit: 
- Notebook/script: 
- Model weights location:

### Random Seeds
```python
random_seed = 42
torch.manual_seed(random_seed)
```

## Additional Notes
Any other relevant information, observations, or context.
