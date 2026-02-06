# Models Directory

This directory contains trained model weights and model architecture definitions.

## Structure

### `saved_models/`
Trained model weights and checkpoints.

**Contents**:
- Best model weights (e.g., `best_model.pth`)
- Training checkpoints (e.g., `checkpoint_epoch_10.pth`)
- Model state dictionaries

**Naming Convention**:
- `{model_name}_{version}_{metric}.pth`
- Example: `video_interpreter_v1_acc_0.95.pth`

**Note**: Model files are not tracked in git due to their large size. Use model versioning tools like MLflow or Weights & Biases for proper tracking.

### `architectures/`
Model architecture definitions and configurations.

**Contents**:
- Architecture specification files
- Model configuration JSON/YAML files
- Custom layer implementations
- Architecture diagrams

## Model Management

### Saving Models

```python
import torch

# Save full model
torch.save(model, 'models/saved_models/model_name.pth')

# Save only state dict (recommended)
torch.save(model.state_dict(), 'models/saved_models/model_name.pth')

# Save with training metadata
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, 'models/saved_models/checkpoint.pth')
```

### Loading Models

```python
import torch

# Load full model
model = torch.load('models/saved_models/model_name.pth')

# Load state dict
model = VideoInterpreter()
model.load_state_dict(torch.load('models/saved_models/model_name.pth'))
```

## Best Practices

1. **Version Control**: Tag models with version numbers and performance metrics
2. **Documentation**: Document model architecture and training details
3. **Checkpointing**: Save checkpoints during training for recovery
4. **Best Model**: Always keep the best performing model separately
5. **Cleanup**: Remove old checkpoints to save storage space

## Model Registry

Keep a record of trained models:

| Model Name | Version | Date | Accuracy | Notes |
|------------|---------|------|----------|-------|
| - | - | - | - | - |

## Storage Considerations

- Use cloud storage for large models (AWS S3, Google Cloud Storage)
- Consider model compression techniques
- Use model versioning tools (MLflow, DVC)
- Keep only essential checkpoints locally
