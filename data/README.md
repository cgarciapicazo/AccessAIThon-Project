# Data Directory

This directory contains all data used for training, validation, and testing the video interpretation model.

## Structure

### `raw/`
Original, immutable data dumps. Never modify files in this directory.

**Contents**:
- Raw video files
- Original datasets
- Downloaded data

**Note**: Large data files are not tracked in git. Use `.gitkeep` to preserve directory structure.

### `processed/`
Cleaned and transformed data ready for model training.

**Contents**:
- Preprocessed video frames
- Feature extractions
- Training/validation/test splits
- Normalized datasets

## Data Management

### Adding New Data

1. Place raw data files in `data/raw/`
2. Run preprocessing scripts to generate processed data
3. Processed data will be saved to `data/processed/`

### Data Versioning

For proper data versioning, consider using tools like:
- DVC (Data Version Control)
- Git LFS for large files
- Cloud storage with versioning (S3, Azure Blob)

## Important Notes

- **Never commit large data files to git**
- Keep raw data immutable
- Document data sources and preprocessing steps
- Maintain data lineage for reproducibility
- Ensure proper data licensing and usage rights

## Data Format Guidelines

### Video Files
- Supported formats: MP4, AVI, MOV
- Recommended resolution: 1080p or higher
- Frame rate: 30 fps minimum

### Processed Data
- Frames stored as numpy arrays or image files
- Metadata in JSON format
- Clear naming conventions for easy tracking
