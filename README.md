# Kinship Recognition Project

This project implements a deep learning system for kinship verification using facial images. The system can determine whether two people are related based on their photographs.

## Project Structure
```
📁 project_root/
├── 📁 src/
│   ├── 📄 config.py          # Configuration settings
│   ├── 📄 dataset.py         # Dataset management
│   ├── 📄 dataloader.py      # Data loading and preprocessing
│   ├── 📄 models.py          # Neural network architectures
│   ├── 📄 training.py        # Training procedures
│   └── 📄 app.py            # FastAPI application
├── 📁 data/
│   ├── 📁 models/           # Trained model checkpoints
│   │   └── 📁 old_models/   # Archived model versions
│   └── 📁 temp/            # Cached data and temporary files
└── 📄 README.md
```

## Prerequisites

### System Requirements
- Python 3.8+
- CUDA-capable GPU (optional, but recommended for training)
- 8GB+ RAM

### Required Packages
```bash
pip install torch torchvision
pip install fastapi uvicorn
pip install pillow numpy pandas
pip install scikit-learn
pip install tqdm
pip install requests
pip install python-multipart
```

Or install all dependencies using the requirements file:
```bash
pip install -r requirements.txt
```

## Configuration

### Dataset Configuration
The project uses the KinFaceW-II dataset. Configure your dataset path in `config.py`:

```python
PATH_CONFIG = {
    'DATASET_PATH': Path('/path/to/your/KinFaceW-II/'),
    'MODELS_DIR': PROJECT_ROOT / 'data' / 'models',
    'TEMP_DIR': PROJECT_ROOT / 'data' / 'temp',
}
```

### Dataset Structure
The KinFaceW-II dataset should have the following structure, but you can add any images, the main thing is that the file name should clearly indicate that they belong to a pair of relatives:
```
KinFaceW-II/
├── images/
│   ├── father-dau/
│   ├── father-son/
│   ├── mother-dau/
│   └── mother-son/
└── meta_data/
```

### Training Configuration
Key training parameters in `config.py`:
```python
TRAINING_CONFIG = {
    'EPOCHS': 1000,
    'INITIAL_LR': 0.01,
    'MIN_LR': 1e-10,
    'PATIENCE': 20,
    'EARLY_STOP_PATIENCE': 50,
    'WEIGHT_DECAY': 1e-5,
    'MOMENTUM': 0.9,
}
```

### Hardware Configuration
By default, the system will use:
- CUDA GPU if available
- Apple Silicon MPS if available
- CPU as fallback

Configure device preference in `config.py`:
```python
INFERENCE_CONFIG = {
    "DEVICE": 'cpu'  # or 'cuda' or 'mps'
}
```

## Dataset Management

The dataset module (`dataset.py`) handles:
- Dataset organization
- Train/validation/test splits
- Positive/negative pair generation
- Data augmentation

Key features:
- Configurable train/val/test split ratios
- Balanced class distribution
- Automatic negative pair generation
- Data integrity verification

## Data Loading

The dataloader module (`dataloader.py`) provides:
- Efficient data loading
- Image preprocessing
- Data augmentation
- Caching mechanisms

Key features:
- Memory-efficient data handling
- Real-time augmentation
- Normalization and preprocessing
- Multi-threaded data loading

## Model Architecture

The project includes three model architectures (`models.py`):
1. `SimpleNet`: Lightweight architecture
2. `KinshipNet`: Standard architecture
3. `KinshipNet_2`: Enhanced architecture with regularization

## Training

### Starting Training
To train all models:
```bash
python src/training.py
```

The training process:
1. Loads and preprocesses the dataset
2. Trains multiple model architectures
3. Saves checkpoints and metrics
4. Performs model selection
5. Cleans up old checkpoints

### Model Checkpoints
Models are saved in `data/models/` with naming convention:
```
best_[ModelName]_F1_[Score]_[Timestamp].pth
```

## FastAPI Application

### Starting the Server
```bash
python src/app.py
```

The server will start at `http://localhost:8000`

### API Endpoints

#### Predict Endpoint
```bash
POST /predict
```

Request body:
```json
{
    "image1_url": "http://example.com/person1.jpg",
    "image2_url": "http://example.com/person2.jpg"
}
```

Response:
```json
{
    "probability": 0.85,
    "are_related": true,
    "processing_time_ms": 245.5
}
```

#### Health Check
```bash
GET /health
```

## Model Deployment

1. Ensure all dependencies are installed
2. Configure paths in `config.py`
3. Place trained model in `data/models/`
4. Start the FastAPI server

## Troubleshooting

### Common Issues

1. **GPU Out of Memory**
   - Reduce batch size in `config.py`
   - Use gradient accumulation
   - Switch to CPU mode

2. **Slow Training**
   - Check number of workers in dataloader
   - Verify dataset is cached properly
   - Check disk I/O performance

3. **Poor Model Performance**
   - Check data preprocessing
   - Verify dataset balance
   - Adjust learning rate
   - Increase model capacity

### Logs
- Training logs are stored in console output
- API server logs are in console output
- Set logging level in respective files

## Development

### Adding New Features

1. **New Model Architecture**
   - Add model class in `models.py`
   - Register in training pipeline
   - Update configuration if needed

2. **Data Augmentation**
   - Modify `DataPreprocessor` in `dataloader.py`
   - Update augmentation parameters in config

3. **API Extensions**
   - Add new endpoints in `app.py`
   - Update response models
   - Add new processing functions

## Comments:
You can add new files to the dataset directory.
To check the main structures, you need to run the main functions in .py files, for example: run training to create new models, run app to start the server (it will automatically select the best available model). You can modify parameters in src config for experiments.