from pathlib import Path
import torch

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Path configurations
PATH_CONFIG = {
    'DATASET_PATH': Path('/Users/admin/projects/data/KinFace/KinFaceW-II/'),
    'MODELS_DIR': PROJECT_ROOT / 'data' / 'models',
    'TEMP_DIR': PROJECT_ROOT / 'data' / 'temp',
}

# Dataset parameters
DATASET_CONFIG = {
    # Split ratios (must sum to 1)
    'TRAIN_RATIO': 0.8,
    'VAL_RATIO': 0.1,
    'TEST_RATIO': 0.1,

    # Class balance parameters
    'RELATIVE_TO_NON_RELATIVE_RATIO_TRAIN': 1.0,  # ratio between relatives and non-relatives
    'RELATIVE_TO_NON_RELATIVE_RATIO_TEST': 1.0,  # 1:1 ratio between relatives and non-relatives

    # Augmentation parameters
    'AUGMENTATION_RATIO': 1.0,  # Add augmentation to training data

    # Image parameters
    'IMAGE_SIZE': 64,
    'CHANNELS': 3,

    # Data loading parameters
    'BATCH_SIZE': 64,
    'NUM_WORKERS': 4,
    'PIN_MEMORY': True,

    # Data normalization parameters
    'NORMALIZE_MEAN': [0.485, 0.456, 0.406],
    'NORMALIZE_STD': [0.229, 0.224, 0.225]
}

# Training parameters
TRAINING_CONFIG = {
    # Basic training parameters
    'EPOCHS': 1000,
    'INITIAL_LR': 0.01,
    'MIN_LR': 1e-10,

    # Early stopping parameters
    'PATIENCE': 20,  # Number of epochs to wait before reducing LR
    'EARLY_STOP_PATIENCE': 100,  # Number of epochs to wait before stopping

    # Optimizer parameters
    'WEIGHT_DECAY': 1e-5,
    'MOMENTUM': 0.9,

    # Learning rate scheduler parameters
    'LR_SCHEDULER_FACTOR': 0.1,  # Factor by which to reduce LR
    'LR_SCHEDULER_MODE': 'min',  # Reduce LR when monitored value stops decreasing

    # Model checkpointing
    'SAVE_BEST_ONLY': True,
    'SAVE_FREQUENCY': 10  # Save model every N epochs
}

# Inference parameters
INFERENCE_CONFIG = {
    # Device selection
    'DEVICE': torch.device('cuda' if torch.cuda.is_available() else
                           'mps' if torch.backends.mps.is_available() else
                           'cpu'),
    # "DEVICE": 'cpu',

    # Inference batch size (can be larger than training batch size)
    'BATCH_SIZE': 64,

    # Threshold for binary classification
    'CLASSIFICATION_THRESHOLD': 0.5,

    # Number of workers for data loading during inference
    'NUM_WORKERS': 2
}


# Create directories if they don't exist
def create_directories():
    """Create necessary directories for the project"""
    for dir_path in [PATH_CONFIG['MODELS_DIR'], PATH_CONFIG['TEMP_DIR']]:
        dir_path.mkdir(parents=True, exist_ok=True)


# Initialize directories when config is imported
create_directories()