import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import pickle
import hashlib
import json
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
from dataset import KinshipDataset
from config import (
    PATH_CONFIG,
    DATASET_CONFIG,
    INFERENCE_CONFIG
)


class DataPreprocessor:
    """Handles data preprocessing and augmentation"""

    def __init__(self, is_training: bool = False):
        self.is_training = is_training

        # Base transforms that are applied to all images
        self.base_transforms = transforms.Compose([
            transforms.Resize((DATASET_CONFIG['IMAGE_SIZE'], DATASET_CONFIG['IMAGE_SIZE'])),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=DATASET_CONFIG['NORMALIZE_MEAN'],
                std=DATASET_CONFIG['NORMALIZE_STD']
            )
        ])

        # Additional augmentation for training
        if is_training:
            self.train_transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2)
            ])
        else:
            self.train_transforms = None

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Apply preprocessing to a single image"""
        try:
            img = self.base_transforms(image)
            if self.is_training and self.train_transforms:
                img = self.train_transforms(img)
            return img
        except Exception as e:
            logging.error(f"Error preprocessing image: {e}")
            raise

    @staticmethod
    def load_image(path: Path) -> Image.Image:
        """Load image from path with error handling"""
        try:
            with Image.open(path) as img:
                # Convert to RGB and make a copy to ensure the file is closed
                return img.convert('RGB').copy()
        except Exception as e:
            logging.error(f"Error loading image {path}: {e}")
            raise


class MemoryDataset(Dataset):
    """Memory-efficient dataset for cached data"""

    def __init__(self, data_list: List[Tuple[torch.Tensor, torch.Tensor, int]]):
        self.data = data_list
        self._validate_data()

    def _validate_data(self):
        """Validate data integrity"""
        if not self.data:
            raise ValueError("Dataset is empty")

        # Verify first item structure
        first_item = self.data[0]
        if not isinstance(first_item, tuple) or len(first_item) != 3:
            raise ValueError("Invalid data format")

        # Verify data types and shapes
        img1, img2, label = first_item
        if not (isinstance(img1, torch.Tensor) and isinstance(img2, torch.Tensor)):
            raise ValueError("Images must be torch tensors")
        if img1.shape != img2.shape:
            raise ValueError("Image pair shapes don't match")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img1, img2, label = self.data[idx]
        return img1.clone(), img2.clone(), torch.tensor(label, dtype=torch.float32)


class KinFaceDataManager:
    """
    Enhanced data manager with robust caching and memory management
    """

    def __init__(self):
        self.dataset_path = PATH_CONFIG['DATASET_PATH']
        self.temp_dir = PATH_CONFIG['TEMP_DIR']
        self.config_hash = self._get_config_hash()
        self.cache_file = self.temp_dir / f'kinface_data_{self.config_hash}.pkl'

        # Initialize loaders as None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        # Set up logging
        logging.basicConfig(level=logging.INFO)

        self._initialize_data()

    def _get_config_hash(self) -> str:
        """Create hash of relevant config parameters"""
        relevant_config = {
            'dataset': DATASET_CONFIG,
            'dataset_path': str(PATH_CONFIG['DATASET_PATH']),
            'image_size': DATASET_CONFIG['IMAGE_SIZE'],
            'normalization': {
                'mean': DATASET_CONFIG['NORMALIZE_MEAN'],
                'std': DATASET_CONFIG['NORMALIZE_STD']
            }
        }
        config_str = json.dumps(relevant_config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:10]

    def _validate_cache(self, cached_data: Dict) -> bool:
        """Validate cached data structure and integrity"""
        required_keys = {'config_hash', 'train_data', 'val_data', 'test_data'}
        if not all(key in cached_data for key in required_keys):
            return False

        if cached_data['config_hash'] != self.config_hash:
            return False

        # Validate data structures
        for split in ['train_data', 'val_data', 'test_data']:
            if not isinstance(cached_data[split], list):
                return False
            if not cached_data[split]:  # Empty split
                return False

            # Validate first item in each split
            first_item = cached_data[split][0]
            if not (isinstance(first_item, tuple) and len(first_item) == 3):
                return False

        return True

    def _save_cache(self, cache_data: Dict):
        """Save cache with error handling"""
        try:
            os.makedirs(self.temp_dir, exist_ok=True)
            temp_file = self.cache_file.with_suffix('.tmp')

            # First write to temporary file
            with open(temp_file, 'wb') as f:
                pickle.dump(cache_data, f)

            # Then rename to final file
            if os.path.exists(self.cache_file):
                os.remove(self.cache_file)
            os.rename(temp_file, self.cache_file)

        except Exception as e:
            logging.error(f"Error saving cache: {e}")
            if os.path.exists(temp_file):
                os.remove(temp_file)
            raise

    def _load_cache(self) -> Optional[Dict]:
        """Load cache with validation"""
        try:
            if not self.cache_file.exists():
                return None

            with open(self.cache_file, 'rb') as f:
                cached_data = pickle.load(f)

            if not self._validate_cache(cached_data):
                logging.warning("Cache validation failed")
                return None

            return cached_data

        except Exception as e:
            logging.error(f"Error loading cache: {e}")
            return None

    def _initialize_data(self):
        """Initialize or load cached data with progress tracking"""
        cached_data = self._load_cache()

        if cached_data is not None:
            logging.info("Loading cached dataset...")
            self._create_dataloaders(cached_data)
            return

        logging.info("Building dataset from scratch...")
        self._build_and_cache_data()

    def _build_and_cache_data(self):
        """Build dataset from scratch with progress tracking"""
        try:
            # Initialize dataset
            dataset = KinshipDataset(self.dataset_path)
            train_pairs, val_pairs, test_pairs = dataset.create_train_val_test_split()

            # Process each split with progress bar
            splits = [
                ('train', train_pairs, True),
                ('validation', val_pairs, False),
                ('test', test_pairs, False)
            ]

            processed_data = {}

            for split_name, pairs, is_training in splits:
                logging.info(f"Processing {split_name} split...")
                preprocessor = DataPreprocessor(is_training=is_training)
                processed_pairs = []

                for img1_path, img2_path, label in tqdm(pairs, desc=f"Loading {split_name} images"):
                    try:
                        img1 = preprocessor.load_image(img1_path)
                        img2 = preprocessor.load_image(img2_path)

                        img1_tensor = preprocessor.preprocess_image(img1)
                        img2_tensor = preprocessor.preprocess_image(img2)

                        processed_pairs.append((img1_tensor, img2_tensor, label))
                    except Exception as e:
                        logging.error(f"Error processing pair ({img1_path}, {img2_path}): {e}")
                        continue

                processed_data[f'{split_name}_data'] = processed_pairs

            # Prepare cache data
            cache_data = {
                'config_hash': self.config_hash,
                'train_data': processed_data['train_data'],
                'val_data': processed_data['validation_data'],
                'test_data': processed_data['test_data']
            }

            # Save cache
            self._save_cache(cache_data)

            # Create dataloaders
            self._create_dataloaders(cache_data)

        except Exception as e:
            logging.error(f"Error building dataset: {e}")
            raise

    def _create_dataloaders(self, data: Dict):
        """Create DataLoaders with proper error handling"""
        try:
            self.train_loader = DataLoader(
                MemoryDataset(data['train_data']),
                batch_size=DATASET_CONFIG['BATCH_SIZE'],
                shuffle=True,
                num_workers=0,  # Using 0 for better stability
                pin_memory=DATASET_CONFIG['PIN_MEMORY']
            )

            self.val_loader = DataLoader(
                MemoryDataset(data['val_data']),
                batch_size=DATASET_CONFIG['BATCH_SIZE'],
                shuffle=False,
                num_workers=0,
                pin_memory=DATASET_CONFIG['PIN_MEMORY']
            )

            self.test_loader = DataLoader(
                MemoryDataset(data['test_data']),
                batch_size=DATASET_CONFIG['BATCH_SIZE'],
                shuffle=False,
                num_workers=0,
                pin_memory=DATASET_CONFIG['PIN_MEMORY']
            )

        except Exception as e:
            logging.error(f"Error creating dataloaders: {e}")
            raise


class DataVisualizer:
    """
    Enhanced visualizer with better label visibility
    """

    @staticmethod
    def denormalize_image(img: torch.Tensor) -> np.ndarray:
        """Denormalize image tensor to numpy array"""
        img = img.permute(1, 2, 0).numpy()
        mean = np.array(DATASET_CONFIG['NORMALIZE_MEAN'])
        std = np.array(DATASET_CONFIG['NORMALIZE_STD'])
        img = std * img + mean
        return np.clip(img, 0, 1)

    @staticmethod
    def visualize_samples(data_loader: DataLoader, split_name: str, num_samples: int = 3):
        """Visualize random samples from the dataset with enhanced labels"""
        batch = next(iter(data_loader))
        images1, images2, labels = [item for item in batch]

        # Create figure with more space for labels
        fig = plt.figure(figsize=(10, 3 * num_samples))
        plt.suptitle(f"{split_name} Set Samples", fontsize=16, y=0.95)

        for i in range(min(num_samples, len(images1))):
            # Create subplot for pair of images
            ax1 = plt.subplot(num_samples, 2, i * 2 + 1)
            ax2 = plt.subplot(num_samples, 2, i * 2 + 2)

            # Get and denormalize images
            img1 = DataVisualizer.denormalize_image(images1[i])
            img2 = DataVisualizer.denormalize_image(images2[i])

            # Display images
            ax1.imshow(img1)
            ax2.imshow(img2)

            # Remove axes
            ax1.axis('off')
            ax2.axis('off')

            # Add labels with colored background for better visibility
            label_text = "Related" if labels[i].item() > 0.5 else "Not Related"
            color = 'green' if labels[i].item() > 0.5 else 'red'

            # Add rectangle background for label
            ax1.text(-0.1, -0.15, f"Sample {i + 1}: {label_text}",
                     transform=ax1.transAxes,
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor=color, pad=5),
                     color=color,
                     fontsize=12,
                     horizontalalignment='left')

            # Add arrows or lines connecting related images
            if labels[i].item() > 0.5:
                con = plt.matplotlib.patches.ConnectionPatch(
                    xyA=(1, 0.5), xyB=(0, 0.5),
                    coordsA='axes fraction', coordsB='axes fraction',
                    axesA=ax1, axesB=ax2,
                    color='green', arrowstyle='<->', linewidth=2
                )
                fig.add_artist(con)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()


def analyze_data_split(loader, split_name: str):
    """Analyze a data split and return comprehensive statistics"""
    positive = 0
    total = 0
    # Get shapes from first batch
    sample_batch = next(iter(loader))
    batch_size = sample_batch[0].shape[0]
    img_shape = sample_batch[0].shape[1:]

    # Calculate class distribution
    for _, _, labels in loader:
        positive += (labels > 0.5).sum().item()
        total += len(labels)

    negative = total - positive

    stats = {
        'split_name': split_name,
        'total_samples': total,
        'positive_samples': positive,
        'negative_samples': negative,
        'positive_ratio': positive / total,
        'batch_size': batch_size,
        'num_batches': len(loader),
        'image_shape': img_shape,
        'data_type': sample_batch[0].dtype
    }

    return stats


def print_split_stats(stats: dict):
    """Print formatted statistics for a data split"""
    print(f"\n{stats['split_name']} Set Statistics:")
    print(f"Total Samples: {stats['total_samples']}")
    print(f"├── Positive (Related) pairs: {stats['positive_samples']}")
    print(f"├── Negative (Not Related) pairs: {stats['negative_samples']}")
    print(f"└── Positive Ratio: {stats['positive_ratio']:.2%}")
    print(f"\nBatch Information:")
    print(f"├── Batch Size: {stats['batch_size']}")
    print(f"├── Number of Batches: {stats['num_batches']}")
    print(f"├── Image Shape: {stats['image_shape']}")
    print(f"└── Data Type: {stats['data_type']}")


def main():
    """Test data loading and visualization with comprehensive analysis"""
    try:
        print("\n=== Initializing Data Manager ===")
        data_manager = KinFaceDataManager()

        # Analyze each split
        splits = [
            (data_manager.train_loader, "Training"),
            (data_manager.val_loader, "Validation"),
            (data_manager.test_loader, "Testing")
        ]

        all_stats = []
        for loader, split_name in splits:
            stats = analyze_data_split(loader, split_name)
            all_stats.append(stats)
            print_split_stats(stats)

        # Print overall dataset statistics
        total_samples = sum(stats['total_samples'] for stats in all_stats)
        total_positive = sum(stats['positive_samples'] for stats in all_stats)
        total_negative = sum(stats['negative_samples'] for stats in all_stats)

        print("\n=== Overall Dataset Statistics ===")
        print(f"Total Dataset Size: {total_samples}")
        print(f"├── Total Related Pairs: {total_positive}")
        print(f"├── Total Unrelated Pairs: {total_negative}")
        print(f"└── Overall Positive Ratio: {total_positive / total_samples:.2%}")

        # Memory usage estimation
        sample_batch = next(iter(data_manager.train_loader))
        single_image_size = sample_batch[0][0].element_size() * sample_batch[0][0].nelement()
        total_memory_mb = (single_image_size * 2 * total_samples) / (1024 * 1024)  # *2 for pairs

        print("\n=== Memory Usage ===")
        print(f"Estimated Memory Usage: {total_memory_mb:.2f} MB")

        # Visualize samples from each split
        print("\n=== Visualizing Samples ===")
        for loader, split_name in splits:
            print(f"\nVisualizing {split_name} samples...")
            DataVisualizer.visualize_samples(loader, split_name)

    except Exception as e:
        logging.error(f"Error in main: {e}")
        raise


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    main()