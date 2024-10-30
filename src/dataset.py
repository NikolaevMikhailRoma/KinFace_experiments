from pathlib import Path
import random
from typing import List, Tuple, Dict
from sklearn.model_selection import train_test_split
import os
from config import PATH_CONFIG, DATASET_CONFIG
import numpy as np


class KinshipDataset:
    """
    Enhanced dataset class for managing file paths for kinship verification.
    Includes pair augmentation by swapping relatives and balanced negative pair generation.
    """

    def __init__(self, dataset_path: Path):
        self.dataset_path = dataset_path
        self.image_paths = self._get_all_image_paths()
        self.pairs = self._create_all_pairs()

    def _get_all_image_paths(self) -> Dict[str, Path]:
        """
        Get all image paths from the dataset directory and its subdirectories
        Returns a dictionary with filename as key and full path as value
        """
        image_paths = {}
        for root, _, files in os.walk(self.dataset_path / 'images'):
            for file in files:
                if file.endswith('.jpg'):
                    full_path = Path(root) / file
                    image_paths[file] = full_path
        return image_paths

    def _is_relative_pair(self, img1_name: str, img2_name: str) -> bool:
        """Check if two images form a relative pair based on their filenames"""
        # Extract base names without extension and suffix
        base1 = img1_name.rsplit('.', 1)[0].rsplit('_', 1)[0]
        base2 = img2_name.rsplit('.', 1)[0].rsplit('_', 1)[0]
        return base1 == base2

    def _create_all_pairs(self) -> List[Tuple[Path, Path, int]]:
        """
        Create all possible pairs with augmentation for relatives
        Returns list of tuples (img1_path, img2_path, label)
        """
        image_files = list(self.image_paths.keys())
        pairs = []

        # Create positive pairs (relatives) with augmentation
        relative_pairs = []
        for img1 in image_files:
            if img1.endswith('_1.jpg'):
                img2 = img1.replace('_1.jpg', '_2.jpg')
                if img2 in self.image_paths:
                    # Add original pair
                    relative_pairs.append((
                        self.image_paths[img1],
                        self.image_paths[img2],
                        1
                    ))
                    # Add swapped pair for augmentation
                    relative_pairs.append((
                        self.image_paths[img2],
                        self.image_paths[img1],
                        1
                    ))

        # Calculate required number of negative pairs based on config ratios
        num_relatives = len(relative_pairs)
        target_ratio = DATASET_CONFIG['RELATIVE_TO_NON_RELATIVE_RATIO_TRAIN']
        num_non_relatives = int(num_relatives * target_ratio)

        # Create negative pairs (non-relatives)
        all_images = list(self.image_paths.values())
        negative_pairs = set()
        max_attempts = num_non_relatives * 10  # Avoid infinite loop
        attempts = 0

        while len(negative_pairs) < num_non_relatives and attempts < max_attempts:
            img1, img2 = random.sample(all_images, 2)
            if not self._is_relative_pair(img1.name, img2.name):
                negative_pairs.add((img1, img2, 0))
                # Also add swapped version for balance
                negative_pairs.add((img2, img1, 0))
            attempts += 1

        # Combine and shuffle all pairs
        pairs = relative_pairs + list(negative_pairs)
        random.shuffle(pairs)

        return pairs

    def create_train_val_test_split(self) -> Tuple[List, List, List]:
        """
        Split dataset into train, validation and test sets.
        Maintains balanced ratios for relatives/non-relatives in each split.
        """
        # Split positive and negative pairs
        positive_pairs = [p for p in self.pairs if p[2] == 1]
        negative_pairs = [p for p in self.pairs if p[2] == 0]

        # Calculate split sizes
        train_size = DATASET_CONFIG['TRAIN_RATIO']
        val_size = DATASET_CONFIG['VAL_RATIO'] / (1 - train_size)

        # Split positive pairs
        train_pos, temp_pos = train_test_split(
            positive_pairs, train_size=train_size, random_state=42
        )
        val_pos, test_pos = train_test_split(
            temp_pos, train_size=val_size, random_state=42
        )

        # Split negative pairs maintaining the same ratio
        train_neg, temp_neg = train_test_split(
            negative_pairs, train_size=train_size, random_state=42
        )
        val_neg, test_neg = train_test_split(
            temp_neg, train_size=val_size, random_state=42
        )

        # Combine and shuffle splits
        train_set = train_pos + train_neg
        val_set = val_pos + val_neg
        test_set = test_pos + test_neg

        random.shuffle(train_set)
        random.shuffle(val_set)
        random.shuffle(test_set)

        return train_set, val_set, test_set

    def get_dataset_stats(self, split_name: str, split_data: List) -> Dict:
        """Calculate and return statistics for a dataset split"""
        total_samples = len(split_data)
        positive_samples = sum(1 for _, _, label in split_data if label == 1)
        negative_samples = total_samples - positive_samples

        return {
            'split_name': split_name,
            'total_samples': total_samples,
            'positive_samples': positive_samples,
            'negative_samples': negative_samples,
            'positive_ratio': positive_samples / total_samples if total_samples > 0 else 0,
            'relative_to_non_relative_ratio': (
                positive_samples / negative_samples if negative_samples > 0 else float('inf')
            )
        }


def main():
    """Test dataset creation and print statistics"""
    print("\n=== Dataset Initialization ===")
    dataset = KinshipDataset(PATH_CONFIG['DATASET_PATH'])

    # Create splits
    train_set, val_set, test_set = dataset.create_train_val_test_split()

    # Print statistics for each split
    for split_name, split_data in [
        ('Training', train_set),
        ('Validation', val_set),
        ('Testing', test_set)
    ]:
        stats = dataset.get_dataset_stats(split_name, split_data)
        print(f"\n{split_name} Set Statistics:")
        print(f"Total samples: {stats['total_samples']}")
        print(f"Positive samples: {stats['positive_samples']}")
        print(f"Negative samples: {stats['negative_samples']}")
        print(f"Positive ratio: {stats['positive_ratio']:.2%}")
        print(f"Relative to non-relative ratio: {stats['relative_to_non_relative_ratio']:.2f}")

    # Print sample paths from training and test sets
    def print_sample_pairs(split_data, split_name, num_samples=3):
        print(f"\nSample paths from {split_name} set:")
        samples = random.sample(split_data, num_samples)
        for i, sample in enumerate(samples, 1):
            print(f"\nSample {i}:")
            print(f"Image 1: {sample[0]}")
            print(f"Image 2: {sample[1]}")
            print(f"Label: {sample[2]} ({'Related' if sample[2] == 1 else 'Not Related'})")

    # Print samples for training set
    print_sample_pairs(train_set, "training")

    # Print samples for test set
    print_sample_pairs(test_set, "test")


if __name__ == "__main__":
    main()