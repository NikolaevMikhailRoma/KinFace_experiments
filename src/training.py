import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import logging
from datetime import datetime
from sklearn.metrics import f1_score
from typing import Dict, List, Tuple, Type, Optional
import json
from dataloader import KinFaceDataManager
from models import KinshipNet, KinshipNet_2, SimpleNet
from config import (
    PATH_CONFIG,
    TRAINING_CONFIG,
    INFERENCE_CONFIG
)
# В начало файла training.py добавим:
import shutil
import os
import re
from typing import Dict, List, Tuple, Type, Optional, Pattern


class ModelTrainer:
    """
    Handles model training, evaluation, and checkpointing
    """

    def __init__(
            self,
            model_class: Type[nn.Module],
            data_manager: KinFaceDataManager,
            device: str = INFERENCE_CONFIG['DEVICE']
    ):
        self.model_class = model_class
        self.model = model_class().to(device)
        self.device = device
        self.data_manager = data_manager
        self.criterion = nn.BCEWithLogitsLoss()

        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=TRAINING_CONFIG['INITIAL_LR'],
            weight_decay=TRAINING_CONFIG['WEIGHT_DECAY']
        )

        # Initialize learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode=TRAINING_CONFIG['LR_SCHEDULER_MODE'],
            factor=TRAINING_CONFIG['LR_SCHEDULER_FACTOR'],
            patience=TRAINING_CONFIG['PATIENCE'],
            verbose=True
        )

        # Training tracking
        self.best_val_f1 = 0
        self.best_model_state = None
        self.epochs_without_improvement = 0

        # Set up logging
        self.logger = logging.getLogger(f"{self.model_class.__name__}_trainer")
        # self.logger.setLevel(logging.INFO)

    def _calculate_metrics(self, outputs: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """Calculate F1 score and loss"""
        # Ensure outputs and labels have the same shape
        outputs = outputs.squeeze()
        labels = labels.squeeze()

        predictions = (torch.sigmoid(outputs) > INFERENCE_CONFIG['CLASSIFICATION_THRESHOLD']).float()
        f1 = f1_score(labels.cpu().numpy(), predictions.cpu().numpy())

        # Reshape for loss calculation
        outputs = outputs.view(-1, 1)
        labels = labels.view(-1, 1)
        loss = self.criterion(outputs, labels).item()

        return {
            'f1': f1,
            'loss': loss
        }

    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        all_outputs = []
        all_labels = []

        for img1, img2, labels in self.data_manager.train_loader:
            img1, img2 = img1.to(self.device), img2.to(self.device)
            labels = labels.to(self.device).view(-1, 1)  # Reshape labels to match output

            self.optimizer.zero_grad()
            outputs = self.model(img1, img2)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            all_outputs.append(outputs.detach())
            all_labels.append(labels)

        # Calculate metrics
        outputs = torch.cat(all_outputs)
        labels = torch.cat(all_labels)
        metrics = self._calculate_metrics(outputs, labels)
        metrics['loss'] = total_loss / len(self.data_manager.train_loader)

        return metrics

    def evaluate(self, data_loader) -> Dict[str, float]:
        """Evaluate model on given data loader"""
        self.model.eval()
        all_outputs = []
        all_labels = []
        total_loss = 0

        with torch.no_grad():
            for img1, img2, labels in data_loader:
                img1, img2 = img1.to(self.device), img2.to(self.device)
                labels = labels.to(self.device).view(-1, 1)  # Reshape labels to match output

                outputs = self.model(img1, img2)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                all_outputs.append(outputs)
                all_labels.append(labels)

        outputs = torch.cat(all_outputs)
        labels = torch.cat(all_labels)
        metrics = self._calculate_metrics(outputs, labels)
        metrics['loss'] = total_loss / len(data_loader)

        return metrics

    def save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint with metrics"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'timestamp': timestamp
        }

        # Create save directory if it doesn't exist
        save_dir = Path(PATH_CONFIG['MODELS_DIR'])
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save checkpoint
        model_name = self.model_class.__name__
        f1_str = f"{metrics['val_f1']:.3f}".replace(".", "")
        filename = f"{model_name}_F1_{f1_str}_{timestamp}.pth"

        if is_best:
            filename = f"best_{filename}"

        save_path = save_dir / filename
        torch.save(checkpoint, save_path)

        # Save readable metrics
        metrics_path = save_path.with_suffix('.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)

        self.logger.info(f"Saved checkpoint to {save_path}")

    def train(self) -> Dict[str, List[float]]:
        """Full training loop"""
        self.logger.info(f"Starting training for {self.model_class.__name__}")

        for epoch in range(TRAINING_CONFIG['EPOCHS']):
            # Train epoch
            train_metrics = self._train_epoch()

            # Evaluate
            val_metrics = self.evaluate(self.data_manager.val_loader)
            test_metrics = self.evaluate(self.data_manager.test_loader)

            # Update scheduler
            self.scheduler.step(val_metrics['loss'])
            current_lr = self.optimizer.param_groups[0]['lr']

            # Log metrics
            self.logger.info(
                f"Epoch {epoch + 1}/{TRAINING_CONFIG['EPOCHS']} "
                f"Train - Loss: {train_metrics['loss']:.4f}, F1: {train_metrics['f1']:.4f} "
                f"Val - Loss: {val_metrics['loss']:.4f}, F1: {val_metrics['f1']:.4f} "
                f"Test - Loss: {test_metrics['loss']:.4f}, F1: {test_metrics['f1']:.4f} "
                f"Learning Rate: {current_lr:.2e}"
            )

            # Check for improvement
            if val_metrics['f1'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['f1']
                self.best_model_state = self.model.state_dict()
                self.epochs_without_improvement = 0

                # Save best model
                metrics = {
                    'epoch': epoch,
                    'train_f1': train_metrics['f1'],
                    'train_loss': train_metrics['loss'],
                    'val_f1': val_metrics['f1'],
                    'val_loss': val_metrics['loss'],
                    'test_f1': test_metrics['f1'],
                    'test_loss': test_metrics['loss'],
                    'learning_rate': current_lr
                }
                self.save_checkpoint(metrics, is_best=True)
            else:
                self.epochs_without_improvement += 1

            # Early stopping checks
            if current_lr < TRAINING_CONFIG['MIN_LR']:
                self.logger.info("Learning rate too small. Stopping...")
                break

            if (self.epochs_without_improvement >=
                    TRAINING_CONFIG['EARLY_STOP_PATIENCE']):
                self.logger.info(
                    f"No improvement for {TRAINING_CONFIG['EARLY_STOP_PATIENCE']} "
                    "epochs. Stopping..."
                )
                break

        return {
            'best_val_f1': self.best_val_f1,
            'final_train_metrics': train_metrics,
            'final_val_metrics': val_metrics,
            'final_test_metrics': test_metrics
        }


def cleanup_models(models_dir: Path, keep_best_n: int = 1) -> None:
    """
    Cleanup old model checkpoints, keeping only the best N models for each architecture.

    Args:
        models_dir: Directory containing model checkpoints
        keep_best_n: Number of best models to keep for each architecture
    """
    if not models_dir.exists():
        return

    # Create backup directory
    backup_dir = models_dir / 'old_models'
    backup_dir.mkdir(exist_ok=True)

    # Group files by model architecture
    model_files: Dict[str, List[Tuple[Path, float]]] = {}

    # Pattern to extract model name and F1 score
    # todo Files with the required underscore format are not matching the pattern:
    #  best_KinshipNet_2_F1_0827_20241030_0542
    # pattern: Pattern = re.compile(r"(best_)?([A-Za-z_]+)_F1_(\d+)")
    pattern: Pattern = re.compile(r"(best_)?([^_F1]+(?:_[^_F1]+)*)_F1_(\d+)")

    # Collect all model files
    for file_path in models_dir.glob("*.pth"):
        match = pattern.search(file_path.name)
        if match:
            is_best = bool(match.group(1))
            model_name = match.group(2)
            f1_score = float(match.group(3)) / 1000  # Convert back to float

            if model_name not in model_files:
                model_files[model_name] = []
            model_files[model_name].append((file_path, f1_score, is_best))

    # Process each model architecture
    for model_name, files in model_files.items():
        # Sort by F1 score and best flag (best models first)
        files.sort(key=lambda x: (x[2], x[1]), reverse=True)

        # Keep the best N models
        keep_files = files[:keep_best_n]
        move_files = files[keep_best_n:]

        # Move older models to backup
        for file_path, _, _ in move_files:
            # Move both .pth and .json files
            for ext in ['.pth', '.json']:
                src_path = file_path.with_suffix(ext)
                if src_path.exists():
                    dst_path = backup_dir / src_path.name
                    shutil.move(str(src_path), str(dst_path))

        logging.info(f"Cleaned up {len(move_files)} old checkpoints for {model_name}")



# В класс ModelTrainer добавим метод для очистки памяти:
def cleanup(self):
    """Clean up model resources"""
    if hasattr(self, 'model'):
        del self.model
    if hasattr(self, 'optimizer'):
        del self.optimizer
    if hasattr(self, 'scheduler'):
        del self.scheduler
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


# В функцию train_all_models() добавим очистку после каждой модели:

def train_all_models():
    """Train all available models"""
    data_manager = KinFaceDataManager()
    model_classes = [KinshipNet, KinshipNet_2, SimpleNet]
    results = {}

    for model_class in model_classes:
        logging.info(f"Training {model_class.__name__}...")
        trainer = ModelTrainer(model_class, data_manager)

        try:
            model_results = trainer.train()
            results[model_class.__name__] = model_results

            # Log final results
            logging.info(f"Final Results for {model_class.__name__}:")
            logging.info(f"Best Validation F1: {model_results['best_val_f1']:.4f}")
            logging.info("Final Metrics:")
            logging.info(
                f"Train - F1: {model_results['final_train_metrics']['f1']:.4f}, "
                f"Loss: {model_results['final_train_metrics']['loss']:.4f}")
            logging.info(
                f"Val - F1: {model_results['final_val_metrics']['f1']:.4f}, "
                f"Loss: {model_results['final_val_metrics']['loss']:.4f}")
            logging.info(
                f"Test - F1: {model_results['final_test_metrics']['f1']:.4f}, "
                f"Loss: {model_results['final_test_metrics']['loss']:.4f}")

        finally:
            # Cleanup resources
            # trainer.cleanup()
            pass

        # Cleanup old model checkpoints after each model training
        cleanup_models(PATH_CONFIG['MODELS_DIR'])

    return results


# В if __name__ == "__main__": добавим обработку исключений:

def clean_old_models():
    # Clean up old_models directory
    old_models_dir = PATH_CONFIG['MODELS_DIR'] / 'old_models'
    if old_models_dir.exists():
        try:
            # Delete all files in old_models directory
            for file_path in old_models_dir.glob('*.*'):
                try:
                    file_path.unlink()  # Delete file
                    logging.info(f"Deleted old model file: {file_path.name}")
                except Exception as e:
                    logging.error(f"Error deleting file {file_path}: {e}")

            # # Delete the directory itself
            # old_models_dir.rmdir()
            # logging.info("Removed old_models directory")

            # Recreate empty directory
            old_models_dir.mkdir(exist_ok=True)
            logging.info("Created new empty old_models directory")
        except Exception as e:
            logging.error(f"Error cleaning up old_models directory: {e}")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    try:
        results = train_all_models()
    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
        # Cleanup on interrupt
        cleanup_models(PATH_CONFIG['MODELS_DIR'])
    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        # Cleanup on error
        cleanup_models(PATH_CONFIG['MODELS_DIR'])
    finally:
        # Final cleanup
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    clean_old_models()

if __name__ == "__main__":
    main()
