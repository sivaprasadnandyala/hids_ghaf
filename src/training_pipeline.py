import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import traceback
import numpy as np
import json
import logging
import joblib
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix

from .config import config
from .model_utils import Autoencoder, normalize_data
from .preprocessing import preprocess_data
from .monitoring_utils import SystemMonitor, AnomalyVisualizer
from .logging_setup import get_logger

from typing import TypedDict, List, Dict

class ThresholdDict(TypedDict):
    trained: float
    unseen: float

class TrainingMetrics(TypedDict):
    train_loss: List[float]
    val_loss: List[float]
    best_val_loss: float
    best_epoch: int
    thresholds: ThresholdDict


class TrainingPipeline:
    def __init__(self, force_reprocess=True):
    #def __init__(self):

        self.training_metrics: TrainingMetrics = {
            'train_loss': [],
            'val_loss': [],
            'best_val_loss': float('inf'),
            'best_epoch': 0,
            'thresholds': {
                'trained': 0.0,  # Initialize with float instead of None
                'unseen': 0.0  # Initialize with float instead of None
            }
        }

        self.force_reprocess = force_reprocess

        """Initialize the training pipeline"""
        self.logger = get_logger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Add these new attributes
        self.start_time = datetime.now()
        self.total_samples = 0
        self.train_samples = 0
        self.val_samples = 0
        self.feature_dim = 0

        # Initialize components
        self.model = None
        self.scaler = None
        self.system_monitor = SystemMonitor()
        self.visualizer = AnomalyVisualizer()

        # Training metrics
        self.training_metrics = {
            'train_loss': [],
            'val_loss': [],
            'best_val_loss': float('inf'),
            'best_epoch': 0,
            'thresholds': {'trained': None, 'unseen': None}
        }

        # Initialize directories
        self._setup_directories()

        # Set random seeds
        self._set_random_seeds()

    def _set_random_seeds(self, seed: int = 42):
        """Set random seeds for reproducibility"""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    def _setup_directories(self):
        """Create necessary directories"""
        dirs = [
            config.DATA_STORAGE['models'],
            config.DATA_STORAGE['results'],
            os.path.join(config.DATA_STORAGE['results'], 'training'),
            os.path.join(config.DATA_STORAGE['models'], 'checkpoints')
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)
            self.logger.info(f"Created directory: {d}")

    def validate_feature_files(self, feature_files: List[Path]) -> Optional[int]:
        """Validate feature files and return expected dimension"""
        dimensions = []
        for file in feature_files:
            try:
                data = np.load(file)
                if len(data.shape) != 2:
                    self.logger.error(f"Invalid data shape in {file}: {data.shape}")
                    continue
                dimensions.append(data.shape[1])
            except Exception as e:
                self.logger.error(f"Error loading {file}: {e}")
                continue

        if not dimensions:
            return None

        # Check if all dimensions match
        if len(set(dimensions)) > 1:
            self.logger.error(f"Inconsistent feature dimensions found: {set(dimensions)}")
            return None

        return dimensions[0]


    def load_training_data(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Load and preprocess training data with forced SSG generation"""
        try:
            self.logger.info("Loading training data...")

            # First, verify directory structure
            process_dir = os.path.join(config.DATA_STORAGE['training_data'], 'logs')
            self.logger.info(f"Looking for training data in: {process_dir}")

            # List all files in the directory
            if os.path.exists(process_dir):
                log_files = list(Path(process_dir).glob('*.log'))
                self.logger.info(f"Found {len(log_files)} log files:")
                for log_file in log_files:
                    self.logger.info(f"- {log_file.name}")
            else:
                self.logger.error(f"Training data directory not found: {process_dir}")
                return None

            # Process each training process with flexible name matching
            processed_features = []
            feature_dim = None
            total_windows = 0

            for process_name in config.TRAINING_PROCESSES:
                base_name = process_name.split('/')[-1].split()[0].lower()  # Get base process name

                # Find matching log files
                log_files = []
                log_patterns = [
                    f"*{base_name}*.log",  # Match base name
                    f"*{process_name.replace('/', '_')}*.log"  # Match full path
                ]

                for pattern in log_patterns:
                    log_files.extend(list(Path(process_dir).glob(pattern)))

                if not log_files:
                    self.logger.warning(f"No log files found for process: {process_name}")
                    continue

                self.logger.info(f"Processing logs for {process_name}")
                self.logger.info(f"Found log files: {[f.name for f in log_files]}")

                for log_file in sorted(log_files):
                    try:
                        # Read syscalls from log in text format
                        syscalls = []
                        with open(log_file, 'r') as f:
                            for line in f:
                                try:
                                    # Parse syscall line
                                    # Expected format: "binary syscall(args)"
                                    line = line.strip()
                                    if not line:
                                        continue

                                    parts = line.split(' ', 1)
                                    if len(parts) != 2:
                                        continue

                                    binary, syscall_info = parts
                                    syscall_name = syscall_info.split('(')[0]
                                    args_str = syscall_info[syscall_info.find('(') + 1:syscall_info.rfind(')')]
                                    args = [arg.strip() for arg in args_str.split(',') if arg.strip()]

                                    syscall = {
                                        'binary': binary,
                                        'name': syscall_name,
                                        'args': args,
                                        'timestamp': int(datetime.now().timestamp())
                                    }
                                    syscalls.append(syscall)
                                except Exception as e:
                                    self.logger.debug(f"Error parsing line in {log_file}: {e}")
                                    continue

                        if syscalls:
                            self.logger.info(f"Processing {len(syscalls)} syscalls from {log_file}")

                            # Process syscalls with interval tracking for SSG generation
                            features = self.process_syscalls(
                                syscalls=syscalls,
                                process_name=process_name,
                                interval_base=total_windows,
                                mode='train'
                            )

                            if features is not None:
                                if feature_dim is None:
                                    feature_dim = features.shape[1]
                                elif features.shape[1] != feature_dim:
                                    self.logger.error(
                                        f"Inconsistent feature dimensions: {features.shape[1]} != {feature_dim}")
                                    continue

                                processed_features.append(features)
                                total_windows += len(features)
                                self.logger.info(f"Extracted {len(features)} feature windows from {log_file}")

                    except Exception as e:
                        self.logger.error(f"Error processing {log_file}: {e}")
                        continue

            if not processed_features:
                self.logger.error("No features extracted from training data")
                return None

            # Combine all features
            X = np.vstack(processed_features)
            self.logger.info(f"Total feature shape: {X.shape}")

            # Save processed features for future use
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            feature_dir = os.path.join(config.DATA_STORAGE['training_data'], 'features')
            os.makedirs(feature_dir, exist_ok=True)

            feature_file = os.path.join(feature_dir, f'processed_features_{timestamp}.npy')
            np.save(feature_file, X)
            self.logger.info(f"Saved processed features to {feature_file}")

            # Normalize the data
            self.scaler = StandardScaler()
            X_normalized = self.scaler.fit_transform(X)

            # Save the scaler
            scaler_path = os.path.join(config.DATA_STORAGE['models'], 'scaler.pkl')
            joblib.dump(self.scaler, scaler_path)
            self.logger.info(f"Saved scaler to {scaler_path}")

            # Split into train and validation
            train_size = int(len(X_normalized) * (1 - config.VALIDATION_SPLIT))
            indices = np.random.permutation(len(X_normalized))

            X_train = X_normalized[indices[:train_size]]
            X_val = X_normalized[indices[train_size:]]

            # Store dataset information
            self.total_samples = len(X)
            self.train_samples = len(X_train)
            self.val_samples = len(X_val)
            self.feature_dim = feature_dim

            self.logger.info(f"Prepared {self.train_samples} training and {self.val_samples} validation samples")
            self.logger.info(f"Feature dimension: {self.feature_dim}")

            return X_train, X_val

        except Exception as e:
            self.logger.error(f"Error loading training data: {e}")
            self.logger.error(traceback.format_exc())
            return None

    def process_syscalls(self, syscalls: List[Dict], process_name: str, interval_base: int, mode: str) -> Optional[
        np.ndarray]:
        """Process syscalls and generate features with proper interval tracking"""
        try:
            window_size = config.MIN_SAMPLES_REQUIRED
            stride = window_size // 4  # 75% overlap
            features = []

            # Create windows
            for i in range(0, len(syscalls) - window_size + 1, stride):
                window = syscalls[i:i + window_size]

                if len(window) == window_size:
                    # Calculate current interval
                    current_interval = interval_base + (i // stride)

                    # Process window
                    processed_data = preprocess_data(
                        collected_syscalls=[window],
                        interval_counter=current_interval,
                        mode=mode
                    )

                    if processed_data is not None and len(processed_data) > 0:
                        features.append(processed_data)

            if not features:
                return None

            return np.vstack(features)

        except Exception as e:
            self.logger.error(f"Error processing syscalls for {process_name}: {e}")
            self.logger.error(traceback.format_exc())
            return None

    def initialize_model(self, input_dim: int):
        """Initialize the autoencoder model"""
        try:
            self.model = Autoencoder(input_dim).to(self.device)
            num_params = sum(p.numel() for p in self.model.parameters())
            self.logger.info(f"Initialized model with {num_params} parameters")
            self.logger.info(f"Model architecture:\n{self.model}")
        except Exception as e:
            self.logger.error(f"Error initializing model: {e}")
            raise



    def train(self, X_train: np.ndarray, X_val: np.ndarray) -> bool:
        """Train the model with properly initialized metrics"""
        try:
            # Create data loaders
            train_dataset = TensorDataset(torch.FloatTensor(X_train))
            val_dataset = TensorDataset(torch.FloatTensor(X_val))

            # Ensure minimum batch size
            batch_size = min(config.MODEL_CONFIG['training']['batch_size'], len(X_train))
            self.logger.info(f"Using batch size: {batch_size}")

            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False
            )

            # Initialize training components
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=config.MODEL_CONFIG['training']['learning_rate'],
                weight_decay=config.MODEL_CONFIG['training']['weight_decay']
            )

            # Use cosine annealing scheduler
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=config.MODEL_CONFIG['scheduler']['T_0'],
                T_mult=config.MODEL_CONFIG['scheduler']['T_mult'],
                eta_min=config.MODEL_CONFIG['scheduler']['eta_min']
            )

            criterion = nn.MSELoss()

            # Initialize training metrics
            self.training_metrics = {
                'train_loss': [],
                'val_loss': [],
                'learning_rates': [],
                'best_val_loss': float('inf'),
                'best_epoch': 0,
                'thresholds': {'trained': None, 'unseen': None}
            }

            self.logger.info("Starting training...")
            self.logger.info(f"Training samples: {len(X_train)}")
            self.logger.info(f"Validation samples: {len(X_val)}")
            self.logger.info(f"Initial learning rate: {config.MODEL_CONFIG['training']['learning_rate']}")

            best_val_loss = float('inf')
            early_stopping_counter = 0
            min_epochs = config.MODEL_CONFIG['training']['min_epochs']
            training_start_time = datetime.now()

            for epoch in range(config.MODEL_CONFIG['training']['num_epochs']):
                epoch_start_time = datetime.now()

                # Training phase
                self.model.train()
                train_loss = 0.0
                batch_count = 0

                # Learning rate warmup
                if epoch < config.MODEL_CONFIG['training']['warmup_epochs']:
                    warmup_factor = epoch / config.MODEL_CONFIG['training']['warmup_epochs']
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = config.MODEL_CONFIG['training']['learning_rate'] * warmup_factor

                for batch_idx, batch in enumerate(train_loader):
                    data = batch[0].to(self.device)
                    optimizer.zero_grad()

                    output = self.model(data)
                    loss = criterion(output, data)
                    loss.backward()

                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()

                    train_loss += loss.item()
                    batch_count += 1

                # Calculate average training loss
                avg_train_loss = train_loss / batch_count
                self.training_metrics['train_loss'].append(avg_train_loss)

                # Validation phase
                val_loss = self._validate(val_loader, criterion)
                self.training_metrics['val_loss'].append(val_loss)

                # Get current learning rate
                current_lr = optimizer.param_groups[0]['lr']
                self.training_metrics['learning_rates'].append(current_lr)

                # Update scheduler
                if epoch >= config.MODEL_CONFIG['training']['warmup_epochs']:
                    scheduler.step()

                # Early stopping logic with model saving
                if val_loss < best_val_loss:
                    improvement = (best_val_loss - val_loss) / best_val_loss * 100
                    best_val_loss = val_loss
                    self.training_metrics['best_val_loss'] = val_loss
                    self.training_metrics['best_epoch'] = epoch
                    self._save_model(is_best=True)
                    early_stopping_counter = 0

                    self.logger.info(
                        f"Epoch {epoch + 1}: Validation loss improved by {improvement:.2f}% "
                        f"New best: {val_loss:.6f}"
                    )
                else:
                    early_stopping_counter += 1

                # Calculate epoch time
                epoch_time = datetime.now() - epoch_start_time

                # Log progress
                if (epoch + 1) % 5 == 0 or epoch == 0:
                    self.logger.info(
                        f"Epoch [{epoch + 1}/{config.MODEL_CONFIG['training']['num_epochs']}] "
                        f"Train Loss: {avg_train_loss:.6f}, "
                        f"Val Loss: {val_loss:.6f}, "
                        f"LR: {current_lr:.2e}, "
                        f"Time: {epoch_time}"
                    )

                    # Save training plot periodically
                    self._save_training_plot(
                        os.path.join(config.DATA_STORAGE['results'], 'training'),
                        datetime.now().strftime("%Y%m%d_%H%M%S")
                    )

                # Early stopping check with minimum epochs requirement
                if epoch >= min_epochs and early_stopping_counter >= config.MODEL_CONFIG['training'][
                    'early_stopping_patience']:
                    self.logger.info(
                        f"Early stopping triggered at epoch {epoch + 1}. "
                        f"No improvement for {config.MODEL_CONFIG['training']['early_stopping_patience']} epochs"
                    )
                    break

                # Just before early stopping triggers:
                if epoch >= min_epochs and early_stopping_counter >= config.MODEL_CONFIG['training'][
                    'early_stopping_patience']:
                    self.logger.info(
                        f"Early stopping triggered at epoch {epoch + 1}. "
                        f"No improvement for {config.MODEL_CONFIG['training']['early_stopping_patience']} epochs"
                    )
                    break

            # Calculate total training time
            total_training_time = datetime.now() - training_start_time

            # Calculate final thresholds - FIXED HERE
            self._calculate_thresholds(train_loader)  # Remove the arguments

            # Final logging
            self.logger.info("\nTraining Summary:")
            self.logger.info("=" * 50)
            self.logger.info(f"Total training time: {total_training_time}")
            self.logger.info(f"Best epoch: {self.training_metrics['best_epoch']}")
            self.logger.info(f"Best validation loss: {self.training_metrics['best_val_loss']:.6f}")
            self.logger.info(f"Final learning rate: {current_lr:.2e}")
            self.logger.info(f"Total epochs trained: {epoch + 1}")
            self.logger.info(
                f"Average time per epoch: {total_training_time / (epoch + 1)}"
            )

            # Save final training plot
            self._save_training_plot(
                os.path.join(config.DATA_STORAGE['results'], 'training'),
                datetime.now().strftime("%Y%m%d_%H%M%S")
            )

            # Validate model performance
            self.validate_model_performance()

            # Calculate training metrics using model thresholds
            self.model.eval()
            with torch.no_grad():
                train_tensor = torch.FloatTensor(X_train).to(self.device)
                train_output = self.model(train_tensor)
                train_errors = torch.mean(torch.pow(train_tensor - train_output, 2), dim=1).cpu().numpy()

                # Use the trained thresholds
                trained_predictions = (train_errors > self.training_metrics['thresholds']['trained']).astype(int)
                # All training samples are normal (label 0)
                train_labels = np.zeros(len(X_train))

                # Calculate confusion matrix
                tn, fp, fn, tp = confusion_matrix(train_labels, trained_predictions).ravel()

                # Calculate metrics with proper error handling
                total = tn + fp + fn + tp

                # Accuracy: Correctly classified samples (both normal and anomalous)
                accuracy = (tn + tp) / total if total > 0 else 0.0

                # For normal class (0):
                # Precision for normal class: TN / (TN + FN)
                normal_precision = tn / (tn + fn) if (tn + fn) > 0 else 1.0

                # Recall for normal class: TN / (TN + FP)
                normal_recall = tn / (tn + fp) if (tn + fp) > 0 else 1.0

                # F1 for normal class
                normal_f1 = 2 * (normal_precision * normal_recall) / (normal_precision + normal_recall) if (normal_precision + normal_recall) > 0 else 0.0

                # Specificity remains the same: TN / (TN + FP)
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 1.0

                # Print training metrics with context
                print("\nFinal Training Metrics:")
                print("=" * 50)
                print(f"Total Samples: {total}")
                print(f"Anomaly Rate: {(fp) / total:.4f}")  # Only FP are predicted anomalies in training set

                print("\nPerformance Metrics:")
                print("-" * 30)
                print(f"Accuracy: {accuracy:.4f} (Correct classifications / Total)")
                print(f"Normal Class Precision: {normal_precision:.4f} (TN / (TN + FN))")
                print(f"Normal Class Recall: {normal_recall:.4f} (TN / (TN + FP))")
                print(f"Normal Class F1: {normal_f1:.4f}")
                print(f"Specificity: {specificity:.4f} (TN / (TN + FP))")

                print("\nConfusion Matrix:")
                print("-" * 30)
                print(f"True Negatives (TN): {tn} (Correct normal predictions)")
                print(f"False Positives (FP): {fp} (Normal samples predicted as anomalies)")
                print(f"False Negatives (FN): {fn} (Should be 0 for training)")
                print(f"True Positives (TP): {tp} (Should be 0 for training)")

                print("\nError Statistics:")
                print("-" * 30)
                print(f"Mean Reconstruction Error: {np.mean(train_errors):.6f}")
                print(f"Error Std Dev: {np.std(train_errors):.6f}")
                print(f"Used Threshold: {self.training_metrics['thresholds']['trained']:.6f}")
                print(f"Max Error: {np.max(train_errors):.6f}")
                print(f"Min Error: {np.min(train_errors):.6f}")

                # Add metrics to training metrics dictionary
                self.training_metrics.update({
                    'final_metrics': {
                        'total_samples': int(total),
                        'anomaly_rate': float(fp / total if total > 0 else 0.0),
                        'accuracy': float(accuracy),
                        'normal_precision': float(normal_precision),
                        'normal_recall': float(normal_recall),
                        'normal_f1': float(normal_f1),
                        'specificity': float(specificity),
                        'mean_error': float(np.mean(train_errors)),
                        'error_std': float(np.std(train_errors)),
                        'confusion_matrix': {
                            'tn': int(tn),
                            'fp': int(fp),
                            'fn': int(fn),
                            'tp': int(tp)
                        }
                    }
                })

            return True

        except Exception as e:
            self.logger.error(f"Error during training: {e}")
            self.logger.error(traceback.format_exc())
            return False

            return True

        except Exception as e:
            self.logger.error(f"Error during training: {e}")
            self.logger.error(traceback.format_exc())
            return False

    def _save_checkpoint(self, epoch: int, val_loss: float):
        """Save intermediate checkpoints"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'train_loss': self.training_metrics['train_loss'][-1],
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }

        checkpoint_path = os.path.join(
            config.DATA_STORAGE['models'],
            'checkpoints',
            f'checkpoint_epoch_{epoch}.pth'
        )
        torch.save(checkpoint, checkpoint_path)

    def clean_training_artifacts(self):
        """Clean up old training artifacts"""
        try:
            # Clean up old model files
            model_dir = os.path.join(config.DATA_STORAGE['models'], 'final')
            metrics_dir = os.path.join(config.DATA_STORAGE['results'], 'training')

            # Remove old model files
            for f in Path(model_dir).glob('best_model_*.pth'):
                try:
                    os.remove(f)
                except Exception as e:
                    self.logger.error(f"Error removing old model file {f}: {e}")

            # Remove old metrics files
            for f in Path(metrics_dir).glob('training_metrics_*.json'):
                try:
                    os.remove(f)
                except Exception as e:
                    self.logger.error(f"Error removing old metrics file {f}: {e}")

            self.logger.info("Cleaned up old training artifacts")

        except Exception as e:
            self.logger.error(f"Error cleaning training artifacts: {e}")


    def _validate(self, val_loader: DataLoader, criterion: nn.Module) -> float:
        """Validate the model"""
        self.model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                data = batch[0].to(self.device)
                output = self.model(data)
                val_loss += criterion(output, data).item()

        return val_loss / len(val_loader)

    # def _calculate_thresholds(self, train_loader: DataLoader) -> None:
    #     """Calculate anomaly detection thresholds using global config"""
    #     self.logger.info("Calculating detection thresholds...")
    #     try:
    #         self.model.eval()
    #         reconstruction_errors: List[float] = []
    #
    #         with torch.no_grad():
    #             for batch in train_loader:
    #                 data = batch[0].to(self.device)
    #                 outputs = self.model(data)
    #                 errors = torch.mean(torch.pow(data - outputs, 2), dim=1)
    #                 reconstruction_errors.extend(errors.cpu().numpy().tolist())
    #
    #         reconstruction_errors_array = np.array(reconstruction_errors)
    #
    #         # Get threshold values from global config
    #         trained_percentile = config.MODEL_CONFIG['thresholds']['trained_percentile']
    #         unseen_multiplier = config.MODEL_CONFIG['thresholds']['unseen_multiplier']
    #
    #         # Calculate thresholds
    #         trained_threshold = float(np.percentile(reconstruction_errors_array, trained_percentile))
    #         unseen_threshold = float(trained_threshold * unseen_multiplier)
    #
    #         # Update training metrics with thresholds
    #         self.training_metrics['thresholds'] = {'trained': trained_threshold,'unseen': unseen_threshold}
    #
    #         # Save thresholds
    #         threshold_path = os.path.join(config.DATA_STORAGE['models'], 'final', 'thresholds.npy')
    #         np.save(threshold_path, np.array([trained_threshold, unseen_threshold]))
    #
    #         self.logger.info(
    #             f"Calculated thresholds - Trained: {trained_threshold:.6f}, "
    #             f"Unseen: {unseen_threshold:.6f}"
    #         )
    #
    #     except Exception as e:
    #         self.logger.error(f"Error calculating thresholds: {e}")
    #         self.logger.error(traceback.format_exc())
    #         # Set default thresholds in case of error
    #         self.training_metrics['thresholds'] = {'trained': 0.0, 'unseen': 0.0}

    def _calculate_thresholds(self, train_loader: DataLoader) -> None:
        """Calculate anomaly detection thresholds using more robust statistics"""
        try:
            self.logger.info("Calculating detection thresholds...")
            reconstruction_errors = []

            self.model.eval()
            with torch.no_grad():
                for batch in train_loader:
                    data = batch[0].to(self.device)
                    outputs = self.model(data)
                    errors = torch.mean(torch.pow(data - outputs, 2), dim=1)
                    reconstruction_errors.extend(errors.cpu().numpy().tolist())

            reconstruction_errors = np.array(reconstruction_errors)

            # Calculate robust statistics
            mean_error = np.mean(reconstruction_errors)
            std_error = np.std(reconstruction_errors)
            median_error = np.median(reconstruction_errors)
            q75, q25 = np.percentile(reconstruction_errors, [75, 25])
            iqr = q75 - q25

            # Calculate thresholds using multiple methods
            threshold_std = mean_error + 2 * std_error
            threshold_iqr = median_error + 1.5 * iqr
            threshold_percentile = np.percentile(reconstruction_errors, 95)

            # Use the maximum of these methods for more robust thresholding
            trained_threshold = max(threshold_std, threshold_iqr, threshold_percentile)

            # Set a more conservative threshold for unseen data
            unseen_threshold = trained_threshold * 1.5

            self.training_metrics['thresholds'] = {'trained': float(trained_threshold),'unseen': float(unseen_threshold)}

            # Save thresholds
            threshold_path = os.path.join(config.DATA_STORAGE['models'], 'final', 'thresholds.npy')
            np.save(threshold_path, np.array([trained_threshold, unseen_threshold]))

            self.logger.info(f"Calculated thresholds:")
            self.logger.info(f"- Trained threshold: {trained_threshold:.6f}")
            self.logger.info(f"- Unseen threshold: {unseen_threshold:.6f}")
            self.logger.info(f"- Mean error: {mean_error:.6f}")
            self.logger.info(f"- Std error: {std_error:.6f}")
            self.logger.info(f"- Median error: {median_error:.6f}")
            self.logger.info(f"- IQR: {iqr:.6f}")

        except Exception as e:
            self.logger.error(f"Error calculating thresholds: {e}")
            self.logger.error(traceback.format_exc())


    def _cleanup_previous_runs(self):
        """Clean up artifacts from previous runs"""
        try:
            # Directories to clean
            final_dir = os.path.join(config.DATA_STORAGE['models'], 'final')
            metrics_dir = os.path.join(config.DATA_STORAGE['results'], 'training')

            # Files to clean up
            patterns = {
                final_dir: ['best_model_*.pth', 'scaler.pkl', 'thresholds.npy'],
                metrics_dir: ['training_metrics_*.json', 'training_plot_*.png']
            }

            for directory, file_patterns in patterns.items():
                if os.path.exists(directory):
                    for pattern in file_patterns:
                        for file in Path(directory).glob(pattern):
                            try:
                                os.remove(file)
                                self.logger.debug(f"Removed old file: {file}")
                            except Exception as e:
                                self.logger.error(f"Error removing {file}: {e}")

            self.logger.info("Cleaned up previous training artifacts")

        except Exception as e:
            self.logger.error(f"Error in cleanup: {e}")

    def _save_model(self, is_best: bool = False):
        """Save model and training artifacts with improved cleanup"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            models_dir = config.DATA_STORAGE['models']
            results_dir = config.DATA_STORAGE['results']

            # Ensure directories exist
            final_dir = os.path.join(models_dir, 'final')
            metrics_dir = os.path.join(results_dir, 'training')
            for d in [final_dir, metrics_dir]:
                os.makedirs(d, exist_ok=True)

            if is_best:
                # Clean up previous files before saving
                self._cleanup_previous_runs()

                # Save model
                model_path = os.path.join(final_dir, f'best_model_{timestamp}.pth')
                torch.save(self.model.state_dict(), model_path)

                # Save scaler if it exists
                if hasattr(self, 'scaler') and self.scaler is not None:
                    scaler_path = os.path.join(final_dir, 'scaler.pkl')
                    joblib.dump(self.scaler, scaler_path)

                # Save comprehensive metrics
                metrics_data = {
                    'timestamp': timestamp,
                    'training_metrics': {
                        'train_loss': [float(x) for x in self.training_metrics['train_loss']],
                        'val_loss': [float(x) for x in self.training_metrics['val_loss']],
                        'best_val_loss': float(self.training_metrics['best_val_loss']),
                        'best_epoch': self.training_metrics['best_epoch'],
                        'thresholds': self.training_metrics.get('thresholds', {'trained': None, 'unseen': None})
                    },
                    'model_info': {
                        'total_params': sum(p.numel() for p in self.model.parameters()),
                        'trainable_params': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                        'architecture': str(self.model)
                    },
                    'training_info': {
                        'total_time': str(datetime.now() - self.start_time),
                        'epochs': len(self.training_metrics['train_loss']),
                        'learning_rate': config.MODEL_CONFIG['training']['learning_rate'],  # Fixed here
                        'batch_size': config.MODEL_CONFIG['training']['batch_size'],  # And here
                        'early_stopping_patience': config.MODEL_CONFIG['training']['early_stopping_patience'],
                        # And here
                        'final_train_loss': float(self.training_metrics['train_loss'][-1]),
                        'final_val_loss': float(self.training_metrics['val_loss'][-1])
                    },
                    'data_info': {
                        'total_samples': self.total_samples,
                        'training_samples': self.train_samples,
                        'validation_samples': self.val_samples,
                        'feature_dimension': self.feature_dim
                    }
                }

                metrics_path = os.path.join(metrics_dir, f'training_metrics_{timestamp}.json')
                with open(metrics_path, 'w') as f:
                    json.dump(metrics_data, f, indent=4)

                # Save training plot
                self._save_training_plot(metrics_dir, timestamp)

                self.logger.info(f"Saved best model and metrics (Epoch {self.training_metrics['best_epoch']})")

        except Exception as e:
            self.logger.error(f"Error saving model and artifacts: {e}")
            self.logger.error(traceback.format_exc())


    def _augment_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """Apply simple data augmentation"""
        # Add small random noise
        noise = torch.randn_like(batch) * 0.01
        batch = batch + noise

        # Random feature scaling
        scale = torch.rand(batch.size(-1)) * 0.1 + 0.95  # 0.95-1.05
        batch = batch * scale

        return batch

    def validate_model_performance(self):
        """Validate model performance and suggest improvements"""
        try:
            train_losses = self.training_metrics['train_loss']
            val_losses = self.training_metrics['val_loss']

            loss_diff = val_losses[-1] - train_losses[-1]
            loss_ratio = val_losses[-1] / train_losses[-1]

            self.logger.info("\nModel Performance Analysis:")
            self.logger.info(f"Loss Difference (Val-Train): {loss_diff:.6f}")
            self.logger.info(f"Loss Ratio (Val/Train): {loss_ratio:.6f}")

            # Analyze for overfitting
            if loss_ratio > 1.5:
                self.logger.warning("Possible overfitting detected")
                self.logger.info("Suggestions:")
                self.logger.info("- Increase dropout rate")
                self.logger.info("- Add L2 regularization")
                self.logger.info("- Reduce model complexity")

            # Analyze convergence
            if self.training_metrics['best_epoch'] > config.MODEL_CONFIG['training']['num_epochs'] - 10:
                self.logger.warning("Model might benefit from longer training")
                self.logger.info("Consider increasing num_epochs")

            # Analyze learning dynamics
            if loss_ratio < 0.1:
                self.logger.warning("Possible underfitting detected")
                self.logger.info("Suggestions:")
                self.logger.info("- Increase model capacity")
                self.logger.info("- Train for more epochs")
                self.logger.info("- Increase learning rate")

            # Check early stopping impact
            if self.training_metrics['best_epoch'] < config.MODEL_CONFIG['training']['min_epochs']:
                self.logger.warning("Training stopped too early")
                self.logger.info("Consider adjusting early stopping parameters")

            return True

        except Exception as e:
            self.logger.error(f"Error in performance validation: {e}")
            self.logger.error(traceback.format_exc())
            return False
    def _save_training_plot(self, save_dir: str, timestamp: str):
        """Save training history plot with proper scaling"""
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[2, 1])

            # Clear any existing plots
            ax1.clear()
            ax2.clear()

            # Plot losses only if we have more than one epoch
            epochs = range(1, len(self.training_metrics['train_loss']) + 1)

            if len(epochs) > 1:
                # Plot training and validation loss
                ax1.plot(epochs, self.training_metrics['train_loss'], 'b-', label='Training Loss')
                ax1.plot(epochs, self.training_metrics['val_loss'], 'r-', label='Validation Loss')

                # Plot best point only if we have it
                if self.training_metrics['best_epoch'] is not None:
                    best_epoch = self.training_metrics['best_epoch']
                    best_val_loss = self.training_metrics['best_val_loss']
                    ax1.plot(best_epoch + 1, best_val_loss, 'go', label='Best Model')

                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Loss')
                ax1.set_title('Training History')
                ax1.grid(True)
                ax1.legend()

                # Plot learning rate
                if self.training_metrics.get('learning_rates'):
                    ax2.plot(epochs, self.training_metrics['learning_rates'], 'g-', label='Learning Rate')
                    ax2.set_xlabel('Epoch')
                    ax2.set_ylabel('Learning Rate')
                    ax2.set_yscale('log')
                    ax2.grid(True)
                    ax2.legend()
            else:
                ax1.text(0.5, 0.5, 'Insufficient epochs for plotting',
                         horizontalalignment='center', verticalalignment='center')

            plt.tight_layout()
            plot_path = os.path.join(save_dir, f'training_plot_{timestamp}.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            self.logger.info(f"Saved training plot to {plot_path}")

        except Exception as e:
            self.logger.error(f"Error saving training plot: {e}")
            self.logger.error(traceback.format_exc())


def verify_training_results():
    """Verify training results and model artifacts"""
    try:
        print("\nVerifying Training Results...")
        print("=" * 50)

        # Base directories
        model_dir = config.DATA_STORAGE['models']
        results_dir = config.DATA_STORAGE['results']

        # Check model files
        model_files = list(Path(f"{model_dir}/final").glob("best_model_*.pth"))
        if model_files:
            print("\nFound Model Files:")
            for f in model_files:
                print(f"- {f.name} ({f.stat().st_size / 1024:.2f} KB)")
        else:
            print("No model files found!")

        # Check thresholds
        threshold_file = Path(f"{model_dir}/final/thresholds.npy")
        if threshold_file.exists():
            thresholds = np.load(threshold_file)
            print("\nThresholds:")
            print(f"- Trained: {thresholds[0]:.6f}")
            print(f"- Unseen: {thresholds[1]:.6f}")
        else:
            print("No threshold file found!")

        # Check training metrics
        metrics_files = list(Path(f"{results_dir}/training").glob("training_metrics_*.json"))
        if metrics_files:
            print("\nTraining Metrics Files:")
            for f in metrics_files:
                try:
                    with open(f, 'r') as file:
                        metrics = json.load(file)
                        print(f"\nMetrics from {f.name}:")
                        # Extract training metrics with proper nesting
                        training_metrics = metrics.get('training_metrics', {})
                        print(f"- Best Validation Loss: {training_metrics.get('val_loss', [])[-1]:.6f}")
                        print(f"- Best Epoch: {training_metrics.get('best_epoch', 0)}")
                        print(f"- Final Training Loss: {training_metrics.get('train_loss', [])[-1]:.6f}")

                        # Print additional information if available
                        if 'training_info' in metrics:
                            print("\nTraining Info:")
                            training_info = metrics['training_info']
                            print(f"- Total Time: {training_info.get('total_time', 'N/A')}")
                            print(f"- Total Epochs: {training_info.get('epochs', 0)}")

                        if 'data_info' in metrics:
                            print("\nData Info:")
                            data_info = metrics['data_info']
                            print(f"- Total Samples: {data_info.get('total_samples', 0)}")
                            print(f"- Training Samples: {data_info.get('training_samples', 0)}")
                            print(f"- Validation Samples: {data_info.get('validation_samples', 0)}")
                            print(f"- Feature Dimension: {data_info.get('feature_dimension', 0)}")
                except Exception as e:
                    print(f"Error reading metrics file {f.name}: {str(e)}")
                    continue
        else:
            print("No metrics files found!")

        # Check logs
        log_files = list(Path(config.LOG_DIR).glob("*.log"))
        if log_files:
            print("\nLog Files:")
            for f in log_files:
                print(f"- {f.name} ({f.stat().st_size / 1024:.2f} KB)")
        else:
            print("No log files found!")

        print("\n" + "=" * 50)

    except Exception as e:
        print(f"Error verifying results: {e}")
        import traceback
        print(traceback.format_exc())


def main():
    """Main training execution with enhanced logging"""
    try:
        logger = get_logger(__name__)

        # Add console handler for immediate feedback
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        print("\n" + "=" * 50)
        print("Starting HIDS Model Training Pipeline")
        print("=" * 50 + "\n")

        # Initialize pipeline
        logger.info("Initializing training pipeline...")
        pipeline = TrainingPipeline(force_reprocess=True)

        # Clean up old artifacts
        pipeline.clean_training_artifacts()

        # Load and prepare data
        logger.info("Loading training data...")
        data = pipeline.load_training_data()
        if data is None:
            logger.error("Failed to load training data")
            return

        X_train, X_val = data
        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Validation data shape: {X_val.shape}")

        # Initialize model
        input_dim = X_train.shape[1]
        logger.info(f"Initializing model with input dimension: {input_dim}")
        pipeline.initialize_model(input_dim)

        # Print training parameters
        print("\nTraining Parameters:")
        print(f"Batch Size: {config.MODEL_CONFIG['training']['batch_size']}")
        print(f"Learning Rate: {config.MODEL_CONFIG['training']['learning_rate']}")
        print(f"Max Epochs: {config.MODEL_CONFIG['training']['num_epochs']}")
        print(f"Early Stopping Patience: {config.MODEL_CONFIG['training']['early_stopping_patience']}")
        print("=" * 50 + "\n")

        # Train model
        logger.info("Starting model training...")
        training_start = datetime.now()

        if pipeline.train(X_train, X_val):
            training_time = datetime.now() - training_start
            logger.info("Training completed successfully!")
            logger.info(f"Total training time: {training_time}")

            # Print training results
            print("\nTraining Results:")
            print(f"Best Validation Loss: {pipeline.training_metrics['best_val_loss']:.6f}")
            print(f"Best Epoch: {pipeline.training_metrics['best_epoch']}")
            print(f"Final Training Loss: {pipeline.training_metrics['train_loss'][-1]:.6f}")
            print(f"Final Validation Loss: {pipeline.training_metrics['val_loss'][-1]:.6f}")

            # Print model save locations
            print("\nModel Artifacts Saved:")
            print(f"Model Directory: {config.DATA_STORAGE['models']}/final/")
            print(f"Results Directory: {config.DATA_STORAGE['results']}/training/")
            print(f"Log Directory: {config.LOG_DIR}")

        else:
            logger.error("Training failed!")
            print("\nTraining failed! Check logs for details.")

        print("\n" + "=" * 50)
        print("Training Pipeline Completed")
        print("=" * 50 + "\n")

    except Exception as e:
        logger.error(f"Error in training pipeline: {e}")
        print(f"\nCritical Error: {str(e)}")
        print("Check logs for details.")
        raise


if __name__ == "__main__":
    main()
    verify_training_results()
