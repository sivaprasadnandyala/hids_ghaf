import os
# disable
import warnings
warnings.filterwarnings('ignore')
import time
from .model_utils import Autoencoder, detect_anomalies_with_fixed_threshold
import torch
import numpy as np
import joblib
import sys
import gc
import json
from datetime import datetime, timedelta
from multiprocessing import Process, Queue, Event, Manager
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# sklearn metrics imports
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    auc,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score
)

from collections import defaultdict
from datetime import datetime

import traceback
from .config import config
from .utils import (
    collect_syscalls,
    clean_up_files,
    verify_tetragon_running,
    start_tetragon_collection,
    stop_tetragon,
    get_process_pids,
    print_confusion_matrix_and_metrics,
    handle_exit,
    cleanup_files_on_exit
)
from .model_utils import load_autoencoder
from .logging_setup import get_logger
from .preprocessing import preprocess_data
from .monitoring_utils import SystemMonitor, AnomalyVisualizer


class ModelTester:
    def __init__(self, force_preprocess=True):
        """Initialize model tester with verification"""
        self.logger = get_logger(__name__)
        self.device = torch.device("cpu")
        self.force_preprocess = force_preprocess
        self.config = config  # Add reference to config

        # Add collection metrics initialization here
        self.collection_metrics = {
            'processes': {},
            'detection_times': [],
            'total_syscalls': 0,
            'total_anomalies': 0,
            'start_time': datetime.now().isoformat()
        }

        # Initialize directories
        self.test_directories = {
            'logs': os.path.join(self.config.DATA_STORAGE['testing_data'], 'logs'),
            'features': os.path.join(self.config.DATA_STORAGE['testing_data'], 'features'),
            'processed': os.path.join(self.config.DATA_STORAGE['testing_data'], 'processed'),
            'results': os.path.join(self.config.DATA_STORAGE['results'], 'testing')
        }

        # Initialize metrics
        self._initialize_metrics()

        # Initialize components
        self.model = None
        self.scaler = None
        self.threshold_trained = None
        self.threshold_unseen = None
        self.stop_event = Event()
        self.manager = Manager()

        # Initialize metrics and results storage
        self.initialize_metrics()

        # Initialize directories
        self._initialize_directories()

        # Verify required files
        if not self.verify_required_files():
            self.logger.error("Required files verification failed")
            raise RuntimeError("Missing required files")

        # Load model artifacts
        if not self.load_model_artifacts():
            self.logger.error("Failed to load model artifacts")
            raise RuntimeError("Failed to load model artifacts")

        # Adjust thresholds if needed
        self._adjust_thresholds()

        self.logger.info("Model tester initialized successfully")

        self.logger.info("Model tester initialized successfully")

    def _initialize_directories(self) -> None:
        """Initialize required directories with proper permissions"""
        try:
            # Ensure base directories exist
            directories = [
                os.path.join(self.config.DATA_STORAGE['results'], 'testing'),
                os.path.join(self.config.DATA_STORAGE['results'], 'testing', 'plots'),
                os.path.join(self.config.DATA_STORAGE['results'], 'testing', 'metrics'),
                self.config.DATA_STORAGE['testing_data'],
                self.test_directories['logs'],
                self.test_directories['features'],
                self.test_directories['processed'],
                self.config.GRAPHS_DIR
            ]

            for directory in directories:
                os.makedirs(directory, mode=0o755, exist_ok=True)
                os.system(f"sudo chown -R {os.getenv('USER')}:{os.getenv('USER')} {directory}")
                self.logger.info(f"Created/verified directory: {directory}")

            # Test write permissions
            for directory in directories:
                test_file = os.path.join(directory, 'test_write.tmp')
                try:
                    with open(test_file, 'w') as f:
                        f.write('test')
                    os.remove(test_file)
                except Exception as e:
                    self.logger.error(f"Write permission test failed for {directory}: {e}")
                    raise

        except Exception as e:
            self.logger.error(f"Error creating directories: {e}")
            raise

    def load_model_artifacts(self) -> bool:
        """Load trained model and artifacts"""
        try:
            # Load scaler
            scaler_path = os.path.join(self.config.DATA_STORAGE['models'], 'final', 'scaler.pkl')
            self.logger.info(f"Looking for scaler at: {scaler_path}")

            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                self.logger.info(f"Loaded scaler with {self.scaler.n_features_in_} features")
            else:
                self.logger.error(f"Scaler not found at {scaler_path}")
                return False

            # Find and load latest model file
            model_dir = os.path.join(self.config.DATA_STORAGE['models'], 'final')
            self.logger.info(f"Looking for model files in: {model_dir}")

            model_files = list(Path(model_dir).glob('best_model_*.pth'))
            if not model_files:
                self.logger.error(f"No model files found in {model_dir}")
                return False

            latest_model = max(model_files, key=os.path.getctime)
            self.logger.info(f"Found latest model file: {latest_model}")

            input_dim = self.scaler.n_features_in_
            self.logger.info(f"Creating Autoencoder with input dimension: {input_dim}")

            self.model = Autoencoder(input_dim)
            self.model.load_state_dict(torch.load(latest_model, map_location=self.device, weights_only=True))
            self.model.to(self.device)
            self.model.eval()
            self.logger.info(f"Successfully loaded model from {latest_model}")

            # Load thresholds
            threshold_path = os.path.join(self.config.DATA_STORAGE['models'], 'final', 'thresholds.npy')
            self.logger.info(f"Looking for thresholds at: {threshold_path}")

            if os.path.exists(threshold_path):
                thresholds = np.load(threshold_path)
                self.threshold_trained, self.threshold_unseen = thresholds
                self.logger.info(
                    f"Loaded thresholds: trained={self.threshold_trained:.6f}, unseen={self.threshold_unseen:.6f}")
            else:
                self.logger.error(f"Thresholds not found at {threshold_path}")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error loading model artifacts: {e}")
            self.logger.error(traceback.format_exc())
            return False

    def verify_required_files(self) -> bool:
        """Verify all required model artifacts exist"""
        try:
            required_directories = [
                self.config.DATA_STORAGE['models'],
                os.path.join(self.config.DATA_STORAGE['models'], 'final'),
                self.config.DATA_STORAGE['results'],
                self.config.DATA_STORAGE['testing_data']
            ]

            for directory in required_directories:
                if not os.path.exists(directory):
                    self.logger.error(f"Required directory not found: {directory}")
                    return False

            # Print training artifacts locations
            self.logger.info("\nLooking for model artifacts in:")
            self.logger.info(f"Models directory: {self.config.DATA_STORAGE['models']}")
            self.logger.info(f"Results directory: {self.config.DATA_STORAGE['results']}")
            self.logger.info(f"Testing data directory: {self.config.DATA_STORAGE['testing_data']}")

            # List files in models directory
            model_files = list(Path(self.config.DATA_STORAGE['models']).rglob('*'))
            for file in model_files:
                if file.is_file():
                    self.logger.info(f"Found: {file}")

            return True

        except Exception as e:
            self.logger.error(f"Error verifying required files: {e}")
            return False

    def verify_test_data_exists(self) -> bool:
        """Verify test data exists for all processes"""
        try:
            test_dir = os.path.join(config.DATA_STORAGE['testing_data'], 'logs')

            # Create directory if it doesn't exist
            os.makedirs(test_dir, exist_ok=True)

            # List all available files
            all_files = list(Path(test_dir).glob('*.log'))
            self.logger.info(f"Found {len(all_files)} log files in {test_dir}")
            print(f"\nFound {len(all_files)} log files in test directory:")
            for file in all_files:
                print(f"- {file.name}")

            # Check each process
            process_data_status = {}
            for process in config.TESTING_PROCESSES:
                base_name = process.split('/')[-1].lower()
                matching_files = [f for f in all_files if base_name in f.name.lower()]
                process_data_status[process] = len(matching_files)

            # Print status
            print("\nData Status for Test Processes:")
            print("-" * 40)
            missing_data = False
            for process, file_count in process_data_status.items():
                status = "✓" if file_count > 0 else "✗"
                print(f"{status} {process}: {file_count} files")
                if file_count == 0:
                    missing_data = True

            if missing_data:
                print("\nMissing test data for some processes.")
                print("Please run data collection:")
                print("1. sudo python3.8 data_gathering.py")
                print("2. Verify data in:", test_dir)
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error verifying test data: {e}")
            return False

    def sanitize_process_name(self, process_name: str) -> str:
        """Create a safe filename from process name"""
        try:
            self.logger.debug(f"Sanitizing process name: {process_name}")

            # Handle empty process names
            if not process_name:
                self.logger.warning("Empty process name provided")
                return "unknown_process"

            # Handle VirtualBox VM names specially
            if "VirtualBoxVM" in process_name:
                parts = process_name.split("--")
                vm_name = None
                vm_id = None

                for part in parts:
                    if part.startswith("comment"):
                        vm_name = part.split(" ", 1)[1].strip()
                    elif part.startswith("startvm"):
                        vm_id = part.split(" ", 1)[1].strip()

                if vm_name:
                    safe_name = f"VirtualBoxVM_{vm_name}"
                elif vm_id:
                    safe_name = f"VirtualBoxVM_{vm_id[:8]}"
                else:
                    safe_name = "VirtualBoxVM_unknown"

                self.logger.debug(f"Sanitized VirtualBox name: {safe_name}")
                return safe_name

            # Handle regular process names
            # First, get the base name from the path
            base_name = process_name.split('/')[-1]

            # Handle special cases
            special_cases = {
                'python3': 'python3',
                'python': 'python',
                'atom': 'atom',
                'chrome': 'chrome',
                'firefox': 'firefox',
                'teams': 'teams'
            }

            # Check for special cases first
            for case, replacement in special_cases.items():
                if case in base_name.lower():
                    self.logger.debug(f"Using special case name: {replacement}")
                    return replacement

            # Remove any arguments (split on first space)
            base_name = base_name.split()[0]

            # Replace any remaining special characters with underscore
            safe_name = ''.join(c if c.isalnum() else '_' for c in base_name)

            # Remove repeated underscores
            safe_name = '_'.join(filter(None, safe_name.split('_')))

            # Ensure the name starts with a letter (prepend 'proc_' if not)
            if safe_name and not safe_name[0].isalpha():
                safe_name = f'proc_{safe_name}'

            # Limit length while keeping readability
            if len(safe_name) > 50:
                safe_name = safe_name[:47] + '...'

            self.logger.debug(f"Original: {process_name} -> Sanitized: {safe_name}")
            return safe_name.lower()

        except Exception as e:
            self.logger.error(f"Error sanitizing process name '{process_name}': {e}")
            self.logger.error(traceback.format_exc())
            return "unknown_process"

    def process_test_data(self, process_name: str) -> Optional[np.ndarray]:
        """Process test data for a specific process"""
        try:
            test_dir = os.path.join(config.DATA_STORAGE['testing_data'], 'logs')
            base_name = process_name.split('/')[-1].lower()

            # Find matching log files
            log_files = list(Path(test_dir).glob(f'*{base_name}*.log'))

            if not log_files:
                self.logger.warning(f"No log files found for {process_name} in {test_dir}")
                return None

            self.logger.info(f"Processing {len(log_files)} log files for {process_name}")
            all_features = []
            total_syscalls = 0

            for log_file in sorted(log_files):
                try:
                    with open(log_file, 'r') as f:
                        syscalls = []
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue

                            parts = line.split(' ', 1)
                            if len(parts) != 2:
                                continue

                            binary, syscall_info = parts
                            # Don't filter by binary here - trust the file naming
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

                    if syscalls:
                        self.logger.info(f"Found {len(syscalls)} syscalls in {log_file}")
                        features = preprocess_data(
                            [syscalls],
                            interval_counter=total_syscalls,
                            mode='test',
                            scaler=self.scaler
                        )

                        if features is not None and len(features) > 0:
                            all_features.append(features)
                            total_syscalls += len(syscalls)
                            self.logger.info(f"Processed {len(features)} feature vectors")

                except Exception as e:
                    self.logger.error(f"Error processing {log_file}: {e}")
                    continue

            if not all_features:
                self.logger.error(f"No features extracted for {process_name}")
                return None

            X_test = np.vstack(all_features)
            self.logger.info(f"Final feature shape for {process_name}: {X_test.shape}")
            return X_test

        except Exception as e:
            self.logger.error(f"Error processing test data: {e}")
            self.logger.error(traceback.format_exc())
            return None


    # def detect_anomalies(self, X_test: np.ndarray, process_name: str) -> bool:
    #     """Detect anomalies with adjusted thresholds"""
    #     try:
    #         start_time = time.time()
    #
    #         # Initialize process metrics
    #         process_metrics = self.testing_metrics['process_metrics'].get(process_name, {
    #             'mse_scores': [],
    #             'detection_times': [],
    #             'anomaly_counts': 0,
    #             'true_labels': [],
    #             'predicted_labels': []
    #         })
    #
    #         # Determine process type and threshold
    #         is_trained = process_name in self.config.TRAINING_PROCESSES
    #         threshold = self.threshold_trained if is_trained else self.threshold_unseen
    #         expected_label = 0 if is_trained else 1
    #
    #         print(f"\nAnalyzing process: {process_name}")
    #         print(f"Process type: {'Trained' if is_trained else 'Untrained'}")
    #         print(f"Using threshold: {threshold:.6f}")
    #
    #         # Detect anomalies
    #         self.model.eval()
    #         with torch.no_grad():
    #             batch_size = config.BATCH_SIZE
    #             total_anomalies = 0
    #             total_samples = len(X_test)
    #             all_mse_scores = []
    #
    #             for i in range(0, total_samples, batch_size):
    #                 batch = X_test[i:i + batch_size]
    #                 batch_tensor = torch.FloatTensor(batch).to(self.device)
    #
    #                 # Get reconstructions
    #                 reconstructions = self.model(batch_tensor)
    #
    #                 # Calculate MSE
    #                 mse = torch.mean(torch.pow(batch_tensor - reconstructions, 2), dim=1)
    #                 mse = mse.cpu().numpy()
    #                 all_mse_scores.extend(mse)
    #
    #                 # Detect anomalies
    #                 anomalies = (mse > threshold).astype(int)
    #                 total_anomalies += np.sum(anomalies)
    #
    #                 # Update metrics
    #                 process_metrics['mse_scores'].extend(mse.tolist())
    #                 process_metrics['predicted_labels'].extend(anomalies.tolist())
    #                 process_metrics['true_labels'].extend([expected_label] * len(batch))
    #
    #         # Update process metrics
    #         process_metrics['detection_times'].append(time.time() - start_time)
    #         process_metrics['anomaly_counts'] = total_anomalies
    #         self.testing_metrics['process_metrics'][process_name] = process_metrics
    #
    #         # Print detection results
    #         mse_array = np.array(all_mse_scores)
    #         print(f"\nDetection Results:")
    #         print(f"Total samples: {total_samples}")
    #         print(f"Anomalies detected: {total_anomalies}")
    #         print(f"Anomaly rate: {(total_anomalies / total_samples) * 100:.2f}%")
    #         print(f"MSE statistics:")
    #         print(f"- Mean: {np.mean(mse_array):.6f}")
    #         print(f"- Std: {np.std(mse_array):.6f}")
    #         print(f"- Min: {np.min(mse_array):.6f}")
    #         print(f"- Max: {np.max(mse_array):.6f}")
    #
    #         return True
    #
    #     except Exception as e:
    #         self.logger.error(f"Error detecting anomalies: {e}")
    #         self.logger.error(traceback.format_exc())
    #         return False

    def detect_anomalies(self, X_test: np.ndarray, process_name: str) -> bool:
        """Detect anomalies with optimized thresholds for ADFA-LD"""
        try:
            start_time = time.time()

            # Initialize process metrics (same as before)
            if process_name not in self.testing_metrics['process_metrics']:
                self.testing_metrics['process_metrics'][process_name] = {
                    'mse_scores': [],
                    'detection_times': [],
                    'anomaly_counts': 0,
                    'true_labels': [],
                    'predicted_labels': [],
                    'samples_processed': 0
                }

            process_metrics = self.testing_metrics['process_metrics'][process_name]
            is_trained = process_name in self.config.TRAINING_PROCESSES
            base_threshold = self.threshold_trained if is_trained else self.threshold_unseen
            expected_label = 0 if is_trained else 1

            print(f"\nAnalyzing process: {process_name}")
            print(f"Process type: {'Trained' if is_trained else 'Untrained'}")
            print(f"Base threshold: {base_threshold:.6f}")

            # Detect anomalies in batches
            self.model.eval()
            with torch.no_grad():
                batch_size = config.BATCH_SIZE
                total_samples = len(X_test)
                all_mse_scores = []

                # Collect MSE scores
                for i in range(0, total_samples, batch_size):
                    batch = X_test[i:i + batch_size]
                    batch_tensor = torch.FloatTensor(batch).to(self.device)
                    reconstructions = self.model(batch_tensor)
                    mse = torch.mean(torch.pow(batch_tensor - reconstructions, 2), dim=1)
                    all_mse_scores.extend(mse.cpu().numpy().tolist())

                # Calculate statistics
                mse_array = np.array(all_mse_scores)
                mean_mse = np.mean(mse_array)
                std_mse = np.std(mse_array)
                median_mse = np.median(mse_array)
                q75, q25 = np.percentile(mse_array, [75, 25])
                iqr = q75 - q25

                # Normalize MSE scores
                normalized_mse = (mse_array - mean_mse) / std_mse

                # Adjusted thresholds for ADFA-LD
                z_score_threshold = 2.0  # Lowered from 2.5
                iqr_threshold = np.median(normalized_mse) + 1.5 * np.std(normalized_mse)  # Modified IQR calculation
                percentile_threshold = 90  # Lowered from 95

                # Multiple detection methods
                z_score_anomalies = (normalized_mse > z_score_threshold)
                iqr_anomalies = (normalized_mse > iqr_threshold)
                percentile_anomalies = (mse_array > np.percentile(mse_array, percentile_threshold))

                # Ensemble approach: Require at least 2 methods to agree
                anomalies = ((z_score_anomalies.astype(int) +
                              iqr_anomalies.astype(int) +
                              percentile_anomalies.astype(int)) >= 2).astype(int)

                total_anomalies = np.sum(anomalies)

                # Update metrics
                process_metrics['mse_scores'] = all_mse_scores
                process_metrics['predicted_labels'] = anomalies.tolist()
                process_metrics['true_labels'].extend([expected_label] * total_samples)
                process_metrics['anomaly_counts'] = total_anomalies
                process_metrics['samples_processed'] = total_samples
                process_metrics['detection_times'].append(time.time() - start_time)

                # Calculate performance metrics
                tp = np.sum((np.array(process_metrics['true_labels']) == 1) & (anomalies == 1))
                fp = np.sum((np.array(process_metrics['true_labels']) == 0) & (anomalies == 1))
                tn = np.sum((np.array(process_metrics['true_labels']) == 0) & (anomalies == 0))
                fn = np.sum((np.array(process_metrics['true_labels']) == 1) & (anomalies == 0))

                accuracy = (tp + tn) / total_samples if total_samples > 0 else 0
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

                # Print statistics
                print(f"\nMSE Statistics (Original Scale):")
                print(f"- Mean: {mean_mse:.6f}")
                print(f"- Median: {median_mse:.6f}")
                print(f"- Std: {std_mse:.6f}")
                print(f"- IQR: {iqr:.6f}")
                print(f"- Q25: {q25:.6f}")
                print(f"- Q75: {q75:.6f}")

                print(f"\nNormalized Statistics:")
                print(f"- Z-score threshold: {z_score_threshold:.2f}")
                print(f"- IQR threshold: {iqr_threshold:.2f}")
                print(f"- Percentile threshold: {percentile_threshold:.2f}")
                print(f"- Mean Z-score: {np.mean(normalized_mse):.2f}")
                print(f"- Max Z-score: {np.max(normalized_mse):.2f}")

                print(f"\nDetection Results:")
                print(f"Total samples: {total_samples}")
                print(f"Anomalies detected: {total_anomalies}")
                print(f"Anomaly rate: {(total_anomalies / total_samples) * 100:.2f}%")
                print(f"\nBreakdown by method:")
                print(f"- Z-score anomalies: {np.sum(z_score_anomalies)}")
                print(f"- IQR anomalies: {np.sum(iqr_anomalies)}")
                print(f"- Percentile anomalies: {np.sum(percentile_anomalies)}")
                print(f"- Ensemble anomalies: {total_anomalies}")

                print("\nMetrics:")
                print(f"Accuracy: {accuracy:.4f}")
                print(f"Precision: {precision:.4f}")
                print(f"Recall: {recall:.4f}")
                print(f"Specificity: {specificity:.4f}")
                print(f"F1 Score: {f1:.4f}")

                # Save results
                results = {
                    'process_name': process_name,
                    'statistics': {
                        'original': {
                            'mean': float(mean_mse),
                            'median': float(median_mse),
                            'std': float(std_mse),
                            'iqr': float(iqr),
                            'q25': float(q25),
                            'q75': float(q75)
                        },
                        'normalized': {
                            'z_score_threshold': float(z_score_threshold),
                            'iqr_threshold': float(iqr_threshold),
                            'percentile_threshold': float(percentile_threshold),
                            'mean_z_score': float(np.mean(normalized_mse)),
                            'max_z_score': float(np.max(normalized_mse))
                        }
                    },
                    'detection_results': {
                        'total_samples': int(total_samples),
                        'anomalies': {
                            'total': int(total_anomalies),
                            'z_score': int(np.sum(z_score_anomalies)),
                            'iqr': int(np.sum(iqr_anomalies)),
                            'percentile': int(np.sum(percentile_anomalies))
                        },
                        'anomaly_rate': float((total_anomalies / total_samples) * 100)
                    },
                    'metrics': {
                        'accuracy': float(accuracy),
                        'precision': float(precision),
                        'recall': float(recall),
                        'specificity': float(specificity),
                        'f1_score': float(f1)
                    }
                }

                results_file = os.path.join(
                    self.config.DATA_STORAGE['results'],
                    'testing',
                    f'detection_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
                )

                os.makedirs(os.path.dirname(results_file), exist_ok=True)
                with open(results_file, 'w') as f:
                    json.dump(results, f, indent=4)

                return True

        except Exception as e:
            self.logger.error(f"Error detecting anomalies: {e}")
            self.logger.error(traceback.format_exc())
            return False

    def _adjust_thresholds(self) -> None:
        """Adjust detection thresholds based on the new model's reconstruction errors"""
        try:
            self.logger.info("Calculating detection thresholds...")

            # Set initial thresholds based on training results
            # Since validation loss is around 0.14, start with thresholds near this range
            self.threshold_trained = 0.3  # Slightly above training loss
            self.threshold_unseen = 0.5  # Higher threshold for unseen data

            # Load test data to verify thresholds
            X_test = self.process_test_data("/usr/bin/python3")
            if X_test is not None:
                with torch.no_grad():
                    self.model.eval()
                    X_tensor = torch.FloatTensor(X_test).to(self.device)
                    reconstructions = self.model(X_tensor)
                    mse = torch.mean(torch.pow(X_tensor - reconstructions, 2), dim=1)
                    mse = mse.cpu().numpy()

                    # Calculate distribution statistics
                    percentiles = np.percentile(mse, [25, 50, 75, 90, 95])
                    mse_mean = np.mean(mse)
                    mse_std = np.std(mse)

                    # Adjust thresholds based on distribution
                    self.threshold_trained = percentiles[1]  # Use median
                    self.threshold_unseen = percentiles[0]  # Use 25th percentile to catch more anomalies

                    self.logger.info(f"\nMSE Distribution for test data:")
                    self.logger.info(f"Mean: {mse_mean:.6f}")
                    self.logger.info(f"Std: {mse_std:.6f}")
                    self.logger.info(f"25th percentile: {percentiles[0]:.6f}")
                    self.logger.info(f"Median: {percentiles[1]:.6f}")
                    self.logger.info(f"75th percentile: {percentiles[2]:.6f}")
                    self.logger.info(f"90th percentile: {percentiles[3]:.6f}")
                    self.logger.info(f"95th percentile: {percentiles[4]:.6f}")

                    self.logger.info(f"\nSelected Thresholds:")
                    self.logger.info(f"Trained threshold: {self.threshold_trained:.6f}")
                    self.logger.info(f"Unseen threshold: {self.threshold_unseen:.6f}")

                    # Print analysis of expected anomalies
                    anomalies_expected = np.sum(mse > self.threshold_unseen)
                    anomaly_rate = (anomalies_expected / len(mse)) * 100
                    self.logger.info(f"\nExpected Results:")
                    self.logger.info(f"Total samples: {len(mse)}")
                    self.logger.info(f"Expected anomalies: {anomalies_expected}")
                    self.logger.info(f"Expected anomaly rate: {anomaly_rate:.2f}%")

        except Exception as e:
            self.logger.error(f"Error adjusting thresholds: {e}")
            self.logger.error(traceback.format_exc())



    def plot_additional_metrics(self) -> None:
        """Plot additional metrics for analysis"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plots_dir = os.path.join(self.config.DATA_STORAGE['results'], 'testing', 'plots')
            os.makedirs(plots_dir, exist_ok=True)

            # Plot MSE distribution for each process
            plt.figure(figsize=(12, 6))
            for process_name, metrics in self.testing_metrics['process_metrics'].items():
                mse_scores = np.array(metrics['mse_scores'])
                plt.hist(mse_scores, bins=50, alpha=0.5, label=process_name)

            plt.axvline(self.threshold_trained, color='r', linestyle='--', label='Trained Threshold')
            plt.axvline(self.threshold_unseen, color='g', linestyle='--', label='Unseen Threshold')
            plt.xlabel('MSE Score')
            plt.ylabel('Count')
            plt.title('MSE Distribution by Process')
            plt.legend()
            plt.grid(True)

            # Save plot
            plot_path = os.path.join(plots_dir, f'mse_distribution_{timestamp}.png')
            plt.savefig(plot_path)
            plt.close()

            self.logger.info(f"Additional metrics plots saved to: {plot_path}")

        except Exception as e:
            self.logger.error(f"Error plotting additional metrics: {e}")

    def calculate_metrics(self) -> None:
        """Calculate and store performance metrics with proper handling of process metrics"""
        try:
            # Initialize overall metrics
            overall_metrics = {
                'total_samples': 0,
                'total_anomalies': 0,
                'all_true_labels': [],
                'all_pred_labels': [],
                'all_mse_scores': [],
                'all_detection_times': []
            }

            # Collect metrics from all processes
            for process_name, process_metrics in self.testing_metrics['process_metrics'].items():
                # Update totals
                overall_metrics['total_samples'] += process_metrics['samples_processed']
                overall_metrics['total_anomalies'] += process_metrics['anomaly_counts']

                # Collect labels and scores
                overall_metrics['all_true_labels'].extend(process_metrics['true_labels'])
                overall_metrics['all_pred_labels'].extend(process_metrics['predicted_labels'])
                overall_metrics['all_mse_scores'].extend(process_metrics['mse_scores'])
                overall_metrics['all_detection_times'].extend(process_metrics['detection_times'])

            # Calculate overall metrics if we have data
            if overall_metrics['total_samples'] > 0:
                true_labels = np.array(overall_metrics['all_true_labels'])
                pred_labels = np.array(overall_metrics['all_pred_labels'])

                # Calculate confusion matrix elements
                tp = np.sum((true_labels == 1) & (pred_labels == 1))
                fp = np.sum((true_labels == 0) & (pred_labels == 1))
                tn = np.sum((true_labels == 0) & (pred_labels == 0))
                fn = np.sum((true_labels == 1) & (pred_labels == 0))

                # Calculate metrics
                self.testing_metrics['overall_metrics'].update({
                    'total_samples': overall_metrics['total_samples'],
                    'total_anomalies': overall_metrics['total_anomalies'],
                    'accuracy': (tp + tn) / overall_metrics['total_samples'] if overall_metrics[
                                                                                    'total_samples'] > 0 else 0,
                    'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
                    'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
                    'f1_score': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
                    'avg_mse': np.mean(overall_metrics['all_mse_scores']) if overall_metrics['all_mse_scores'] else 0,
                    'avg_detection_time': np.mean(overall_metrics['all_detection_times']) if overall_metrics[
                        'all_detection_times'] else 0
                })

            # Print detailed metrics
            print("\nOverall Performance Metrics:")
            print("=" * 50)
            print(f"Total Samples Processed: {self.testing_metrics['overall_metrics']['total_samples']}")
            print(f"Total Anomalies Detected: {self.testing_metrics['overall_metrics']['total_anomalies']}")
            if self.testing_metrics['overall_metrics']['total_samples'] > 0:
                print(f"Average MSE: {self.testing_metrics['overall_metrics']['avg_mse']:.6f}")
                print(f"Average Detection Time: {self.testing_metrics['overall_metrics']['avg_detection_time']:.4f}s")
                print(f"Accuracy: {self.testing_metrics['overall_metrics']['accuracy']:.4f}")
                print(f"Precision: {self.testing_metrics['overall_metrics']['precision']:.4f}")
                print(f"Recall: {self.testing_metrics['overall_metrics']['recall']:.4f}")
                print(f"F1 Score: {self.testing_metrics['overall_metrics']['f1_score']:.4f}")

            # Print process-specific metrics
            for process_name, metrics in self.testing_metrics['process_metrics'].items():
                if metrics['samples_processed'] > 0:
                    print(f"\nProcess: {process_name}")
                    print(f"{'Trained' if process_name in self.config.TRAINING_PROCESSES else 'Untrained'} Process")
                    print(f"Total Samples: {metrics['samples_processed']}")
                    print(f"Anomalies Detected: {metrics['anomaly_counts']}")
                    print(f"Average MSE: {np.mean(metrics['mse_scores']):.6f}")
                    print(f"Average Detection Time: {np.mean(metrics['detection_times']):.4f}s")

        except Exception as e:
            self.logger.error(f"Error calculating metrics: {e}")
            self.logger.error(traceback.format_exc())

    def plot_results(self) -> None:
        """Generate comprehensive result plots"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plots_dir = os.path.join(self.config.DATA_STORAGE['results'], 'testing', 'plots')
            os.makedirs(plots_dir, exist_ok=True)

            # Create overall results plot
            self._plot_overall_results(plots_dir, timestamp)

            # Create process-specific plots
            self._plot_process_results(plots_dir, timestamp)

            # Plot ROC curves
            self._plot_roc_curves(plots_dir, timestamp)

            # Plot anomaly score distributions
            self._plot_anomaly_distributions(plots_dir, timestamp)

            self.logger.info(f"Saved result plots to {plots_dir}")

        except Exception as e:
            self.logger.error(f"Error plotting results: {e}")
            self.logger.error(traceback.format_exc())

    def initialize_metrics(self):
        """Initialize testing metrics structure"""
        self.testing_metrics = {
            'process_metrics': defaultdict(lambda: {
                'mse_scores': [],
                'detection_times': [],
                'anomaly_counts': 0,
                'true_labels': [],
                'predicted_labels': []
            }),
            'overall_metrics': {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'roc_auc': 0.0,
                'detection_time_mean': 0.0,
                'detection_time_std': 0.0
            }
        }

    def _plot_roc_curves(self, plots_dir: str, timestamp: str) -> None:
        """Plot ROC curves for all processes"""
        try:
            plt.figure(figsize=(12, 8))

            # Plot ROC curve for each process
            for process_name, metrics in self.testing_metrics['process_metrics'].items():
                if len(metrics['true_labels']) > 0:
                    fpr, tpr, _ = roc_curve(
                        metrics['true_labels'],
                        metrics['mse_scores']
                    )
                    roc_auc = auc(fpr, tpr)
                    plt.plot(
                        fpr, tpr,
                        label=f'{process_name} (AUC = {roc_auc:.2f})'
                    )

            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curves by Process')
            plt.legend()
            plt.grid(True)

            plt.savefig(os.path.join(plots_dir, f'roc_curves_{timestamp}.png'))
            plt.close()

        except Exception as e:
            self.logger.error(f"Error plotting ROC curves: {e}")
            self.logger.error(traceback.format_exc())

    def _plot_anomaly_distributions(self, plots_dir: str, timestamp: str) -> None:
        """Plot anomaly score distributions"""
        try:
            plt.figure(figsize=(12, 8))

            for process_name, metrics in self.testing_metrics['process_metrics'].items():
                if metrics['mse_scores']:
                    sns.kdeplot(
                        data=metrics['mse_scores'],
                        label=process_name
                    )

            plt.axvline(
                self.threshold_trained,
                color='r',
                linestyle='--',
                label='Trained Threshold'
            )
            plt.axvline(
                self.threshold_unseen,
                color='g',
                linestyle='--',
                label='Unseen Threshold'
            )

            plt.xlabel('Anomaly Score')
            plt.ylabel('Density')
            plt.title('Anomaly Score Distributions by Process')
            plt.legend()
            plt.grid(True)

            plt.savefig(os.path.join(plots_dir, f'anomaly_distributions_{timestamp}.png'))
            plt.close()

        except Exception as e:
            self.logger.error(f"Error plotting anomaly distributions: {e}")
            self.logger.error(traceback.format_exc())

    def generate_test_report(self) -> None:
        """Generate HTML test report with proper error handling"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_dir = os.path.join(self.config.DATA_STORAGE['results'], 'testing')
            os.makedirs(report_dir, exist_ok=True)

            # Generate HTML content
            html_content = """
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    body {{ 
                        font-family: Arial, sans-serif;
                        margin: 20px;
                        background-color: #f5f5f5;
                    }}
                    .container {{ 
                        max-width: 1200px;
                        margin: 0 auto;
                        background-color: white;
                        padding: 20px;
                        border-radius: 8px;
                        box-shadow: 0 0 10px rgba(0,0,0,0.1);
                    }}
                    h2, h3 {{ 
                        color: #2c3e50;
                        border-bottom: 2px solid #3498db;
                        padding-bottom: 10px;
                    }}
                    table {{ 
                        width: 100%;
                        border-collapse: collapse;
                        margin: 20px 0;
                    }}
                    th, td {{ 
                        padding: 12px;
                        text-align: left;
                        border: 1px solid #ddd;
                    }}
                    th {{ 
                        background-color: #3498db;
                        color: white;
                    }}
                    .metric {{ 
                        font-weight: bold;
                        color: #2980b9;
                    }}
                    .warning {{
                        color: #e74c3c;
                    }}
                    .success {{
                        color: #27ae60;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h2>Test Results Report</h2>
                    <p>Generated: {timestamp}</p>

                    <h3>Overall Metrics</h3>
                    <table>
                        <tr><th>Metric</th><th>Value</th></tr>
            """.format(timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

            # Add overall metrics
            metrics = self.testing_metrics['overall_metrics']
            for metric_name, value in metrics.items():
                if value is not None:
                    if isinstance(value, float):
                        formatted_value = f"{value:.4f}"
                    else:
                        formatted_value = str(value)
                    html_content += f"""
                        <tr>
                            <td>{metric_name.replace('_', ' ').title()}</td>
                            <td class="metric">{formatted_value}</td>
                        </tr>"""

            html_content += """
                    </table>
                    <h3>Process-Specific Results</h3>
            """

            # Add process-specific sections
            for process_name, metrics in self.testing_metrics['process_metrics'].items():
                if metrics.get('mse_scores'):
                    html_content += f"""
                    <h4>{process_name}</h4>
                    <table>
                        <tr>
                            <td>Total Samples</td>
                            <td class="metric">{len(metrics['mse_scores'])}</td>
                        </tr>
                        <tr>
                            <td>Anomalies Detected</td>
                            <td class="metric">{metrics.get('anomaly_counts', 0)}</td>
                        </tr>
                        <tr>
                            <td>Average MSE</td>
                            <td class="metric">{np.mean(metrics['mse_scores']):.6f}</td>
                        </tr>
                        <tr>
                            <td>Average Detection Time</td>
                            <td class="metric">{np.mean(metrics['detection_times']):.4f}s</td>
                        </tr>
                    """

                    # Add process-specific metrics if available
                    for metric_name in ['accuracy', 'precision', 'recall', 'f1_score']:
                        if metric_name in metrics:
                            html_content += f"""
                        <tr>
                            <td>{metric_name.replace('_', ' ').title()}</td>
                            <td class="metric">{metrics[metric_name]:.4f}</td>
                        </tr>"""

                    html_content += """
                    </table>
                    """

            # Close HTML
            html_content += """
                </div>
            </body>
            </html>
            """

            # Save report
            report_path = os.path.join(report_dir, f'test_report_{timestamp}.html')
            with open(report_path, 'w') as f:
                f.write(html_content)

            self.logger.info(f"Test report generated: {report_path}")

        except Exception as e:
            self.logger.error(f"Error generating test report: {e}")
            self.logger.error(traceback.format_exc())

        finally:
            # Cleanup any temporary files if needed
            pass

    def _plot_overall_results(self, plots_dir: str, timestamp: str) -> None:
        """Plot overall test results"""
        try:
            if not self.collection_metrics['processes']:
                self.logger.warning("No data available for plotting")
                return

            # Get overall metrics
            all_true = []
            all_pred = []
            all_scores = []
            for metrics in self.collection_metrics['processes'].values():
                all_true.extend(metrics['true_labels'])
                all_pred.extend(metrics['predicted_labels'])
                all_scores.extend(metrics['mse_scores'])

            if not all_true or not all_pred:
                self.logger.warning("No prediction data available for plotting")
                return

            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))

            # Plot confusion matrix
            cm = confusion_matrix(all_true, all_pred)
            if cm.size > 0:  # Only plot if we have data
                sns.heatmap(cm, annot=True, fmt='d', ax=ax1, cmap='Blues')
                ax1.set_title('Overall Confusion Matrix')
                ax1.set_xlabel('Predicted')
                ax1.set_ylabel('True')

            # Plot ROC curve
            if len(np.unique(all_true)) > 1:  # Only plot ROC if we have both classes
                fpr, tpr, _ = roc_curve(all_true, all_scores)
                roc_auc = auc(fpr, tpr)
                ax2.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
                ax2.plot([0, 1], [0, 1], 'k--')
                ax2.set_xlabel('False Positive Rate')
                ax2.set_ylabel('True Positive Rate')
                ax2.set_title('Overall ROC Curve')
                ax2.legend()
                ax2.grid(True)

            # Plot detection times
            if self.collection_metrics.get('detection_times'):
                ax3.hist(self.collection_metrics['detection_times'], bins=30)
                ax3.set_title('Detection Time Distribution')
                ax3.set_xlabel('Time (seconds)')

            # Plot MSE distribution
            if all_scores:
                ax4.hist(all_scores, bins=50)
                ax4.axvline(self.threshold_trained, color='r', linestyle='--', label='Trained Threshold')
                ax4.axvline(self.threshold_unseen, color='g', linestyle='--', label='Unseen Threshold')
                ax4.set_title('MSE Score Distribution')
                ax4.legend()

            plt.tight_layout()
            plot_path = os.path.join(plots_dir, f'overall_results_{timestamp}.png')
            plt.savefig(plot_path)
            plt.close()

        except Exception as e:
            self.logger.error(f"Error plotting overall results: {e}")
            self.logger.error(traceback.format_exc())

    def _plot_process_results(self, plots_dir: str, timestamp: str) -> None:
        """Plot process-specific results"""
        try:
            for process_name, metrics in self.testing_metrics['process_metrics'].items():
                if not metrics['mse_scores']:
                    continue

                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

                # Process name for display
                display_name = process_name.split('/')[-1]

                # Plot MSE distribution
                sns.histplot(metrics['mse_scores'], ax=ax1, bins=50)
                ax1.axvline(self.threshold_trained, color='r', linestyle='--', label='Trained')
                ax1.axvline(self.threshold_unseen, color='g', linestyle='--', label='Unseen')
                ax1.set_title(f'MSE Distribution - {display_name}')
                ax1.legend()

                # Plot confusion matrix
                cm = confusion_matrix(metrics['true_labels'], metrics['predicted_labels'])
                sns.heatmap(cm, annot=True, fmt='d', ax=ax2, cmap='Blues')
                ax2.set_title(f'Confusion Matrix - {display_name}')

                # Plot detection times
                sns.histplot(metrics['detection_times'], ax=ax3, bins=30)
                ax3.set_title(f'Detection Times - {display_name}')
                ax3.set_xlabel('Time (seconds)')

                # Plot anomaly scores over time
                ax4.plot(metrics['mse_scores'], label='MSE Score')
                ax4.axhline(y=self.threshold_trained, color='r', linestyle='--', label='Trained Threshold')
                ax4.axhline(y=self.threshold_unseen, color='g', linestyle='--', label='Unseen Threshold')
                ax4.set_title(f'Anomaly Scores Over Time - {display_name}')
                ax4.legend()

                plt.tight_layout()
                safe_name = process_name.replace('/', '_').replace(' ', '_')
                plt.savefig(os.path.join(plots_dir, f'process_results_{safe_name}_{timestamp}.png'))
                plt.close()

        except Exception as e:
            self.logger.error(f"Error plotting process results: {e}")
            self.logger.error(traceback.format_exc())


    def run_testing(self) -> bool:
        """Run complete testing pipeline with comprehensive output"""
        try:
            print("\n" + "=" * 50)
            print("Starting Testing Pipeline")
            print("=" * 50 + "\n")

            # First verify test data exists
            if not self.verify_test_data_exists():
                return False

            self.logger.info("Starting testing pipeline...")
            start_time = time.time()

            # Initialize metrics
            self.testing_metrics = {
                'process_metrics': defaultdict(lambda: {
                    'total_syscalls': 0,
                    'processed_syscalls': 0,
                    'features_extracted': 0,
                    'mse_scores': [],
                    'detection_times': [],
                    'anomaly_counts': 0,
                    'true_labels': [],
                    'predicted_labels': []
                }),
                'overall_metrics': {
                    'accuracy': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1_score': 0.0,
                    'roc_auc': None
                }
            }

            # Process each test process
            for process_name in config.TESTING_PROCESSES:
                print(f"\nProcessing: {process_name}")
                print("-" * 50)

                # Process test data
                X_test = self.process_test_data(process_name)
                if X_test is None:
                    print(f"Skipping {process_name} - no valid test data")
                    continue

                # Detect anomalies
                if not self.detect_anomalies(X_test, process_name):
                    print(f"Failed to detect anomalies for {process_name}")
                    continue

                # Print initial process results
                process_metrics = self.testing_metrics['process_metrics'][process_name]
                print(f"\nResults for {process_name}:")
                print(f"Total samples: {len(process_metrics['mse_scores'])}")
                print(f"Anomalies detected: {process_metrics['anomaly_counts']}")
                if len(process_metrics['mse_scores']) > 0:
                    detection_rate = (process_metrics['anomaly_counts'] / len(process_metrics['mse_scores']) * 100)
                    print(f"Detection rate: {detection_rate:.2f}%")
                    print(f"Average MSE: {np.mean(process_metrics['mse_scores']):.6f}")
                    print(f"Average detection time: {np.mean(process_metrics['detection_times']):.4f}s")

            # Calculate metrics and generate visualizations
            self.calculate_metrics()
            self.plot_results()
            self.plot_additional_metrics()

            # Generate and save reports
            report_dir = os.path.join(config.DATA_STORAGE['results'], 'testing')
            self.generate_test_report()

            print("\nResults have been saved to:")
            print(f"- Reports: {report_dir}")
            print(f"- Plots: {os.path.join(report_dir, 'plots')}")
            print(f"- Metrics: {os.path.join(report_dir, 'metrics')}")

            # Calculate final totals
            total_anomalies = sum(m['anomaly_counts'] for m in self.testing_metrics['process_metrics'].values())
            total_samples = sum(len(m['mse_scores']) for m in self.testing_metrics['process_metrics'].values())

            # Print FINAL summary
            print("\n" + "=" * 50)
            print("Testing Pipeline Completed Successfully")
            print("=" * 50)

            print("\nOverall Testing Summary:")
            print("=" * 50)
            print(f"Total processes tested: {len(self.testing_metrics['process_metrics'])}")
            print(f"Total samples processed: {total_samples}")
            print(f"Total anomalies detected: {total_anomalies}")
            if total_samples > 0:
                print(f"Overall detection rate: {(total_anomalies / total_samples) * 100:.2f}%")
            print(f"Testing time: {time.time() - start_time:.2f} seconds")

            print("\nProcess-Specific Results:")
            print("-" * 50)
            for process_name, metrics in self.testing_metrics['process_metrics'].items():
                print(f"\nProcess: {process_name}")
                print(f"{'Trained' if process_name in config.TRAINING_PROCESSES else 'Untrained'} Process")
                print(f"Total Samples: {len(metrics['mse_scores'])}")
                print(f"Anomalies Detected: {metrics['anomaly_counts']}")
                if len(metrics['mse_scores']) > 0:
                    print(f"Average MSE: {np.mean(metrics['mse_scores']):.6f}")
                if metrics['detection_times']:
                    print(f"Average Detection Time: {np.mean(metrics['detection_times']):.4f}s")

            print("\nModel Performance Metrics:")
            print("-" * 50)
            print(f"Accuracy: {self.testing_metrics['overall_metrics']['accuracy']:.4f}")
            print(f"Precision: {self.testing_metrics['overall_metrics']['precision']:.4f}")
            print(f"Recall: {self.testing_metrics['overall_metrics']['recall']:.4f}")
            print(f"F1 Score: {self.testing_metrics['overall_metrics']['f1_score']:.4f}")
            if self.testing_metrics['overall_metrics']['roc_auc'] is not None:
                print(f"ROC AUC: {self.testing_metrics['overall_metrics']['roc_auc']:.4f}")

            print("\nThreshold Information:")
            print("-" * 50)
            print(f"Trained threshold: {self.threshold_trained:.6f}")
            print(f"Unseen threshold: {self.threshold_unseen:.6f}")

            return True

        except Exception as e:
            self.logger.error(f"Error in testing pipeline: {e}")
            self.logger.error(traceback.format_exc())
            print("\nTesting Pipeline Failed")
            print("Please check:")
            print("1. Data collection has been run")
            print("2. Model has been trained")
            print("3. Log files exist in the correct location")
            print(f"4. Check logs at: {config.LOG_DIR}")
            return False

    def _initialize_metrics(self):
        """Initialize testing metrics structure"""
        self.testing_metrics = {
            'process_metrics': defaultdict(lambda: {
                'total_syscalls': 0,
                'processed_syscalls': 0,
                'features_extracted': 0,
                'mse_scores': [],
                'detection_times': [],
                'anomaly_counts': 0,
                'true_labels': [],
                'predicted_labels': [],
                'start_time': None,
                'end_time': None,
                'mode': None,
                'errors': []
            }),
            'overall_metrics': {
                'total_samples': 0,
                'total_anomalies': 0,
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'roc_auc': 0.0,
                'detection_time_mean': 0.0,
                'detection_time_std': 0.0
            },
            'system_metrics': {
                'cpu_usage': [],
                'memory_usage': [],
                'processing_times': [],
                'start_time': datetime.now().isoformat(),
                'end_time': None
            }
        }

        # Initialize per-class metrics
        self.testing_metrics['class_metrics'] = {
            'normal': {
                'true_positives': 0,
                'false_positives': 0,
                'true_negatives': 0,
                'false_negatives': 0
            },
            'anomaly': {
                'true_positives': 0,
                'false_positives': 0,
                'true_negatives': 0,
                'false_negatives': 0
            }
        }

        # Initialize thresholds if not already set
        if not hasattr(self, 'threshold_trained'):
            self.threshold_trained = 0.3  # Default value
        if not hasattr(self, 'threshold_unseen'):
            self.threshold_unseen = 0.5  # Default value

        self.logger.info("Testing metrics initialized")

    def show_latest_results(self):
        """Display the latest test results"""
        try:
            results_dir = self.config.DATA_STORAGE['results']
            test_dir = os.path.join(results_dir, 'testing')

            print("\nTest Results Summary:")
            print("=" * 50)

            # Display process-specific results
            print("\nProcess-Specific Results:")
            print("-" * 50)
            for process_name, process_metrics in self.testing_metrics['process_metrics'].items():
                print(f"\nProcess: {process_name}")
                print(f"{'Trained' if process_name in self.config.TRAINING_PROCESSES else 'Untrained'} Process")
                print(f"Total Samples: {len(process_metrics['mse_scores'])}")
                print(f"Anomalies Detected: {process_metrics['anomaly_counts']}")

                if process_metrics['mse_scores']:
                    print(f"Average MSE: {np.mean(process_metrics['mse_scores']):.6f}")
                if process_metrics['detection_times']:
                    print(f"Average Detection Time: {np.mean(process_metrics['detection_times']):.4f}s")

            # Display overall metrics
            if 'overall_metrics' in self.testing_metrics:
                print("\nOverall Metrics:")
                print("-" * 50)
                metrics = self.testing_metrics['overall_metrics']
                for metric_name, value in metrics.items():
                    if value is not None:
                        print(f"{metric_name.replace('_', ' ').title()}: {value:.4f}")

            # Show file locations
            print("\nDetailed Results Location:")
            print("-" * 50)
            print(f"Reports Directory: {test_dir}")
            print(f"Plots Directory: {os.path.join(test_dir, 'plots')}")

            # Find and display latest files
            report_files = list(Path(test_dir).glob('test_report_*.html'))
            plot_files = list(Path(os.path.join(test_dir, 'plots')).glob('*.png'))

            if report_files:
                latest_report = max(report_files, key=os.path.getctime)
                print(f"\nLatest Test Report: {latest_report}")

            if plot_files:
                print("\nGenerated Plots:")
                for plot in sorted(plot_files):
                    print(f"- {plot.name}")

        except Exception as e:
            self.logger.error(f"Error showing results: {e}")
            self.logger.error(traceback.format_exc())


def main():
    """Main testing execution with comprehensive error handling and reporting"""
    try:
        # Setup logging
        logger = get_logger(__name__)
        logger.info("Starting testing pipeline...")

        # Initialize tester
        tester = ModelTester(force_preprocess=True)

        try:
            # Run testing
            if tester.run_testing():
                print("\n" + "=" * 50)
                print("Testing Pipeline Completed Successfully")
                print("=" * 50)

                # Show detailed results
                tester.show_latest_results()
                logger.info("\nTesting completed successfully!")

                # Show results locations
                logger.info("Results have been saved to:")
                logger.info(f"- Plots: {os.path.join(config.DATA_STORAGE['results'], 'testing', 'plots')}")
                logger.info(f"- Report: {os.path.join(config.DATA_STORAGE['results'], 'testing')}")
            else:
                logger.error("Testing failed!")
                print("\nTesting Pipeline Failed")
                print("Please check:")
                print("1. Data collection has been run")
                print("2. Model has been trained")
                print("3. Log files exist in the correct location")
                print(f"4. Check logs at: {config.LOG_DIR}")

        except KeyboardInterrupt:
            logger.info("Testing interrupted by user")
            print("\nTesting interrupted by user")

        finally:
            # Cleanup
            cleanup_files_on_exit()
            gc.collect()

    except Exception as e:
        logger.error(f"Critical error in testing: {e}")
        logger.error(traceback.format_exc())
        print("\nCritical Error Occurred!")
        print("Please check the logs for details")
        sys.exit(1)


if __name__ == "__main__":
    main()
