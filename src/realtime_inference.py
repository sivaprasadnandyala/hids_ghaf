# Part 1: Core Initialization and Metrics Setup
# Includes: Class initialization, metrics structures, and basic setup

import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import traceback
import time
import gc
import json
import numpy as np
import torch
import joblib
import signal
# Add these imports at the top of realtime_inference.py
from torch.utils.data import DataLoader, TensorDataset
import torch
from torch import nn
import psutil
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, Set, TypedDict
from multiprocessing import Process, Queue, Event, Manager
from pathlib import Path
from .config import config
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, roc_curve, auc
)
from .logging_setup import get_logger

from .utils import (
    collect_syscalls,
    verify_tetragon_running,
    start_tetragon_collection,
    stop_tetragon,
    get_process_pids,
    handle_exit,
    cleanup_files_on_exit,
    clean_up_files
)
from .model_utils import load_autoencoder, detect_anomalies_with_fixed_threshold
from .logging_setup import get_logger
from .preprocessing import preprocess_data
from .monitoring_utils import SystemMonitor, AnomalyVisualizer
from .syscall_utils import convert_json_to_text, read_syscalls_from_log



# ANSI color codes for console output
RED = '\033[91m'
GREEN = '\033[92m'
BLUE = '\033[94m'
YELLOW = '\033[93m'
RESET = '\033[0m'

class ProcessMetrics(TypedDict):
    """Type definition for process-level metrics"""
    samples_processed: int
    anomalies_detected: int
    mse_values: List[float]
    detection_times: List[float]
    anomaly_rates: List[float]
    total_syscalls: int
    start_time: datetime
    last_update: datetime
    true_labels: List[int]
    predicted_labels: List[int]
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    interval_metrics: Dict[str, List[float]]


class DetectionMetrics(TypedDict):
    """Type definition for detection metrics"""
    window_size: int
    mse_history: Dict[str, List[float]]
    anomaly_history: Dict[str, List[bool]]
    detection_times: Dict[str, List[float]]
    alerts: Dict[str, List[Dict[str, Any]]]
    process_metrics: Dict[str, ProcessMetrics]


class RealTimeDetector:
    def __init__(self):
        """Initialize real-time detector with enhanced metrics tracking"""
        self.config = config
        self.logger = get_logger(__name__)
        self.device = torch.device("cpu")
        self.start_time = datetime.now()

        # Initialize components
        self.model = None
        self.scaler = None
        self.threshold_trained = None
        self.threshold_unseen = None
        self.stop_event = Event()

        # Initialize storage structure
        self.storage_dirs = {
            'temp': config.DATA_STORAGE['realtime_temp'],
            'results': config.DATA_STORAGE['realtime_results'],
            'alerts': config.DATA_STORAGE['realtime_alerts'],
            'plots': config.DATA_STORAGE['realtime_plots'],
            'metrics': config.DATA_STORAGE['realtime_metrics']
        }

        # Enhanced metrics tracking with separate process types
        self.detection_metrics = DetectionMetrics(
            window_size=config.REALTIME_CONFIG['dynamic_window_size'],
            mse_history=defaultdict(list),
            anomaly_history=defaultdict(list),
            detection_times=defaultdict(list),
            alerts=defaultdict(list),
            process_metrics={}
        )

        # Initialize process metrics for each process type
        self._initialize_process_metrics()

        # Initialize plotting data structure
        self.plot_data = defaultdict(lambda: {
            'mse': [],
            'anomaly_rate': [],
            'threshold': [],
            'timestamps': []
        })

        # Initialize monitoring components
        self.system_monitor = SystemMonitor()
        self.anomaly_visualizer = AnomalyVisualizer()

        # Enhanced dynamic thresholds
        self.dynamic_thresholds = {}
        self.alert_thresholds = config.REALTIME_CONFIG['alert_thresholds']

        # Time tracking
        self.last_metrics_update = defaultdict(lambda: datetime.now())
        self.last_plot_update = datetime.now()
        self.metrics_interval = timedelta(seconds=config.REALTIME_CONFIG['metrics_save_interval'])
        self.plot_interval = timedelta(seconds=config.REALTIME_CONFIG['plot_interval'])

        # Create directories
        self._initialize_storage()

        # Initialize process tracking
        self.process_status = {}
        self.consecutive_anomalies = defaultdict(int)

        # Enhanced process type tracking
        self.process_types = {
            'training': set(config.TRAINING_PROCESSES),
            'testing': set(config.TESTING_PROCESSES)
        }

        # Initialize periodic reporting
        self.report_interval = timedelta(seconds=60)
        self.last_report_time = datetime.now()

        self.logger.info("RealTimeDetector initialized successfully")

    def _initialize_storage(self) -> None:
        """Initialize storage directories with proper permissions"""
        try:
            for directory in self.storage_dirs.values():
                if not os.path.exists(directory):
                    os.makedirs(directory, mode=0o755, exist_ok=True)
                    os.system(f"sudo chown -R {os.getenv('USER')}:{os.getenv('USER')} {directory}")
                    self.logger.debug(f"Created directory: {directory}")

            # Verify write permissions
            for directory in self.storage_dirs.values():
                test_file = os.path.join(directory, 'test_write.tmp')
                try:
                    with open(test_file, 'w') as f:
                        f.write('test')
                    os.remove(test_file)
                except Exception as e:
                    self.logger.error(f"Write permission test failed for {directory}: {e}")
                    raise

        except Exception as e:
            self.logger.error(f"Error initializing storage: {e}")
            raise


    def load_model_artifacts(self) -> bool:
        """Load model and artifacts with enhanced validation"""
        try:
            # Load scaler
            self.logger.info("Loading scaler...")
            scaler_path = os.path.join(config.DATA_STORAGE['models'], 'final', 'scaler.pkl')
            self.scaler = joblib.load(scaler_path)
            if not hasattr(self.scaler, 'mean_'):
                raise ValueError("Scaler not properly fitted")

            # Load model
            model_files = list(Path(os.path.join(config.DATA_STORAGE['models'], 'final')).glob('best_model_*.pth'))
            if not model_files:
                raise FileNotFoundError("No model file found")

            latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
            input_dim = self.scaler.n_features_in_
            self.logger.info(f"Model input dimension: {input_dim}")

            self.model = load_autoencoder(str(latest_model), input_dim)
            if self.model is None:
                raise ValueError("Failed to load model")
            self.model.to(self.device)
            self.model.eval()

            # Load thresholds
            threshold_path = os.path.join(config.DATA_STORAGE['models'], 'final', 'thresholds.npy')
            if os.path.exists(threshold_path):
                thresholds = np.load(threshold_path)
                self.threshold_trained, self.threshold_unseen = thresholds
                self.logger.info(
                    f"Loaded thresholds - Trained: {self.threshold_trained:.6f}, "
                    f"Unseen: {self.threshold_unseen:.6f}"
                )
            else:
                raise FileNotFoundError("Thresholds file not found")

            # Initialize dynamic thresholds
            self._initialize_dynamic_thresholds()

            self.logger.info("Model artifacts loaded successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error loading model artifacts: {e}")
            self.logger.error(traceback.format_exc())
            return False


    def _initialize_dynamic_thresholds(self) -> None:
        """Initialize dynamic thresholds for each process"""
        for process_name in self.config.TRAINING_PROCESSES + self.config.TESTING_PROCESSES:
            is_trained = process_name in self.config.TRAINING_PROCESSES
            base_threshold = self.threshold_trained if is_trained else self.threshold_unseen

            self.dynamic_thresholds[process_name] = {
                'base': base_threshold,
                'current': base_threshold,
                'history': deque(maxlen=self.detection_metrics['window_size']),
                'window_scores': deque(maxlen=self.detection_metrics['window_size']),
                'last_update': None
            }



    # def detect_anomalies(self, data: np.ndarray, process_name: str, interval_counter: int) -> Tuple[
    #     np.ndarray, np.ndarray]:
    #     """Detect anomalies with enhanced process type handling and metrics"""
    #     try:
    #         start_time = time.time()
    #         is_trained = process_name in self.config.TRAINING_PROCESSES
    #         # For Chrome (trained), low MSE is normal (0), high MSE is anomaly (1)
    #         # For Teams (untrained), high MSE is anomaly (1)
    #         expected_label = 0 if is_trained else 1
    #
    #         # Ensure process metrics are initialized
    #         if process_name not in self.detection_metrics['process_metrics']:
    #             self._initialize_process_metrics()
    #
    #         metrics = self.detection_metrics['process_metrics'][process_name]
    #
    #         # Convert to tensor and perform detection
    #         data_tensor = torch.FloatTensor(data).to(self.device)
    #
    #         # Base threshold calculation
    #         base_threshold = self.threshold_trained if is_trained else self.threshold_unseen
    #
    #         # Calculate adaptive threshold based on historical data
    #         if len(metrics['mse_values']) > 100:
    #             historical_scores = np.array(metrics['mse_values'][-1000:])
    #             historical_mean = np.mean(historical_scores)
    #             historical_std = np.std(historical_scores)
    #             historical_median = np.median(historical_scores)
    #             historical_mad = np.median(np.abs(historical_scores - historical_median))
    #
    #             # More robust adaptive threshold
    #             threshold = max(
    #                 historical_mean + (2 * historical_std),
    #                 historical_median + (3 * historical_mad),
    #                 base_threshold * (10 if is_trained else 100)  # Lower multiplier for trained processes
    #             )
    #         else:
    #             threshold = base_threshold * (10 if is_trained else 100)
    #
    #         # Perform detection
    #         self.model.eval()
    #         with torch.no_grad():
    #             reconstructions = self.model(data_tensor)
    #             mse = torch.mean(torch.pow(data_tensor - reconstructions, 2), dim=1)
    #             mse_np = mse.cpu().numpy()
    #
    #             # Detect anomalies
    #             raw_anomalies = (mse_np > threshold).astype(int)
    #
    #             if is_trained:
    #                 # For Chrome: low MSE (0) is normal, high MSE (1) is anomaly
    #                 final_anomalies = raw_anomalies
    #                 true_labels = np.zeros_like(raw_anomalies)  # Expect normal behavior
    #             else:
    #                 # For Teams: high MSE (1) is expected anomaly
    #                 final_anomalies = raw_anomalies
    #                 true_labels = np.ones_like(raw_anomalies)  # Expect anomalies
    #
    #             # Print immediate detection results
    #             anomaly_count = np.sum(final_anomalies)
    #             if anomaly_count > 0:
    #                 print(f"\n{RED}[ALERT] Anomalies Detected{RESET}")
    #                 print(f"Process: {process_name}")
    #                 print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    #                 print(f"Number of anomalies: {anomaly_count}")
    #                 print(f"Average MSE score: {np.mean(mse_np):.6f}")
    #                 print(f"Threshold: {threshold:.6f}")
    #                 print(f"Max MSE score: {np.max(mse_np):.6f}")
    #                 print(f"Process type: {'Trained' if is_trained else 'Untrained'}")
    #                 print("-" * 50)
    #
    #             # Log summary periodically
    #             if interval_counter % 10 == 0:
    #                 print(f"\n{BLUE}Detection Summary - {process_name}{RESET}")
    #                 print(f"Samples processed: {metrics['samples_processed']}")
    #                 print(f"Total anomalies: {metrics['anomalies_detected']}")
    #                 if metrics['samples_processed'] > 0:
    #                     print(
    #                         f"Anomaly rate: {(metrics['anomalies_detected'] / metrics['samples_processed']) * 100:.2f}%")
    #                 print("-" * 50)
    #
    #             # Update metrics
    #             detection_time = time.time() - start_time
    #
    #             # For metric updates, use the true labels as reference
    #             self._update_detection_metrics(
    #                 process_name=process_name,
    #                 mse_scores=mse_np,
    #                 anomalies=final_anomalies,
    #                 detection_time=detection_time,
    #                 is_trained=is_trained,
    #                 expected_label=expected_label
    #             )
    #
    #             # Update plot data
    #             self._update_plot_data(process_name, mse_np, final_anomalies, threshold)
    #
    #             # Check alerts with updated threshold
    #             self._check_alerts(process_name, mse_np, final_anomalies, threshold)
    #
    #             # Periodic reporting
    #             self._check_periodic_reporting(process_name)
    #
    #             return final_anomalies, mse_np
    #
    #     except Exception as e:
    #         self.logger.error(f"Error detecting anomalies for {process_name}: {e}")
    #         self.logger.error(traceback.format_exc())
    #         return np.array([]), np.array([])

    def detect_anomalies(self, data: np.ndarray, process_name: str, interval_counter: int) -> Tuple[
        np.ndarray, np.ndarray]:
        """Detect anomalies with enhanced process type handling and metrics"""
        try:
            start_time = time.time()
            is_trained = process_name in self.config.TRAINING_PROCESSES
            # For Chrome (trained), low MSE is normal (0), high MSE is anomaly (1)
            # For Teams (untrained), high MSE is anomaly (1)
            expected_label = 0 if is_trained else 1

            # Ensure process metrics are initialized
            if process_name not in self.detection_metrics['process_metrics']:
                self._initialize_process_metrics()

            metrics = self.detection_metrics['process_metrics'][process_name]

            # Convert to tensor and perform detection
            data_tensor = torch.FloatTensor(data).to(self.device)

            # Base threshold calculation
            base_threshold = self.threshold_trained if is_trained else self.threshold_unseen

            # Calculate adaptive threshold based on historical data
            if len(metrics['mse_values']) > 100:
                # Filter out invalid values
                historical_scores = np.array([
                    score for score in metrics['mse_values'][-1000:]
                    if not np.isnan(score) and not np.isinf(score)
                ])

                if len(historical_scores) > 0:
                    historical_mean = np.mean(historical_scores)
                    historical_std = np.std(historical_scores)
                    historical_median = np.median(historical_scores)
                    historical_mad = np.median(np.abs(historical_scores - historical_median))

                    # More robust adaptive threshold
                    threshold = max(
                        historical_mean + (2 * historical_std),
                        historical_median + (3 * historical_mad),
                        base_threshold * (10 if is_trained else 100)  # Lower multiplier for trained processes
                    )
                else:
                    threshold = base_threshold * (10 if is_trained else 100)
            else:
                threshold = base_threshold * (10 if is_trained else 100)

            # Perform detection
            self.model.eval()
            with torch.no_grad():
                reconstructions = self.model(data_tensor)
                mse = torch.mean(torch.pow(data_tensor - reconstructions, 2), dim=1)
                mse_np = mse.cpu().numpy()

                mse_np = np.maximum(mse_np, np.finfo(float).eps)

                # Validate MSE scores
                valid_indices = ~(np.isnan(mse_np) | np.isinf(mse_np))
                if not np.all(valid_indices):
                    mse_np = mse_np[valid_indices]
                    self.logger.warning(f"Found {np.sum(~valid_indices)} invalid MSE scores")

                # Detect anomalies
                raw_anomalies = (mse_np > threshold).astype(int)

                if is_trained:
                    # For Chrome: low MSE (0) is normal, high MSE (1) is anomaly
                    final_anomalies = raw_anomalies
                    true_labels = np.zeros_like(raw_anomalies)  # Normal behavior expected
                else:
                    # For Teams: high MSE (1) is expected anomaly
                    final_anomalies = raw_anomalies
                    true_labels = np.ones_like(raw_anomalies)  # Anomalous behavior expected

                # Print immediate detection results
                anomaly_count = np.sum(final_anomalies)
                if anomaly_count > 0:
                    print(f"\n{RED}[ALERT] Anomalies Detected{RESET}")
                    print(f"Process: {process_name}")
                    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"Number of anomalies: {anomaly_count}")
                    print(f"Average MSE score: {np.mean(mse_np):.6f}")
                    print(f"Threshold: {threshold:.6f}")
                    print(f"Max MSE score: {np.max(mse_np):.6f}")
                    print(f"Process type: {'Trained' if is_trained else 'Untrained'}")
                    print("-" * 50)

                # Log summary periodically
                if interval_counter % 1000 == 0:
                    print(f"\n{BLUE}Detection Summary - {process_name}{RESET}")
                    print(f"Samples processed: {metrics['samples_processed']}")
                    print(f"Total anomalies: {metrics['anomalies_detected']}")
                    if metrics['samples_processed'] > 0:
                        print(
                            f"Anomaly rate: {(metrics['anomalies_detected'] / metrics['samples_processed']) * 100:.2f}%")
                    print("-" * 50)

                # Update metrics
                detection_time = time.time() - start_time

                # For metric updates, use the proper true labels for comparison
                self._update_detection_metrics(
                    process_name=process_name,
                    mse_scores=mse_np,
                    anomalies=final_anomalies,
                    detection_time=detection_time,
                    is_trained=is_trained,
                    expected_label=expected_label
                )

                # Update plot data
                self._update_plot_data(process_name, mse_np, final_anomalies, threshold)

                # Check alerts with updated threshold
                self._check_alerts(process_name, mse_np, final_anomalies, threshold)

                # Periodic reporting
                self._check_periodic_reporting(process_name)

                return final_anomalies, mse_np

        except Exception as e:
            self.logger.error(f"Error detecting anomalies for {process_name}: {e}")
            self.logger.error(traceback.format_exc())
            return np.array([]), np.array([])


    def _update_dynamic_threshold(self, process_name: str, threshold_info: Dict, recent_scores: List[float]) -> float:
        """Update dynamic threshold with improved adaptation"""
        try:
            current_time = datetime.now()

            # Check if update is needed (every 5 minutes)
            if threshold_info['last_update'] is None or \
                    (current_time - threshold_info['last_update']).total_seconds() > 300:

                # Get base threshold
                is_trained = process_name in self.config.TRAINING_PROCESSES
                base_threshold = self.threshold_trained if is_trained else self.threshold_unseen

                # Add exponential moving average
                if recent_scores:
                    alpha = 0.1
                    new_threshold = alpha * np.mean(recent_scores) + (1 - alpha) * threshold_info['current']
                    return min(new_threshold, threshold_info['base'] * 2)  # Cap maximum threshold

                if len(recent_scores) > 0:
                    # Calculate new threshold based on recent history
                    recent_scores_array = np.array(recent_scores[-100:])  # Use last 100 scores
                    mean_score = np.mean(recent_scores_array)
                    std_score = np.std(recent_scores_array)

                    # Calculate adaptive components
                    percentile_threshold = np.percentile(recent_scores_array, 95)
                    std_threshold = mean_score + 2 * std_score

                    # Use the minimum of different threshold calculations
                    new_threshold = min(
                        max(percentile_threshold, std_threshold),
                        base_threshold * 1.5  # Cap at 150% of base threshold
                    )

                    # Apply smoothing
                    alpha = 0.3
                    threshold_info['current'] = (alpha * new_threshold +
                                                 (1 - alpha) * threshold_info['current'])
                    threshold_info['last_update'] = current_time

                    # Log threshold update
                    self.logger.debug(
                        f"Updated threshold for {process_name}: {threshold_info['current']:.6f} "
                        f"(base: {base_threshold:.6f})"
                    )

            return threshold_info['current']

        except Exception as e:
            self.logger.error(f"Error updating threshold for {process_name}: {e}")
            is_trained = process_name in self.config.TRAINING_PROCESSES
            return self.threshold_trained if is_trained else self.threshold_unseen


    # def _update_detection_metrics(self, process_name: str, mse_scores: np.ndarray,
    #                               anomalies: np.ndarray, detection_time: float,
    #                               is_trained: bool, expected_label: int) -> None:
    #     """Update detection metrics with enhanced tracking"""
    #     try:
    #         metrics = self.detection_metrics['process_metrics'][process_name]
    #
    #         # Validate MSE scores before processing
    #         valid_mse_scores = []
    #         for score in mse_scores.tolist():
    #             if not np.isnan(score) and not np.isinf(score):
    #                 valid_mse_scores.append(abs(score))  # Ensure positive values
    #
    #         # Add exponential smoothing for MSE scores
    #         if valid_mse_scores:
    #             alpha = 0.1  # Smoothing factor
    #             if 'smoothed_mse' not in metrics:
    #                 metrics['smoothed_mse'] = valid_mse_scores[0]
    #             for score in valid_mse_scores:
    #                 metrics['smoothed_mse'] = alpha * score + (1 - alpha) * metrics['smoothed_mse']
    #
    #         # Update basic metrics
    #         metrics['samples_processed'] += len(mse_scores)
    #         metrics['anomalies_detected'] += int(np.sum(anomalies))
    #         metrics['mse_values'].extend(valid_mse_scores)  # Use validated scores
    #         metrics['detection_times'].append(detection_time)
    #
    #         # For trained processes, normal behavior (0) is expected
    #         # For untrained processes, anomalous behavior (1) is expected
    #
    #
    #         # Update predictions and labels
    #         metrics['predicted_labels'].extend(anomalies.tolist())
    #         true_labels = np.zeros_like(anomalies) if is_trained else np.ones_like(anomalies)
    #         metrics['true_labels'].extend(true_labels.tolist())
    #
    #         # Update confusion matrix metrics
    #         for pred, true in zip(anomalies, true_labels):
    #             if true == 1:  # Expected anomalous behavior
    #                 if pred == 1:
    #                     metrics['true_positives'] += 1  # Correctly detected anomaly
    #                 else:
    #                     metrics['false_negatives'] += 1  # Missed anomaly
    #             else:  # Expected normal behavior
    #                 if pred == 1:
    #                     metrics['false_positives'] += 1  # False alarm
    #                 else:
    #                     metrics['true_negatives'] += 1  # Correctly identified normal
    #
    #         # Update interval metrics with validated scores
    #         metrics['interval_metrics']['interval_mse'].extend(valid_mse_scores)
    #         metrics['interval_metrics']['interval_anomalies'].extend(anomalies.tolist())
    #         metrics['interval_metrics']['interval_times'].append(detection_time)
    #
    #         # Calculate performance metrics
    #         tp = metrics['true_positives']
    #         fp = metrics['false_positives']
    #         tn = metrics['true_negatives']
    #         fn = metrics['false_negatives']
    #
    #         # Calculate rates with validation
    #         if metrics['samples_processed'] > 0:
    #             # Precision
    #             metrics['precision'] = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    #
    #             # Recall
    #             metrics['recall'] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    #
    #             # F1 Score
    #             if metrics['precision'] + metrics['recall'] > 0:
    #                 metrics['f1_score'] = float(2 * (metrics['precision'] * metrics['recall']) /
    #                                             (metrics['precision'] + metrics['recall']))
    #             else:
    #                 metrics['f1_score'] = 0.0
    #
    #             # Accuracy
    #             metrics['accuracy'] = float((tp + tn) / metrics['samples_processed'])
    #
    #             # Specificity
    #             metrics['specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    #
    #             # Calculate anomaly rate
    #             anomaly_rate = metrics['anomalies_detected'] / metrics['samples_processed']
    #             if not np.isnan(anomaly_rate):
    #                 metrics['anomaly_rates'].append(float(anomaly_rate))
    #
    #         # Update last update timestamp
    #         metrics['last_update'] = datetime.now()
    #
    #         # Maintain memory efficiency by keeping limited history
    #         max_history = 10000  # Adjust as needed
    #         for key in ['mse_values', 'detection_times', 'anomaly_rates']:
    #             if len(metrics[key]) > max_history:
    #                 metrics[key] = metrics[key][-max_history:]
    #
    #         # Keep interval metrics within reasonable size
    #         for key in metrics['interval_metrics']:
    #             if len(metrics['interval_metrics'][key]) > max_history:
    #                 metrics['interval_metrics'][key] = metrics['interval_metrics'][key][-max_history:]
    #
    #     except Exception as e:
    #         self.logger.error(f"Error updating metrics for {process_name}: {e}")
    #         self.logger.error(traceback.format_exc())

    def _update_detection_metrics(self, process_name: str, mse_scores: np.ndarray,
                                  anomalies: np.ndarray, detection_time: float,
                                  is_trained: bool, expected_label: int) -> None:
        """Update detection metrics with enhanced tracking"""
        try:
            metrics = self.detection_metrics['process_metrics'][process_name]

            # Validate MSE scores before processing
            valid_mask = ~(np.isnan(mse_scores) | np.isinf(mse_scores))
            valid_mse_scores = mse_scores[valid_mask].tolist()
            valid_anomalies = anomalies[valid_mask].tolist()

            # Update basic metrics
            metrics['samples_processed'] += len(valid_mse_scores)
            metrics['anomalies_detected'] += int(np.sum(valid_anomalies))
            metrics['mse_values'].extend(valid_mse_scores)
            metrics['detection_times'].append(detection_time)

            # For trained processes (Chrome), normal behavior is 0, anomaly is 1
            # For untrained processes (Teams), expected behavior is 1
            if is_trained:
                metrics['predicted_labels'].extend(valid_anomalies)  # Predicted anomalies
                expected_behavior = np.zeros_like(valid_anomalies)  # Expect normal (0)
            else:
                metrics['predicted_labels'].extend(valid_anomalies)  # Predicted anomalies
                expected_behavior = np.ones_like(valid_anomalies)  # Expect anomalies (1)

            metrics['true_labels'].extend(expected_behavior.tolist())

            # Calculate confusion matrix
            predictions = np.array(metrics['predicted_labels'])
            true_labels = np.array(metrics['true_labels'])

            if len(predictions) > 0 and len(true_labels) > 0:
                tp = np.sum((true_labels == 1) & (predictions == 1))
                fp = np.sum((true_labels == 0) & (predictions == 1))
                tn = np.sum((true_labels == 0) & (predictions == 0))
                fn = np.sum((true_labels == 1) & (predictions == 0))

                metrics['true_positives'] = int(tp)
                metrics['false_positives'] = int(fp)
                metrics['true_negatives'] = int(tn)
                metrics['false_negatives'] = int(fn)

                total = tp + fp + tn + fn

                # Calculate metrics
                if total > 0:
                    if is_trained:  # For Chrome
                        # Recalculate TP, FP, TN, FN for Chrome
                        # Chrome expects normal behavior (0), so high MSE/anomalies are deviations
                        tp = int(np.sum((anomalies == 1) & (true_labels == 1)))  # Correctly detected anomalies
                        fp = int(np.sum((anomalies == 1) & (true_labels == 0)))  # False alarms
                        tn = int(np.sum((anomalies == 0) & (true_labels == 0)))  # Correctly identified normal
                        fn = int(np.sum((anomalies == 0) & (true_labels == 1)))  # Missed anomalies

                        # Store updated confusion matrix metrics
                        metrics['true_positives'] = tp
                        metrics['false_positives'] = fp
                        metrics['true_negatives'] = tn
                        metrics['false_negatives'] = fn

                        # Accuracy
                        metrics['accuracy'] = float((tp + tn) / total)

                        # For Chrome, precision is how many of our anomaly detections were correct
                        metrics['precision'] = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0

                        # For Chrome, recall is how many actual anomalies we caught
                        metrics['recall'] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

                        # For Chrome, specificity is how well we identify normal behavior
                        metrics['specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

                        # F1 Score
                        if metrics['precision'] > 0 and metrics['recall'] > 0:
                            metrics['f1_score'] = float(2 * (metrics['precision'] * metrics['recall']) /
                                                        (metrics['precision'] + metrics['recall']))
                        else:
                            metrics['f1_score'] = 0.0

                    else:  # Keep your existing code for Teams
                        metrics['accuracy'] = float((tp + tn) / total)
                        metrics['precision'] = float(tp / (tp + fp)) if (tp + fp) > 0 else 1.0
                        metrics['recall'] = float(tp / (tp + fn)) if (tp + fn) > 0 else 1.0
                        metrics['specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 1.0
                        if metrics['precision'] + metrics['recall'] > 0:
                            metrics['f1_score'] = float(2 * (metrics['precision'] * metrics['recall']) /
                                                        (metrics['precision'] + metrics['recall']))
                        else:
                            metrics['f1_score'] = 0.0

                    # Anomaly rate
                    metrics['anomaly_rates'].append(
                        metrics['anomalies_detected'] / metrics['samples_processed']
                    )

            # Update interval metrics
            metrics['interval_metrics']['interval_mse'].extend(valid_mse_scores)
            metrics['interval_metrics']['interval_anomalies'].extend(valid_anomalies)
            metrics['interval_metrics']['interval_times'].append(detection_time)

            # Update timestamp
            metrics['last_update'] = datetime.now()

            # Manage history size
            max_history = 10000
            for key in ['mse_values', 'detection_times', 'anomaly_rates',
                        'predicted_labels', 'true_labels']:
                if key in metrics and len(metrics[key]) > max_history:
                    metrics[key] = metrics[key][-max_history:]

            for key in metrics['interval_metrics']:
                if len(metrics['interval_metrics'][key]) > max_history:
                    metrics['interval_metrics'][key] = metrics['interval_metrics'][key][-max_history:]

        except Exception as e:
            self.logger.error(f"Error updating metrics for {process_name}: {e}")
            self.logger.error(traceback.format_exc())


    def _update_plot_data(self, process_name: str, mse_scores: np.ndarray,
                          anomalies: np.ndarray, threshold: float) -> None:
        """Update plotting data with current detection results"""
        try:
            plot_data = self.plot_data[process_name]
            current_time = datetime.now()

            plot_data['mse'].append(float(np.mean(mse_scores)))
            plot_data['anomaly_rate'].append(float(np.mean(anomalies)))
            plot_data['threshold'].append(float(threshold))
            plot_data['timestamps'].append(current_time)

            # Keep only recent data for plotting (last hour)
            cutoff_time = current_time - timedelta(hours=1)
            while plot_data['timestamps'] and plot_data['timestamps'][0] < cutoff_time:
                for key in ['mse', 'anomaly_rate', 'threshold', 'timestamps']:
                    plot_data[key].pop(0)

        except Exception as e:
            self.logger.error(f"Error updating plot data for {process_name}: {e}")


    # def _check_alerts(self, process_name: str, mse_scores: np.ndarray,
    #                   anomalies: np.ndarray, threshold: float) -> None:
    #     """Check for alert conditions with improved alerting logic"""
    #     try:
    #         # Calculate current metrics
    #         anomaly_rate = np.mean(anomalies)
    #         max_mse = np.max(mse_scores)
    #
    #         # # Adjusted thresholds
    #         # rate_threshold = 0.15  # Reduced from 0.30
    #         # mse_multiplier = 1.5  # Reduced from 2.0
    #
    #         # Get alert thresholds
    #         alert_conditions = []
    #
    #         # Check anomaly rate threshold
    #         if anomaly_rate > self.alert_thresholds['anomaly_rate']:
    #             alert_conditions.append(
    #                 f"High anomaly rate: {anomaly_rate:.2%} "
    #                 f"(threshold: {self.alert_thresholds['anomaly_rate']:.2%})"
    #             )
    #
    #         # Check consecutive anomalies
    #         if np.sum(anomalies) > 0:
    #             self.consecutive_anomalies[process_name] += 1
    #             if self.consecutive_anomalies[process_name] >= self.alert_thresholds['consecutive_anomalies']:
    #                 alert_conditions.append(
    #                     f"Consecutive anomalies: {self.consecutive_anomalies[process_name]}"
    #                 )
    #         else:
    #             self.consecutive_anomalies[process_name] = 0
    #
    #         # Check MSE threshold
    #         if max_mse > threshold * self.alert_thresholds['mse_multiplier']:
    #             alert_conditions.append(
    #                 f"High MSE score: {max_mse:.6f} "
    #                 f"(threshold: {threshold:.6f})"
    #             )
    #
    #         # Generate alert if conditions are met
    #         if alert_conditions:
    #             self._generate_alert(process_name, alert_conditions, mse_scores, anomalies)
    #
    #     except Exception as e:
    #         self.logger.error(f"Error checking alerts for {process_name}: {e}")

    def _check_alerts(self, process_name: str, mse_scores: np.ndarray,
                      anomalies: np.ndarray, threshold: float) -> None:
        """Check for alert conditions with improved alerting logic"""
        try:
            # Calculate current metrics with validation
            anomaly_rate = float(np.mean(anomalies)) if len(anomalies) > 0 else 0.0
            max_mse = float(np.max(mse_scores)) if len(mse_scores) > 0 else 0.0

            alert_conditions = []

            # Check anomaly rate threshold with strict limit
            if anomaly_rate > self.alert_thresholds['anomaly_rate'] and anomaly_rate < 0.1:  # Only if < 10%
                alert_conditions.append(
                    f"High anomaly rate: {anomaly_rate:.2%} "
                    f"(threshold: {self.alert_thresholds['anomaly_rate']:.2%})"
                )

            # Update consecutive anomalies with rate limiting
            if np.sum(anomalies) > 0 and anomaly_rate < 0.1:  # Only count if rate is reasonable
                self.consecutive_anomalies[process_name] += 1
                if self.consecutive_anomalies[process_name] >= self.alert_thresholds['consecutive_anomalies']:
                    alert_conditions.append(
                        f"Consecutive anomalies: {self.consecutive_anomalies[process_name]}"
                    )
            else:
                self.consecutive_anomalies[process_name] = 0

            # Check MSE threshold with dynamic multiplier
            #mse_multiplier = max(1000, threshold)  # Ensure minimum multiplier
            if max_mse > threshold * 1.5:  # Use fixed multiplier for stability
                alert_conditions.append(
                    f"High MSE score: {max_mse:.6f} "
                    f"(threshold: {threshold:.6f})"
                )

            # Generate alert if conditions are met
            if alert_conditions:
                self._generate_alert(process_name, alert_conditions, mse_scores, anomalies)

        except Exception as e:
            self.logger.error(f"Error checking alerts for {process_name}: {e}")


    def _generate_alert(self, process_name: str, conditions: List[str],
                        mse_scores: np.ndarray, anomalies: np.ndarray) -> None:
        """Generate and store alert with enhanced context"""
        try:
            current_time = datetime.now()

            # Create alert with detailed context
            alert = {
                'timestamp': current_time.isoformat(),
                'process': process_name,
                'conditions': conditions,
                'metrics': {
                    'avg_mse': float(np.mean(mse_scores)),
                    'max_mse': float(np.max(mse_scores)),
                    'anomaly_rate': float(np.mean(anomalies)),
                    'sample_count': len(anomalies)
                },
                'context': {
                    'process_type': 'training' if process_name in self.config.TRAINING_PROCESSES else 'testing',
                    'total_anomalies': int(np.sum(anomalies)),
                    'consecutive_anomalies': self.consecutive_anomalies[process_name]
                }
            }

            # Store alert
            self.detection_metrics['alerts'][process_name].append(alert)

            # Save alert to file
            self._save_alert(alert)

            # Log alert
            alert_message = f"\nALERT for {process_name}:\n"
            for condition in conditions:
                alert_message += f"- {condition}\n"
            alert_message += f"Context: {len(anomalies)} samples processed, "
            alert_message += f"{int(np.sum(anomalies))} anomalies detected"

            self.logger.warning(alert_message)

        except Exception as e:
            self.logger.error(f"Error generating alert for {process_name}: {e}")


    def _check_periodic_reporting(self, process_name: str) -> None:
        """Check and generate periodic reports"""
        try:
            current_time = datetime.now()

            # Update metrics if interval has passed
            if (current_time - self.last_metrics_update[process_name]) >= self.metrics_interval:
                self._save_interval_metrics(process_name)
                self.last_metrics_update[process_name] = current_time

            # Update plots if interval has passed
            if (current_time - self.last_plot_update) >= self.plot_interval:
                self._update_plots()
                self.last_plot_update = current_time

            # Generate comprehensive report if report interval has passed
            if (current_time - self.last_report_time) >= self.report_interval:
                self._generate_comprehensive_report()
                self.last_report_time = current_time

        except Exception as e:
            self.logger.error(f"Error in periodic reporting: {e}")


    def _save_interval_metrics(self, process_name: str) -> None:
        """Save interval metrics with enhanced statistics"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            metrics = self.detection_metrics['process_metrics'][process_name]

            # Calculate interval statistics
            interval_stats = {
                'timestamp': timestamp,
                'process_name': process_name,
                'process_type': 'training' if process_name in self.config.TRAINING_PROCESSES else 'testing',
                'interval_metrics': {
                    'samples_processed': len(metrics['interval_metrics']['interval_mse']),
                    'anomalies_detected': sum(metrics['interval_metrics']['interval_anomalies']),
                    'avg_mse': float(np.mean(metrics['interval_metrics']['interval_mse']))
                    if metrics['interval_metrics']['interval_mse'] else 0,
                    'max_mse': float(np.max(metrics['interval_metrics']['interval_mse']))
                    if metrics['interval_metrics']['interval_mse'] else 0,
                    'std_mse': float(np.std(metrics['interval_metrics']['interval_mse']))
                    if metrics['interval_metrics']['interval_mse'] else 0,
                    'avg_detection_time': float(np.mean(metrics['interval_metrics']['interval_times']))
                    if metrics['interval_metrics']['interval_times'] else 0
                },
                'cumulative_metrics': {
                    'total_samples': metrics['samples_processed'],
                    'total_anomalies': metrics['anomalies_detected'],
                    'anomaly_rate': metrics['anomalies_detected'] / metrics['samples_processed']
                    if metrics['samples_processed'] > 0 else 0,
                    'true_positives': metrics['true_positives'],
                    'false_positives': metrics['false_positives'],
                    'true_negatives': metrics['true_negatives'],
                    'false_negatives': metrics['false_negatives']
                }
            }

            # Calculate performance metrics if we have enough data
            if metrics['samples_processed'] > 0:
                interval_stats['performance_metrics'] = self._calculate_performance_metrics(metrics)

            # Save to file
            metrics_file = os.path.join(
                self.storage_dirs['metrics'],
                f'interval_metrics_{process_name.replace("/", "_")}_{timestamp}.json'
            )

            with open(metrics_file, 'w') as f:
                json.dump(interval_stats, f, indent=4)

            # Clear interval metrics after saving
            metrics['interval_metrics'] = {
                'interval_mse': [],
                'interval_anomalies': [],
                'interval_times': []
            }

            self.logger.info(f"Saved interval metrics for {process_name}")

        except Exception as e:
            self.logger.error(f"Error saving interval metrics for {process_name}: {e}")


    def _calculate_performance_metrics(self, metrics: Dict) -> Dict:
        """Calculate comprehensive performance metrics"""
        try:
            tp = metrics['true_positives']
            fp = metrics['false_positives']
            tn = metrics['true_negatives']
            fn = metrics['false_negatives']

            total = tp + fp + tn + fn
            if total == 0:
                return {}

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (tp + tn) / total

            return {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'specificity': float(specificity),
                'f1_score': float(f1),
                'false_positive_rate': float(fp / (fp + tn)) if (fp + tn) > 0 else 0,
                'false_negative_rate': float(fn / (fn + tp)) if (fn + tp) > 0 else 0
            }

        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {e}")
            return {}


    def _save_alert(self, alert: Dict) -> None:
        """Save alert to file with proper formatting"""
        try:
            alert_file = os.path.join(
                self.storage_dirs['alerts'],
                f"alert_{alert['process'].replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )

            with open(alert_file, 'w') as f:
                json.dump(alert, f, indent=4)

            self.logger.info(f"Alert saved to: {alert_file}")

        except Exception as e:
            self.logger.error(f"Error saving alert: {e}")

    def _update_plots(self) -> None:
        """Update visualization plots with enhanced metrics"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plots_dir = self.storage_dirs['plots']
            os.makedirs(plots_dir, exist_ok=True)  # Ensure directory exists

            # Create figure with subplots for different metrics
            fig = plt.figure(figsize=(20, 15))
            gs = plt.GridSpec(3, 2)

            try:
                # Plot 1: MSE Overview
                ax1 = fig.add_subplot(gs[0, :])
                for process_name, plot_data in self.plot_data.items():
                    if plot_data['mse']:
                        ax1.plot(plot_data['timestamps'], plot_data['mse'],
                                 label=f"{process_name} ({'Training' if process_name in self.config.TRAINING_PROCESSES else 'Testing'})")
                ax1.set_title('MSE Scores Over Time')
                ax1.set_ylabel('MSE Score')
                ax1.legend()
                ax1.grid(True)

                # Plot 2: Anomaly Rates
                ax2 = fig.add_subplot(gs[1, 0])
                for process_name, plot_data in self.plot_data.items():
                    if plot_data['anomaly_rate']:
                        ax2.plot(plot_data['timestamps'], plot_data['anomaly_rate'],
                                 label=process_name)
                ax2.set_title('Anomaly Rates')
                ax2.set_ylabel('Rate')
                ax2.legend()
                ax2.grid(True)

                # Plot 3: Detection Times
                ax3 = fig.add_subplot(gs[1, 1])
                for process_name, metrics in self.detection_metrics['process_metrics'].items():
                    if metrics['detection_times']:
                        ax3.plot(metrics['detection_times'][-100:], label=process_name)
                ax3.set_title('Recent Detection Times')
                ax3.set_ylabel('Time (s)')
                ax3.legend()
                ax3.grid(True)

                # Plot 4: System Resources
                ax4 = fig.add_subplot(gs[2, 0])
                system_metrics = self.system_monitor.get_metrics()
                if system_metrics.get('cpu_usage') and system_metrics.get('memory_usage'):
                    times = list(range(len(system_metrics['cpu_usage'])))
                    ax4.plot(times, system_metrics['cpu_usage'], label='CPU')
                    ax4.plot(times, system_metrics['memory_usage'], label='Memory')
                ax4.set_title('System Resource Usage')
                ax4.set_ylabel('Usage %')
                ax4.legend()
                ax4.grid(True)

                # Plot 5: Performance Metrics
                ax5 = fig.add_subplot(gs[2, 1])
                for process_name, metrics in self.detection_metrics['process_metrics'].items():
                    if metrics['samples_processed'] > 0:
                        perf_metrics = self._calculate_performance_metrics(metrics)
                        if perf_metrics:
                            ax5.bar(process_name,
                                    [perf_metrics[m] for m in ['accuracy', 'precision', 'recall']],
                                    label=['Accuracy', 'Precision', 'Recall'])
                ax5.set_title('Performance Metrics by Process')
                ax5.set_ylabel('Score')
                ax5.legend()

                plt.tight_layout()
                plot_path = os.path.join(plots_dir, f'monitoring_{timestamp}.png')

                # Save figure with error handling
                try:
                    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
                    self.logger.info(f"Updated plots saved to: {plot_path}")
                except Exception as e:
                    self.logger.error(f"Error saving plot to {plot_path}: {e}")

            finally:
                # Ensure figure is closed even if plotting fails
                plt.close(fig)
                plt.close('all')

        except Exception as e:
            self.logger.error(f"Error updating plots: {e}")
            self.logger.error(traceback.format_exc())
            plt.close('all')  # Emergency cleanup

    def _generate_comprehensive_report(self) -> None:
        """Generate comprehensive monitoring report with enhanced metrics"""
        try:
            print("\n" + "=" * 50)
            print("Real-Time Detection Status Report")
            print("=" * 50)

            # Runtime information
            runtime = datetime.now() - self.start_time
            print(f"\nRuntime: {runtime}")

            # Overall statistics
            # total_samples = 0
            # total_anomalies = 0
            # all_true_labels = []
            # all_pred_labels = []
            total_samples, total_anomalies = 0, 0
            all_true_labels, all_pred_labels = [], []
            all_detection_times = []

            for process_name, metrics in self.detection_metrics['process_metrics'].items():
                total_samples += metrics.get('samples_processed', 0)
                total_anomalies += metrics.get('anomalies_detected', 0)
                all_true_labels.extend(metrics.get('true_labels', []))
                all_pred_labels.extend(metrics.get('predicted_labels', []))
                # total_samples += metrics['samples_processed']
                # total_anomalies += metrics['anomalies_detected']
                # all_true_labels.extend(metrics['true_labels'])
                # all_pred_labels.extend(metrics['predicted_labels'])
                all_detection_times.extend(metrics.get('detection_times', []))

            print("\nOverall Statistics:")
            print("-" * 30)
            print(f"Total Samples Processed: {total_samples:,}")
            print(f"Total Anomalies Detected: {total_anomalies:,}")

            if all_true_labels and all_pred_labels:
                # Fix: Add data validation before calculations
                y_true = np.array(all_true_labels)
                y_pred = np.array(all_pred_labels)

                # Calculate confusion matrix with validation
                try:
                    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                    total = tn + fp + fn + tp

                    # Calculate metrics with proper validation
                    accuracy = (tp + tn) / total if total > 0 else 0
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

                    print(f"\nPerformance Metrics:")
                    print(f"Accuracy:    {accuracy:.4f}")
                    print(f"Precision:   {precision:.4f}")
                    print(f"Recall:      {recall:.4f}")
                    print(f"F1 Score:    {f1:.4f}")
                    print(f"Specificity: {specificity:.4f}")

                    print(f"\nConfusion Matrix:")
                    print(f"True Negatives:  {tn}")
                    print(f"False Positives: {fp}")
                    print(f"False Negatives: {fn}")
                    print(f"True Positives:  {tp}")

                except Exception as e:
                    self.logger.error(f"Error calculating confusion matrix: {e}")

            if total_samples > 0:
                print(f"Overall Anomaly Rate: {(total_anomalies / total_samples) * 100:.2f}%")
            if all_detection_times:
                print(f"Average Detection Time: {np.mean(all_detection_times):.4f}s")

            # Process summaries with MSE validation
            for process_type in ['Training', 'Testing']:
                processes = self.config.TRAINING_PROCESSES if process_type == 'Training' \
                    else self.config.TESTING_PROCESSES

                print(f"\n{process_type} Processes:")
                print("-" * 30)

                for process_name in processes:
                    metrics = self.detection_metrics['process_metrics'].get(process_name)
                    if metrics and metrics.get('samples_processed', 0) > 0:
                        print(f"\n{process_name}:")
                        samples = metrics['samples_processed']
                        anomalies = metrics.get('anomalies_detected', 0)

                        # Fix: Validate MSE scores before calculations
                        mse_scores = metrics.get('mse_scores', [])
                        mse_scores = [score for score in mse_scores if not np.isnan(score)]  # Remove NaN values
                        avg_mse = np.mean(mse_scores) if mse_scores else 0

                        print(f"- Samples Processed: {samples:,}")
                        print(f"- Anomalies Detected: {anomalies} ({(anomalies / samples) * 100:.2f}%)")
                        print(f"- Average MSE: {avg_mse:.6f}")

                        detection_times = metrics.get('detection_times', [])
                        if detection_times:
                            print(f"- Average Detection Time: {np.mean(detection_times):.4f}s")

                        # Calculate process-specific metrics if available
                        if metrics.get('true_labels') and metrics.get('predicted_labels'):
                            try:
                                y_true = np.array(metrics['true_labels'])
                                y_pred = np.array(metrics['predicted_labels'])

                                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                                total = tn + fp + fn + tp

                                if total > 0:
                                    accuracy = (tp + tn) / total
                                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                                    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                                    f1 = 2 * (precision * recall) / (precision + recall) if (
                                                                                                        precision + recall) > 0 else 0

                                    print("\nMetrics:")
                                    print(f"Accuracy:    {accuracy:.4f}")
                                    print(f"Precision:   {precision:.4f}")
                                    print(f"Recall:      {recall:.4f}")
                                    print(f"F1 Score:    {f1:.4f}")
                                    print(f"Specificity: {specificity:.4f}")

                            except Exception as e:
                                self.logger.error(f"Error calculating process metrics for {process_name}: {e}")

            # Add resource usage with validation
            system_metrics = self.system_monitor.get_metrics()
            print("\nSystem Resource Usage:")
            print("-" * 30)

            cpu_usage = system_metrics.get('cpu_usage', [])
            memory_usage = system_metrics.get('memory_usage', [])

            cpu_avg = np.mean(cpu_usage) if cpu_usage else 0
            memory_avg = np.mean(memory_usage) if memory_usage else 0

            print(f"CPU Usage: {cpu_avg:.2f}%")
            print(f"Memory Usage: {memory_avg:.2f}%")

            # Show recent alerts
            if self.detection_metrics.get('alerts'):
                print("\nRecent Alerts:")
                print("-" * 30)
                recent_alerts = []
                for process_alerts in self.detection_metrics['alerts'].values():
                    recent_alerts.extend(process_alerts[-5:])  # Get 5 most recent alerts

                if recent_alerts:
                    recent_alerts.sort(key=lambda x: x['timestamp'], reverse=True)
                    for alert in recent_alerts[:5]:
                        print(f"\nProcess: {alert['process']}")
                        print(f"Time: {alert['timestamp']}")
                        for condition in alert['conditions']:
                            print(f"- {condition}")
                else:
                    print("No alerts detected")

            print("\n" + "=" * 50)

        except Exception as e:
            self.logger.error(f"Error generating comprehensive report: {e}")
            self.logger.error(traceback.format_exc())

    def monitor_process(self, process_name: str) -> None:
        """Monitor single process with enhanced error handling and metrics"""
        start_time = datetime.now()  # Initialize start_time at the very beginning
        try:
            interval_counter = 0
            syscall_buffer = []
            process_not_found_count = 0
            MAX_NOT_FOUND_COUNT = 10
            last_runtime_log = start_time

            # Get the process metrics from the already initialized metrics
            process_metrics = self.detection_metrics['process_metrics'].get(process_name)
            if not process_metrics:
                process_metrics = {
                    'samples_processed': 0,
                    'anomalies_detected': 0,
                    'mse_values': [],
                    'detection_times': [],
                    'anomaly_rates': [],
                    'total_syscalls': 0,
                    'true_labels': [],
                    'predicted_labels': [],
                    'true_positives': 0,
                    'false_positives': 0,
                    'true_negatives': 0,
                    'false_negatives': 0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1_score': 0.0,
                    'accuracy': 0.0,
                    'specificity': 0.0,
                    'start_time': start_time,
                    'last_update': start_time,
                    'interval_metrics': {
                        'interval_mse': [],
                        'interval_anomalies': [],
                        'interval_times': []
                    }
                }
                self.detection_metrics['process_metrics'][process_name] = process_metrics

            self.logger.info(f"Starting monitoring for {process_name}")
            print(f"\nStarting monitoring for: {process_name}")
            print(f"Process type: {'Training' if process_name in self.config.TRAINING_PROCESSES else 'Testing'}")

            while not self.stop_event.is_set():
                try:
                    current_time = datetime.now()
                    runtime = current_time - start_time

                    # Log runtime every minute
                    if (current_time - last_runtime_log).total_seconds() >= 60:
                        self.logger.info(f"Process {process_name} running for {runtime}")
                        # safe_name = process_name.replace('/', '_').replace(' ', '_')
                        # log_file = os.path.join(self.storage_dirs['temp'],
                        #                         f"realtime_{safe_name}_{curr_time}.log")
                        print(f"Process {process_name} running for {runtime}")
                        last_runtime_log = current_time

                    # Validate process is running
                    pids = get_process_pids(process_name)
                    if not pids:
                        process_not_found_count += 1
                        if process_not_found_count >= MAX_NOT_FOUND_COUNT:
                            self.logger.warning(
                                f"Process {process_name} not found for {MAX_NOT_FOUND_COUNT} consecutive checks. "
                                f"Total runtime: {runtime}"
                            )
                            process_not_found_count = 0
                        time.sleep(config.COLLECTION_INTERVAL)
                        continue
                    else:
                        process_not_found_count = 0

                    # Create file names with timestamp
                    curr_time = int(time.time())
                    safe_name = process_name.replace('/', '_').replace(' ', '_')

                    log_file = os.path.join(
                        self.storage_dirs['temp'],
                        f"realtime_{safe_name}_{curr_time}.log"
                    )
                    text_file = os.path.join(
                        self.storage_dirs['temp'],
                        f"processed_{safe_name}_{curr_time}.log"
                    )

                    # Collect and process syscalls
                    if collect_syscalls(log_file):
                        if convert_json_to_text(log_file, text_file, mode='test'):
                            new_syscalls = read_syscalls_from_log(text_file)

                            if new_syscalls:
                                syscall_buffer.extend(new_syscalls)

                                # Process when buffer reaches minimum size
                                if len(syscall_buffer) >= config.MIN_SAMPLES_REQUIRED:
                                    # Create windows with overlap
                                    total_syscalls = len(syscall_buffer)
                                    window_size = config.MIN_SAMPLES_REQUIRED
                                    stride = window_size // 4  # 75% overlap
                                    num_windows = (total_syscalls - window_size) // stride + 1

                                    for i in range(num_windows):
                                        start_idx = i * stride
                                        end_idx = start_idx + window_size
                                        window = syscall_buffer[start_idx:end_idx]

                                        # Preprocess window
                                        features = preprocess_data(
                                            [window],
                                            interval_counter=interval_counter,
                                            mode='test',
                                            scaler=self.scaler
                                        )

                                        if features is not None and len(features) > 0:
                                            # Detect anomalies
                                            anomalies, mse = self.detect_anomalies(
                                                features,
                                                process_name,
                                                interval_counter
                                            )

                                            if anomalies is not None:
                                                interval_counter += 1

                                                # Log window results
                                                anomaly_count = np.sum(anomalies)
                                                if anomaly_count > 0:
                                                    self.logger.info(
                                                        f"Process: {process_name}, "
                                                        f"Runtime: {runtime}, "
                                                        f"Window: {interval_counter}, "
                                                        f"Anomalies: {anomaly_count}, "
                                                        f"Average MSE: {np.mean(mse):.6f}"
                                                    )

                                    # Clear buffer after processing, keeping overlap
                                    syscall_buffer = syscall_buffer[end_idx - window_size // 2:]

                                    # Update process status
                                    self.process_status[process_name] = {
                                        'last_update': datetime.now(),
                                        'status': 'active',
                                        'windows_processed': interval_counter,
                                        'runtime': str(runtime)
                                    }

                    # Cleanup temporary files
                    clean_up_files(log_file, text_file)

                except Exception as e:
                    self.logger.error(
                        f"Error in monitoring loop for {process_name} "
                        f"(Runtime: {datetime.now() - start_time}): {str(e)}"
                    )
                    self.logger.error(traceback.format_exc())
                    continue

                time.sleep(config.COLLECTION_INTERVAL)

        except Exception as e:
            self.logger.error(
                f"Critical error monitoring {process_name} "
                f"after runtime of {datetime.now() - start_time}: {e}"
            )
            self.logger.error(traceback.format_exc())

        finally:
            # Final cleanup and metrics saving
            total_runtime = datetime.now() - start_time
            self.logger.info(
                f"Monitoring ended for {process_name}. "
                f"Total runtime: {total_runtime}"
            )
            self._save_final_process_metrics(process_name)

    def _calculate_thresholds(self, train_loader: DataLoader) -> None:
        """Calculate anomaly detection thresholds using global config"""
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

            reconstruction_errors_array = np.array(reconstruction_errors)

            # Add validation for extreme values
            if len(reconstruction_errors) == 0:
                raise ValueError("No reconstruction errors available")

            # # Use more robust statistics
            # mean = np.mean(reconstruction_errors_array)
            # std = np.std(reconstruction_errors_array)
            # #max_threshold = mean + 10 * std  # Cap maximum threshold

            # Calculate robust statistics
            median = np.median(reconstruction_errors_array)
            mad = np.median(np.abs(reconstruction_errors_array - median))

            # Calculate base threshold using multiple methods
            percentile_threshold = float(np.percentile(reconstruction_errors_array,
                                                       config.MODEL_CONFIG['thresholds']['trained_percentile']))
            robust_threshold = median + 3 * mad
            mean_std_threshold = np.mean(reconstruction_errors_array) + 2 * np.std(reconstruction_errors_array)

            # Use the most conservative threshold
            trained_threshold = max(
                percentile_threshold,
                robust_threshold,
                mean_std_threshold
            )

            # Set unseen threshold with more conservative multiplier
            unseen_threshold = float(trained_threshold * config.MODEL_CONFIG['thresholds']['unseen_multiplier'] * 1.2)

            # Update thresholds
            self.threshold_trained = trained_threshold
            self.threshold_unseen = unseen_threshold

            # Save thresholds
            threshold_path = os.path.join(config.DATA_STORAGE['models'], 'final', 'thresholds.npy')
            np.save(threshold_path, np.array([trained_threshold, unseen_threshold]))

            self.logger.info(
                f"Calculated thresholds - Trained: {trained_threshold:.6f}, "
                f"Unseen: {unseen_threshold:.6f}"
            )

        except Exception as e:
            self.logger.error(f"Error calculating thresholds: {e}")
            self.logger.error(traceback.format_exc())
            self.threshold_trained = 0.3  # Default fallback value
            self.threshold_unseen = 0.5  # Default fallback value


    def _save_final_process_metrics(self, process_name: str) -> None:
        """Save final metrics for process when monitoring ends"""
        try:
            metrics = self.detection_metrics['process_metrics'].get(process_name)
            if metrics and metrics['samples_processed'] > 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                runtime = datetime.now() - metrics['start_time']

                final_metrics = {
                    'process_name': process_name,
                    'process_type': 'training' if process_name in self.config.TRAINING_PROCESSES else 'testing',
                    'runtime': str(runtime),
                    'samples_processed': metrics['samples_processed'],
                    'anomalies_detected': metrics['anomalies_detected'],
                    'anomaly_rate': metrics['anomalies_detected'] / metrics['samples_processed'],
                    'performance_metrics': self._calculate_performance_metrics(metrics),
                    'average_mse': float(np.mean(metrics['mse_values'])) if metrics['mse_values'] else 0,
                    'average_detection_time': float(np.mean(metrics['detection_times'])) if metrics[
                        'detection_times'] else 0
                }

                # Save to file
                metrics_file = os.path.join(
                    self.storage_dirs['metrics'],
                    f'final_metrics_{process_name.replace("/", "_")}_{timestamp}.json'
                )

                with open(metrics_file, 'w') as f:
                    json.dump(final_metrics, f, indent=4)

                self.logger.info(f"Saved final metrics for {process_name} to {metrics_file}")

        except Exception as e:
            self.logger.error(f"Error saving final metrics for {process_name}: {e}")


    def start_monitoring(self) -> bool:
        """Start real-time monitoring with enhanced process management"""
        try:
            monitoring_processes = []
            self.logger.info("Starting real-time monitoring...")

            # Kill existing Tetragon process
            os.system("sudo pkill tetragon")
            time.sleep(3)

            # Start Tetragon collection
            tetragon_process = Process(
                target=start_tetragon_collection,
                args=(self.stop_event,)
            )
            tetragon_process.start()
            monitoring_processes.append(tetragon_process)

            # Wait for Tetragon initialization
            time.sleep(config.TETRAGON_INIT_WAIT)

            if not verify_tetragon_running():
                self.logger.error("Failed to start Tetragon")
                return False

            # Start monitoring for both training and testing processes
            all_processes = self.config.TRAINING_PROCESSES + self.config.TESTING_PROCESSES

            for process_name in all_processes:
                self.logger.info(f"Starting monitoring for: {process_name}")
                try:
                    p = Process(
                        target=self.monitor_process,
                        args=(process_name,)
                    )
                    p.start()
                    monitoring_processes.append(p)
                except Exception as e:
                    self.logger.error(f"Error starting monitoring process for {process_name}: {e}")

            # Monitor processes and collect metrics
            try:
                while not self.stop_event.is_set():
                    current_time = datetime.now()

                    # Check process health
                    self._check_process_health(monitoring_processes)

                    # Generate periodic reports
                    if (current_time - self.last_report_time) >= self.report_interval:
                        self._generate_comprehensive_report()
                        self.last_report_time = current_time

                    time.sleep(1)

            except KeyboardInterrupt:
                self.logger.info("\nMonitoring interrupted by user")
                self.stop_event.set()

            finally:
                self._cleanup_monitoring(monitoring_processes)

            return True

        except Exception as e:
            self.logger.error(f"Error in monitoring: {e}")
            self.logger.error(traceback.format_exc())
            return False

    # Add this method to the RealTimeDetector class
    def _initialize_process_metrics(self) -> None:
        """Initialize metrics for all processes"""
        try:
            for process_name in self.config.TRAINING_PROCESSES + self.config.TESTING_PROCESSES:
                if process_name not in self.detection_metrics['process_metrics']:
                    self.detection_metrics['process_metrics'][process_name] = {
                        'samples_processed': 0,
                        'anomalies_detected': 0,
                        'mse_values': [],
                        'detection_times': [],
                        'anomaly_rates': [],
                        'total_syscalls': 0,
                        'true_labels': [],
                        'predicted_labels': [],
                        'true_positives': 0,
                        'false_positives': 0,
                        'true_negatives': 0,
                        'false_negatives': 0,
                        'start_time': datetime.now(),
                        'last_update': datetime.now(),
                        'interval_metrics': {
                            'interval_mse': [],
                            'interval_anomalies': [],
                            'interval_times': []
                        }
                    }
                    self.logger.info(f"Initialized metrics for process: {process_name}")

        except Exception as e:
            self.logger.error(f"Error initializing process metrics: {e}")
            self.logger.error(traceback.format_exc())

    def _check_process_health(self, monitoring_processes: List[Process]) -> None:
        """Check health of monitoring processes and restart if needed"""
        try:
            for process in monitoring_processes[1:]:  # Skip Tetragon process
                if not process.is_alive():
                    self.logger.warning(f"Process {process.name} died, restarting...")
                    try:
                        # Terminate cleanly
                        process.terminate()
                        process.join(timeout=self.config.REALTIME_CONFIG['process_timeout'])
                        if process.is_alive():
                            process.kill()
                            process.join(timeout=1)
                    except Exception as e:
                        self.logger.error(f"Error terminating process {process.name}: {e}")

                    # Start new process
                    new_process = Process(
                        target=self.monitor_process,
                        args=(process.name,)
                    )
                    new_process.start()
                    monitoring_processes[monitoring_processes.index(process)] = new_process
                    self.logger.info(f"Restarted process {process.name}")

        except Exception as e:
            self.logger.error(f"Error checking process health: {e}")


    def _cleanup_monitoring(self, monitoring_processes: List[Process]) -> None:
        """Clean up monitoring processes and resources"""
        try:
            self.logger.info("Cleaning up monitoring processes...")

            # Stop all processes
            for process in monitoring_processes:
                try:
                    process.terminate()
                    process.join(timeout=self.config.REALTIME_CONFIG['process_timeout'])
                    if process.is_alive():
                        self.logger.warning(f"Force killing process {process.name}")
                        process.kill()
                except Exception as e:
                    self.logger.error(f"Error terminating process {process.name}: {e}")

            # Stop Tetragon
            stop_tetragon()

            # Save final metrics and reports
            self._save_final_metrics()
            self._generate_final_report()

        except Exception as e:
            self.logger.error(f"Error in cleanup: {e}")
            self.logger.error(traceback.format_exc())


    def _save_final_metrics(self) -> None:
        """Save comprehensive final metrics"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            runtime = datetime.now() - self.start_time

            final_metrics = {
                'runtime': str(runtime),
                'total_processes_monitored': len(self.config.TRAINING_PROCESSES + self.config.TESTING_PROCESSES),
                'training_processes': {},
                'testing_processes': {}
            }

            # Collect metrics for both process types
            for process_type in ['training_processes', 'testing_processes']:
                processes = self.config.TRAINING_PROCESSES if process_type == 'training_processes' \
                    else self.config.TESTING_PROCESSES

                for process_name in processes:
                    metrics = self.detection_metrics['process_metrics'].get(process_name)
                    if metrics and metrics['samples_processed'] > 0:
                        final_metrics[process_type][process_name] = {
                            'samples_processed': metrics['samples_processed'],
                            'anomalies_detected': metrics['anomalies_detected'],
                            'anomaly_rate': metrics['anomalies_detected'] / metrics['samples_processed'],
                            'average_mse': float(np.mean(metrics['mse_values'])) if metrics['mse_values'] else 0,
                            'performance_metrics': self._calculate_performance_metrics(metrics)
                        }

            # Save to file
            metrics_file = os.path.join(
                self.storage_dirs['metrics'],
                f'final_metrics_summary_{timestamp}.json'
            )

            with open(metrics_file, 'w') as f:
                json.dump(final_metrics, f, indent=4)

            self.logger.info(f"Saved final metrics summary to {metrics_file}")

        except Exception as e:
            self.logger.error(f"Error saving final metrics: {e}")


    def _generate_final_report(self) -> None:
        """Generate comprehensive final report"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            runtime = datetime.now() - self.start_time

            print("\n" + "=" * 50)
            print("Final Detection Report")
            print("=" * 50)
            print(f"\nTotal Runtime: {runtime}")

            # Process type summaries
            for process_type in ['Training', 'Testing']:
                processes = self.config.TRAINING_PROCESSES if process_type == 'Training' \
                    else self.config.TESTING_PROCESSES

                print(f"\n{process_type} Processes Summary:")
                print("-" * 30)

                for process_name in processes:
                    metrics = self.detection_metrics['process_metrics'].get(process_name)
                    if metrics and metrics['samples_processed'] > 0:
                        print(f"\n{process_name}:")
                        print(f"Samples Processed: {metrics['samples_processed']}")
                        print(f"Anomalies Detected: {metrics['anomalies_detected']}")

                        if metrics['samples_processed'] > 0:
                            anomaly_rate = metrics['anomalies_detected'] / metrics['samples_processed']
                            print(f"Anomaly Rate: {anomaly_rate:.2%}")

                        if metrics['mse_values']:
                            print(f"Average MSE: {np.mean(metrics['mse_values']):.6f}")

                        perf_metrics = self._calculate_performance_metrics(metrics)
                        if perf_metrics:
                            print("\nPerformance Metrics:")
                            print(f"Accuracy: {perf_metrics['accuracy']:.4f}")
                            print(f"Precision: {perf_metrics['precision']:.4f}")
                            print(f"Recall: {perf_metrics['recall']:.4f}")
                            print(f"F1 Score: {perf_metrics['f1_score']:.4f}")

            # Save final plots
            self._update_plots()

            print("\nResults saved to:")
            print(f"Metrics: {self.storage_dirs['metrics']}")
            print(f"Plots: {self.storage_dirs['plots']}")
            print(f"Alerts: {self.storage_dirs['alerts']}")

        except Exception as e:
            self.logger.error(f"Error generating final report: {e}")
            self.logger.error(traceback.format_exc())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tetragon-bpf', required=True)
    return parser.parse_args()


def main():
    """Main execution function with enhanced error handling and reporting"""
    try:
        args = parse_args()
        os.environ['PROJECT_BASE_DIR'] = os.path.dirname(os.path.abspath(__file__))
        os.environ['TETRAGON_BPF_PATH'] = args.tetragon_bpf
        os.environ['POLICY_FILE'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'syscallpolicy.yaml')
        # Setup logging
        logger = get_logger(__name__)
        logger.info("Starting real-time detection...")

        # Initialize detector
        detector = RealTimeDetector()

        try:
            # Load model
            if not detector.load_model_artifacts():
                logger.error("Failed to load model artifacts")
                return

            # Start monitoring
            if detector.start_monitoring():
                logger.info("Monitoring completed successfully")
            else:
                logger.error("Monitoring failed")

        except KeyboardInterrupt:
            logger.info("Monitoring interrupted by user")
            detector.stop_event.set()

        finally:
            # Cleanup
            stop_tetragon()
            cleanup_files_on_exit()
            gc.collect()

    except Exception as e:
        logger.error(f"Critical error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
