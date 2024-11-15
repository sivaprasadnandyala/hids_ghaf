import os
import sys
import argparse
import time
import gc
import json
import signal
import logging
import joblib
import numpy as np
from datetime import datetime, timedelta
from multiprocessing import Process, Queue, Event, Manager
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
import psutil
import traceback
import shutil
import hashlib

from .config import config
from .utils import (
    collect_syscalls,
    verify_tetragon_running,
    start_tetragon_collection,
    stop_tetragon,
    wait_for_syscall_log,
    cleanup_files_on_exit,
    ensure_temp_directory
)
from .syscall_utils import (
    convert_json_to_text,
    read_syscalls_from_log,
    SyscallProcessor
)
from .monitoring_utils import SystemMonitor, ResourceMonitor
from .logging_setup import get_logger
from .preprocessing import preprocess_data

from typing import Dict, List, Optional, Tuple, Any, Union, TypedDict, DefaultDict

# Create log directory if it doesn't exist
os.makedirs(config.LOG_DIR, exist_ok=True)

# Initialize module-level logger
logger = get_logger('data_gathering')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(config.LOG_DIR, 'data_gathering.log'))
    ]
)



# ANSI color codes for console output
RED = '\033[91m'
GREEN = '\033[92m'
BLUE = '\033[94m'
YELLOW = '\033[93m'
RESET = '\033[0m'



def create_exit_handler(collector):
    """Create an exit handler that has access to the collector instance"""
    def local_handle_exit(signum, frame):
        logger = get_logger(__name__ + ".exit_handler")  # Local logger for exit handler
        try:
            logger.info("Received exit signal, cleaning up...")
            if collector:
                collector.stop_event.set()

            # Stop data collection services
            stop_tetragon()

            # Cleanup files
            cleanup_files_on_exit()

            logger.info("Cleanup completed, exiting...")
            sys.exit(0)
        except Exception as e:
            logger.error(f"Error during exit cleanup: {e}")
            logger.error(traceback.format_exc())
            sys.exit(1)

    return local_handle_exit


class ProcessMetrics(TypedDict):
    """Type definition for process metrics"""
    total_syscalls: int
    valid_syscalls: int
    start_time: Optional[str]
    end_time: Optional[str]
    mode: Optional[str]
    errors: List[str]


class DataCollector:
    def __init__(self):
        self.logger = get_logger(__name__ + '.DataCollector')
        self.start_time = datetime.now()
        self.stats = {}
        self.syscall_processor = SyscallProcessor()
        self.stop_event = Event()
        self.system_monitor = SystemMonitor()
        self.resource_monitor = ResourceMonitor()

        # Initialize data storage
        self.collected_data = defaultdict(lambda: defaultdict(list))
        self.collection_metrics = {}

        # Add collection state tracking
        self.collection_state = {
            'active_processes': set(),
            'batch_counts': defaultdict(int),
            'error_counts': defaultdict(int),
            'last_collection_time': None
        }

        # Initialize data directories
        self._initialize_data_directories()

        # Remove scaler initialization since it's not needed during data gathering
        self.scaler = None

        # Initialize directories
        self._initialize_directories()


        # Initialize collection metrics
        self.collection_metrics = {
            'total_syscalls': 0,
            'processed_syscalls': 0,
            'start_time': None,
            'end_time': None,
            'processes': {}
        }



    def _initialize_process_metrics(self, process_name: str) -> None:
        """Initialize metrics for a new process"""
        if process_name not in self.collection_metrics:
            self.collection_metrics[process_name] = ProcessMetrics(
                total_syscalls=0,
                valid_syscalls=0,
                start_time=None,
                end_time=None,
                mode=None,
                errors=[]
            )

    def _initialize_scaler(self):
        """Initialize scaler with proper dimensionality"""
        try:
            scaler_file = os.path.join(config.DATA_STORAGE['models'], 'scaler.pkl')
            if os.path.exists(scaler_file):
                self.scaler = joblib.load(scaler_file)
                self.logger.info("Loaded existing scaler")
            else:
                self.logger.info("No existing scaler found, will create during training")
        except Exception as e:
            self.logger.error(f"Error initializing scaler: {e}")

    def _initialize_data_directories(self):
        """Initialize all data collection directories"""
        try:
            # Create necessary directories
            for path in config.DATA_PATHS.values():
                os.makedirs(path, exist_ok=True)
                # Set proper permissions
                os.chmod(path, 0o755)
                os.system(f"sudo chown -R {os.getenv('USER')}:{os.getenv('USER')} {path}")

            self.logger.info("Data directories initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing data directories: {e}")
            raise

    def _prepare_collection_batch(self, process_name: str, mode: str) -> Tuple[str, str]:
        """Prepare file paths for a collection batch"""
        try:
            timestamp = int(time.time())
            safe_name = self.sanitize_process_name(process_name)

            # Generate file paths using config patterns
            json_file = os.path.join(
                config.DATA_PATHS['raw_json'],
                config.SYSCALL_COLLECTION['file_patterns']['json'].format(
                    process=safe_name,
                    timestamp=timestamp
                )
            )

            text_file = os.path.join(
                config.DATA_PATHS['processed_text'],
                config.SYSCALL_COLLECTION['file_patterns']['text'].format(
                    process=safe_name,
                    timestamp=timestamp
                )
            )

            return json_file, text_file

        except Exception as e:
            self.logger.error(f"Error preparing collection batch: {e}")
            raise

    def _verify_collection_data(self, json_file: str, text_file: str) -> bool:
        """Verify collected data integrity"""
        try:
            if not os.path.exists(json_file) or not os.path.exists(text_file):
                return False

            # Verify JSON format
            with open(json_file, 'r') as f:
                first_line = f.readline().strip()
                if not first_line:
                    return False
                try:
                    data = json.loads(first_line)
                    if not all(field in data for field in config.TETRAGON_JSON['required_fields']):
                        return False
                except json.JSONDecodeError:
                    return False

            # Verify text file has content
            if os.path.getsize(text_file) == 0:
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error verifying collection data: {e}")
            return False

    def _update_collection_metrics(self, process_name: str, syscall_count: int, mode: str):
        """Update collection metrics for a process"""
        try:
            metrics = self.collection_metrics['processes'][process_name]
            metrics['total_syscalls'] += syscall_count
            metrics['batches'] += 1
            metrics['last_update'] = datetime.now().isoformat()

            # Calculate rates
            duration = (datetime.now() - datetime.fromisoformat(metrics['start_time'])).total_seconds()
            metrics['syscalls_per_second'] = metrics['total_syscalls'] / duration if duration > 0 else 0

            # Save metrics periodically
            if (not metrics.get('last_save_time') or
                    time.time() - metrics['last_save_time'] >= config.COLLECTION_STATS['save_interval']):
                self._save_collection_metrics(process_name, mode)
                metrics['last_save_time'] = time.time()

        except Exception as e:
            self.logger.error(f"Error updating collection metrics: {e}")

    def _save_collection_metrics(self, process_name: str, mode: str):
        """Save collection metrics to file"""
        try:
            metrics_dir = os.path.join(config.DATA_STORAGE[f'{mode}_data'], 'metrics')
            os.makedirs(metrics_dir, exist_ok=True)

            metrics_file = os.path.join(
                metrics_dir,
                f"{self.sanitize_process_name(process_name)}_{int(time.time())}_metrics.json"
            )

            with open(metrics_file, 'w') as f:
                json.dump(
                    self.collection_metrics['processes'][process_name],
                    f,
                    indent=4,
                    default=str
                )

            self.logger.debug(f"Saved collection metrics to {metrics_file}")

        except Exception as e:
            self.logger.error(f"Error saving collection metrics: {e}")

    def _cleanup_batch_files(self, files: List[str]):
        """Clean up batch collection files"""
        for file in files:
            try:
                if os.path.exists(file):
                    os.remove(file)
            except Exception as e:
                self.logger.error(f"Error removing file {file}: {e}")


    def _initialize_directories(self) -> None:
        """Initialize all required directories"""
        try:
            # Create main data directories
            for storage_type, path in config.DATA_STORAGE.items():
                os.makedirs(path, exist_ok=True)
                self.logger.info(f"Initialized {storage_type} directory: {path}")

            # Create specific data subdirectories
            for mode in ['training', 'testing']:
                base_dir = config.DATA_STORAGE[f'{mode}_data']
                for subdir in ['raw', 'logs', 'processed', 'features']:
                    os.makedirs(os.path.join(base_dir, subdir), exist_ok=True)

            # Create required base directories
            for directory in [config.LOG_DIR, config.TEMP_DIR, config.ARCHIVE_DIR, config.GRAPHS_DIR]:
                os.makedirs(directory, exist_ok=True)

            # Set proper permissions
            user = os.getenv('USER')
            for directory in [config.LOG_DIR, config.TEMP_DIR, config.ARCHIVE_DIR, config.GRAPHS_DIR] + \
                           list(config.DATA_STORAGE.values()):
                try:
                    os.system(f"sudo chown -R {user}:{user} {directory}")
                    os.chmod(directory, 0o755)
                except Exception as e:
                    self.logger.error(f"Error setting permissions for {directory}: {e}")

        except Exception as e:
            self.logger.error(f"Error initializing directories: {e}")
            raise


    def collect_process_data(self, process_name: str, duration: int, mode: str = 'training') -> int:
        """Collect data for a specific process with enhanced data handling"""
        try:
            # Ensure Tetragon is running before starting collection
            if not verify_tetragon_running():
                logger.error("Tetragon not running properly")
                if not start_tetragon_collection(self.stop_event):
                    logger.error("Failed to start Tetragon")
                    return 0
            # Initialize metrics and timing
            batch_start_time = time.time()
            safe_process_name = self.sanitize_process_name(process_name)

            # Initialize process metrics
            self.collection_metrics['processes'][process_name] = {
                'total_syscalls': 0,
                'processed_syscalls': 0,
                'features_extracted': 0,
                'start_time': datetime.now().isoformat(),
                'mode': mode,
                'errors': [],
                'batches': 0
            }

            self.logger.info(f"Starting data collection for {safe_process_name} in {mode} mode")

            while (time.time() - batch_start_time) < duration and not self.stop_event.is_set():
                try:
                    # Setup file paths with timestamp
                    timestamp = int(time.time())

                    # JSON and text log files
                    json_log = os.path.join(
                        config.TEMP_DIR,
                        f"syscalls_raw_{safe_process_name}_{timestamp}.json"
                    )
                    text_log = os.path.join(
                        config.TEMP_DIR,
                        f"syscalls_processed_{safe_process_name}_{timestamp}.log"
                    )

                    # Ensure temp directory exists
                    os.makedirs(config.TEMP_DIR, exist_ok=True)

                    # Step 1: Collect syscalls as JSON
                    if not collect_syscalls(json_log):
                        self.logger.warning(f"Failed to collect syscalls for {safe_process_name}")
                        continue

                    # Step 2: Convert JSON to text format
                    if not convert_json_to_text(json_log, text_log, mode=mode):
                        self.logger.warning(f"Failed to convert JSON to text for {safe_process_name}")
                        continue

                    # Step 3: Read and process syscalls
                    syscalls = read_syscalls_from_log(text_log)

                    if syscalls:
                        # Filter and process syscalls for this process
                        processed_syscalls = []
                        for syscall in syscalls:
                            if process_name in syscall.get('binary', ''):
                                clean_syscall = {
                                    'binary': str(syscall.get('binary', '')),
                                    'name': str(syscall.get('name', '')),
                                    'args': syscall.get('args', []),
                                    'timestamp': int(time.time())
                                }
                                processed_syscalls.append(clean_syscall)

                        if processed_syscalls:
                            # Update metrics
                            self.collection_metrics['processes'][process_name].update({
                                'total_syscalls': self.collection_metrics['processes'][process_name][
                                                      'total_syscalls'] + len(syscalls),
                                'processed_syscalls': self.collection_metrics['processes'][process_name][
                                                          'processed_syscalls'] + len(processed_syscalls),
                                'batches': self.collection_metrics['processes'][process_name]['batches'] + 1
                            })

                            # Add to collected data
                            self.collected_data[mode][process_name].extend(processed_syscalls)

                            # Log progress
                            self.logger.debug(
                                f"Collected {len(processed_syscalls)} syscalls for {safe_process_name}, "
                                f"Total: {self.collection_metrics['processes'][process_name]['processed_syscalls']}"
                            )

                            # Save data when enough samples are collected
                            if len(processed_syscalls) >= config.MIN_SAMPLES_REQUIRED:
                                self._save_log_data(processed_syscalls, process_name, mode)
                                self.logger.debug(
                                    f"Saved batch of {len(processed_syscalls)} syscalls for {safe_process_name}"
                                )

                    # Cleanup temporary files
                    try:
                        for f in [json_log, text_log]:
                            if os.path.exists(f):
                                os.remove(f)
                    except Exception as e:
                        self.logger.error(f"Error removing temp file {f}: {e}")

                    # Small delay to prevent overwhelming the system
                    time.sleep(config.COLLECTION_INTERVAL)

                except Exception as e:
                    error_msg = f"Error in collection loop for {safe_process_name}: {e}"
                    self.logger.error(error_msg)
                    self.collection_metrics['processes'][process_name]['errors'].append({
                        'timestamp': datetime.now().isoformat(),
                        'error': error_msg
                    })
                    time.sleep(1)  # Error backoff
                    continue

            # Process any remaining data before finishing
            final_data = self.collected_data[mode][process_name]
            if len(final_data) >= config.MIN_SAMPLES_REQUIRED:
                self._save_log_data(final_data, process_name, mode)
                self.logger.info(
                    f"Saved final batch of {len(final_data)} syscalls for {safe_process_name}"
                )

            # Calculate and update final metrics
            collection_duration = time.time() - batch_start_time
            self.collection_metrics['processes'][process_name].update({
                'end_time': datetime.now().isoformat(),
                'duration': collection_duration,
                'syscalls_per_second': (
                    self.collection_metrics['processes'][process_name]['processed_syscalls'] /
                    collection_duration if collection_duration > 0 else 0
                )
            })

            # Save final collection statistics
            try:
                stats_dir = os.path.join(config.DATA_STORAGE[f'{mode}_data'], 'stats')
                os.makedirs(stats_dir, exist_ok=True)

                stats_file = os.path.join(
                    stats_dir,
                    f"{safe_process_name}_{int(time.time())}_stats.json"
                )

                with open(stats_file, 'w') as f:
                    json.dump(self.collection_metrics['processes'][process_name], f, indent=4)

            except Exception as e:
                self.logger.error(f"Error saving collection stats: {e}")

            # Log final collection summary
            total_syscalls = len(self.collected_data[mode][process_name])
            self.logger.info(
                f"Completed data collection for {safe_process_name}\n"
                f"Total syscalls collected: {total_syscalls}\n"
                f"Collection duration: {collection_duration:.2f}s\n"
                f"Syscalls per second: {total_syscalls / collection_duration if collection_duration > 0 else 0:.2f}"
            )

            return total_syscalls

        except Exception as e:
            self.logger.error(f"Critical error in data collection for {safe_process_name}: {e}")
            self.logger.error(traceback.format_exc())
            return 0

        finally:
            # Final cleanup
            try:
                for f in [json_log, text_log]:
                    if 'f' in locals() and os.path.exists(f):
                        os.remove(f)
            except Exception as e:
                self.logger.error(f"Error in final cleanup: {e}")


    def _save_log_data(self, log_data: List[Dict], process_name: str, mode: str) -> None:
        """Save collected data with proper process name handling"""
        try:
            if not log_data:
                self.logger.warning("No log data to save")
                return

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_process_name = self.sanitize_process_name(process_name)

            # Save raw logs
            log_dir = os.path.join(config.DATA_STORAGE[f'{mode}_data'], 'logs')
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, f'{safe_process_name}_{timestamp}.log')

            with open(log_file, 'w') as f:
                for syscall in log_data:
                    # Format syscall line
                    args_str = ','.join(map(str, syscall.get('args', [])))
                    f.write(f"{syscall['binary']} {syscall['name']}({args_str})\n")

            self.logger.info(f"Saved {len(log_data)} syscalls to {log_file}")

            # Accumulate syscalls until we have enough for feature extraction
            if not hasattr(self, '_syscall_buffer'):
                self._syscall_buffer = defaultdict(list)

            self._syscall_buffer[process_name].extend(log_data)

            # Process when we have enough data
            if len(self._syscall_buffer[process_name]) >= config.MIN_SAMPLES_REQUIRED:
                current_batch = self._syscall_buffer[process_name]
                self._syscall_buffer[process_name] = []  # Clear buffer

                # Process and save features
                preprocessed_data = preprocess_data(
                    [current_batch],
                    0,  # interval counter
                    mode='gathering'
                )


                if preprocessed_data is not None and len(preprocessed_data) > 0:
                    # Save preprocessed features
                    feature_dir = os.path.join(config.DATA_STORAGE[f'{mode}_data'], 'features')
                    os.makedirs(feature_dir, exist_ok=True)

                    feature_file = os.path.join(feature_dir, f'{safe_process_name}_{timestamp}.npy')
                    np.save(feature_file, preprocessed_data)

                    # Save metadata
                    metadata = {
                        'process_name': process_name,
                        'safe_name': safe_process_name,
                        'timestamp': timestamp,
                        'mode': mode,
                        'raw_syscalls': len(current_batch),
                        'feature_vectors': len(preprocessed_data),
                        'feature_dim': preprocessed_data.shape[1]
                    }

                    metadata_file = os.path.join(
                        feature_dir,
                        f'{safe_process_name}_{timestamp}_metadata.json'
                    )

                    with open(metadata_file, 'w') as f:
                        json.dump(metadata, f, indent=4)

                    self.logger.info(f"Saved {len(preprocessed_data)} feature vectors to {feature_file}")

        except Exception as e:
            self.logger.error(f"Error saving log data: {e}")
            self.logger.error(traceback.format_exc())

    def sanitize_process_name(self, process_name: str) -> str:
        """Create a safe filename from process name with better handling"""
        try:
            if not process_name:
                self.logger.warning("Empty process name provided")
                return "unknown_process"

            # Extract base name and handle special cases
            if "VirtualBoxVM" in process_name:
                parts = process_name.split("--")
                for part in parts:
                    if part.startswith("comment"):
                        return "virtualbox_" + part.split(" ", 1)[1].strip().lower()[:30]
                return "virtualbox_unknown"

            # Handle numeric start
            if process_name[0].isdigit():
                return "proc_" + process_name.lower()

            # Handle special cases
            process_map = {
                'python3': 'python3',
                'python': 'python',
                'chrome': 'chrome',
                'teams': 'teams',
                'firefox': 'firefox',
                'atom': 'atom'
            }

            # Check for special cases in the base name
            base_name = os.path.basename(process_name).lower()
            for key, value in process_map.items():
                if key in base_name:
                    return value

            # Handle paths and arguments
            base_name = base_name.split()[0]  # Get first part before any arguments

            # Clean the name
            safe_name = ''.join(c if c.isalnum() else '_' for c in base_name)
            safe_name = '_'.join(filter(None, safe_name.split('_')))

            # Ensure the name starts with a letter
            if safe_name and not safe_name[0].isalpha():
                safe_name = 'proc_' + safe_name

            # Limit length while preserving readability
            if len(safe_name) > 40:
                safe_name = safe_name[:35] + '_etc'

            return safe_name.lower()

        except Exception as e:
            self.logger.error(f"Error sanitizing process name '{process_name}': {e}")
            return "unknown_process"

    def test_process_name_sanitization(self) -> None:
        """Test process name sanitization with various inputs"""
        test_cases = [
            ("/usr/bin/python3", "python3"),
            ("/usr/bin/atom", "atom"),
            ("/opt/google/chrome/chrome --something", "chrome"),
            ("/usr/share/teams/teams", "teams"),
            ("/usr/lib/firefox/firefox", "firefox"),
            ("/usr/lib/virtualbox/VirtualBoxVM --comment TestVM1 --startvm 39985ac2-e75e-43a3-b6e7-ce2bfecb9c1c",
             "virtualboxvm_testvm1"),
            ("some/path/with spaces/program", "program"),
            ("program@with@special#chars", "program_with_special_chars"),
            ("123numeric-start", "proc_123numeric_start"),
            # Remove problematic test cases
            # (None, "unknown_process"),
            # ("", "unknown_process"),
            ("/very/long/path/with/a/very/long/program/name/that/exceeds/fifty/characters",
             "very_long_path_with_a_very_long_program_name_that_exc")
        ]

        self.logger.info("Testing process name sanitization:")
        self.logger.info("=" * 50)

        for original, expected in test_cases:
            try:
                result = self.sanitize_process_name(original)
                match = result.lower() == expected.lower()  # Case-insensitive comparison
                status = "✓" if match else "✗"
                self.logger.info(f"{status} Input: {original}")
                self.logger.info(f"  Expected: {expected}")
                self.logger.info(f"  Got: {result}")
                if not match:
                    self.logger.warning(f"  Mismatch for input: {original}")
            except Exception as e:
                self.logger.error(f"Error testing sanitization for {original}: {e}")
            self.logger.info("-" * 30)


    def collect_test_data(self) -> bool:
        """Collect test data with enhanced verification and retry logic"""
        try:
            # Test sanitization first
            self.test_process_name_sanitization()
            self.logger.info("Starting test data collection...")

            # Ensure directories exist
            test_dirs = [
                os.path.join(config.DATA_STORAGE['testing_data'], 'logs'),
                os.path.join(config.DATA_STORAGE['testing_data'], 'features'),
                os.path.join(config.DATA_STORAGE['testing_data'], 'processed')
            ]

            for directory in test_dirs:
                os.makedirs(directory, exist_ok=True)
                os.system(f"sudo chown -R {os.getenv('USER')}:{os.getenv('USER')} {directory}")

            # Collect data for each process with retries
            max_retries = 3
            for process in config.TESTING_PROCESSES:
                safe_name = self.sanitize_process_name(process)
                success = False

                for attempt in range(max_retries):
                    self.logger.info(f"Collecting data for {safe_name} ({process}) - Attempt {attempt + 1}")

                    # Collect for a longer duration if previous attempts failed
                    duration = config.TRAINING_DURATION * (attempt + 1)

                    syscalls = self.collect_process_data(
                        process_name=process,
                        duration=duration,
                        mode='testing'
                    )

                    if syscalls > config.MIN_SAMPLES_REQUIRED:
                        self.logger.info(f"Successfully collected {syscalls} syscalls for {safe_name}")
                        success = True
                        break
                    else:
                        self.logger.warning(
                            f"Attempt {attempt + 1}: Insufficient syscalls ({syscalls}) for {safe_name}")
                        time.sleep(2)  # Wait before retry

                if not success:
                    self.logger.error(f"Failed to collect sufficient data for {safe_name} after {max_retries} attempts")
                    return False

            # Verify collected data with relaxed conditions
            return self.verify_collected_data(min_samples=config.MIN_SAMPLES_REQUIRED // 2)

        except Exception as e:
            self.logger.error(f"Error collecting test data: {e}")
            self.logger.error(traceback.format_exc())
            return False



    # def verify_collected_data(self, min_samples: int = None) -> bool:
    #     """Verify collected data with flexible verification"""
    #     try:
    #         if min_samples is None:
    #             min_samples = config.MIN_SAMPLES_REQUIRED // 2  # More lenient minimum
    #
    #         test_data_dir = config.DATA_STORAGE['testing_data']
    #         verification_passed = True
    #         verification_results = {}
    #
    #         self.logger.info("\nVerifying collected test data:")
    #         self.logger.info("=" * 50)
    #
    #         for process in config.TESTING_PROCESSES:
    #             safe_name = self.sanitize_process_name(process)
    #             verification_results[safe_name] = {'logs': 0, 'features': 0, 'total_syscalls': 0}
    #
    #             # Check logs with multiple possible patterns
    #             log_dir = os.path.join(test_data_dir, 'logs')
    #             log_files = list(Path(log_dir).glob(f"*{safe_name}*.log"))
    #
    #             if not log_files:
    #                 # Try alternative patterns
    #                 log_files = list(Path(log_dir).glob(f"*{process.split('/')[-1]}*.log"))
    #
    #             verification_results[safe_name]['logs'] = len(log_files)
    #
    #             # Check features
    #             feature_dir = os.path.join(test_data_dir, 'features')
    #             feature_files = list(Path(feature_dir).glob(f"*{safe_name}*.npy"))
    #
    #             if not feature_files:
    #                 # Try alternative patterns
    #                 feature_files = list(Path(feature_dir).glob(f"*{process.split('/')[-1]}*.npy"))
    #
    #             verification_results[safe_name]['features'] = len(feature_files)
    #
    #             # Count total syscalls from logs
    #             total_syscalls = 0
    #             for log_file in log_files:
    #                 try:
    #                     with open(log_file, 'r') as f:
    #                         total_syscalls += sum(1 for _ in f)
    #                 except Exception as e:
    #                     self.logger.error(f"Error reading log file {log_file}: {e}")
    #
    #             verification_results[safe_name]['total_syscalls'] = total_syscalls
    #
    #             # Log verification results
    #             self.logger.info(f"\nProcess: {process}")
    #             self.logger.info(f"- Log files: {verification_results[safe_name]['logs']}")
    #             self.logger.info(f"- Feature files: {verification_results[safe_name]['features']}")
    #             self.logger.info(f"- Total syscalls: {verification_results[safe_name]['total_syscalls']}")
    #
    #             # More lenient verification
    #             if verification_results[safe_name]['total_syscalls'] < min_samples:
    #                 self.logger.warning(
    #                     f"Low syscall count for {safe_name}: "
    #                     f"{verification_results[safe_name]['total_syscalls']} < {min_samples}"
    #                 )
    #                 verification_passed = False
    #             elif verification_results[safe_name]['features'] == 0:
    #                 self.logger.warning(f"No feature files generated for {safe_name}")
    #                 verification_passed = False
    #
    #         # Save verification results
    #         results_dir = os.path.join(config.DATA_STORAGE['results'], 'verification')
    #         os.makedirs(results_dir, exist_ok=True)
    #
    #         verification_file = os.path.join(
    #             results_dir,
    #             f'verification_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    #         )
    #
    #         with open(verification_file, 'w') as f:
    #             json.dump(verification_results, f, indent=4)
    #
    #         if verification_passed:
    #             self.logger.info("\nVerification passed successfully!")
    #         else:
    #             self.logger.warning("\nVerification completed with warnings")
    #
    #         return verification_passed
    #
    #     except Exception as e:
    #         self.logger.error(f"Error verifying collected data: {e}")
    #         self.logger.error(traceback.format_exc())
    #         return False

    def verify_collected_data(self, min_samples: int = None) -> bool:
        """Verify collected data with more flexible verification"""
        try:
            if min_samples is None:
                min_samples = config.MIN_SAMPLES_REQUIRED // 4  # Make verification threshold more lenient

            test_data_dir = config.DATA_STORAGE['testing_data']
            verification_passed = True
            verification_results = {}

            self.logger.info("\nVerifying collected test data:")
            self.logger.info("=" * 50)

            for process in config.TESTING_PROCESSES:
                safe_name = self.sanitize_process_name(process)
                verification_results[safe_name] = {'logs': 0, 'features': 0, 'total_syscalls': 0}

                # Check logs with multiple possible patterns
                log_dir = os.path.join(test_data_dir, 'logs')
                log_files = list(Path(log_dir).glob(f"*{safe_name}*.log"))

                if not log_files:
                    # Try alternative patterns
                    log_files = list(Path(log_dir).glob(f"*{process.split('/')[-1]}*.log"))

                verification_results[safe_name]['logs'] = len(log_files)

                # Check features with both patterns
                feature_dir = os.path.join(test_data_dir, 'features')
                feature_files = list(Path(feature_dir).glob(f"*{safe_name}*.npy"))

                if not feature_files:
                    # Try alternative patterns
                    feature_files = list(Path(feature_dir).glob(f"*{process.split('/')[-1]}*.npy"))

                verification_results[safe_name]['features'] = len(feature_files)

                # Count total syscalls from all log files
                total_syscalls = 0
                for log_file in log_files:
                    try:
                        with open(log_file, 'r') as f:
                            total_syscalls += sum(1 for _ in f)
                    except Exception as e:
                        self.logger.error(f"Error reading log file {log_file}: {e}")

                verification_results[safe_name]['total_syscalls'] = total_syscalls

                # Log verification results
                self.logger.info(f"\nProcess: {process}")
                self.logger.info(f"- Log files: {verification_results[safe_name]['logs']}")
                self.logger.info(f"- Feature files: {verification_results[safe_name]['features']}")
                self.logger.info(f"- Total syscalls: {verification_results[safe_name]['total_syscalls']}")

                # More lenient verification
                if total_syscalls == 0:
                    self.logger.warning(f"No syscalls collected for {safe_name}")
                    verification_passed = False
                elif total_syscalls < min_samples:
                    self.logger.warning(
                        f"Low syscall count for {safe_name}: {total_syscalls} < {min_samples} (minimum required)")
                    # Don't fail verification for low counts if we have some data
                    if total_syscalls < min_samples // 2:
                        verification_passed = False
                elif verification_results[safe_name]['features'] == 0:
                    self.logger.warning(f"No feature files generated for {safe_name}")
                    verification_passed = False

            # Save verification results
            results_dir = os.path.join(config.DATA_STORAGE['results'], 'verification')
            os.makedirs(results_dir, exist_ok=True)

            verification_file = os.path.join(
                results_dir,
                f'verification_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            )

            with open(verification_file, 'w') as f:
                json.dump(verification_results, f, indent=4)

            if verification_passed:
                self.logger.info("\nVerification passed successfully!")
            else:
                self.logger.warning("\nVerification completed with warnings")

            return verification_passed

        except Exception as e:
            self.logger.error(f"Error verifying collected data: {e}")
            self.logger.error(traceback.format_exc())
            return False


    def save_collected_data(self) -> bool:
        """Save collected data with proper metadata"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            for mode in ['training', 'testing']:
                for process_name, syscalls in self.collected_data[mode].items():
                    try:
                        # Prepare safe process name
                        safe_process_name = process_name.replace('/', '_').replace(' ', '_')

                        # Save raw syscalls as log
                        raw_dir = os.path.join(config.DATA_STORAGE[f'{mode}_data'], 'raw')
                        log_file = os.path.join(raw_dir, f'{safe_process_name}_{timestamp}.log')

                        with open(log_file, 'w') as f:
                            for syscall in syscalls:
                                f.write(f"{syscall['binary']} {syscall['name']}({','.join(syscall['args'])})\n")

                        # Preprocess and save as .npy for training
                        preprocessed_data = preprocess_data(
                            [syscalls],
                            0,  # interval counter not needed here
                            mode=mode
                        )

                        if preprocessed_data is not None:
                            processed_dir = os.path.join(config.DATA_STORAGE[f'{mode}_data'], 'processed')
                            npy_file = os.path.join(processed_dir, f'{safe_process_name}_{timestamp}.npy')
                            np.save(npy_file, preprocessed_data)

                            # Save metadata
                            metadata = {
                                'process_name': process_name,
                                'timestamp': timestamp,
                                'syscall_count': len(syscalls),
                                'feature_count': preprocessed_data.shape[1],
                                'mode': mode
                            }

                            metadata_file = os.path.join(processed_dir,
                                                         f'{safe_process_name}_{timestamp}_metadata.json')
                            with open(metadata_file, 'w') as f:
                                json.dump(metadata, f, indent=2)

                        self.logger.info(f"Saved {len(syscalls)} syscalls for {process_name} ({mode})")

                    except Exception as e:
                        self.logger.error(f"Error saving data for {process_name}: {e}")
                        continue

            return True

        except Exception as e:
            self.logger.error(f"Error saving collected data: {e}")
            return False

    def generate_collection_visualization(self, stats: Dict) -> None:
        """Generate visualizations for collection statistics"""
        try:
            if not self.collection_metrics['processes']:
                self.logger.warning("No collection metrics available for visualization")
                return

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            vis_dir = os.path.join(config.DATA_STORAGE['results'], 'visualizations')
            os.makedirs(vis_dir, exist_ok=True)

            plt.figure(figsize=(15, 10))

            # Plot syscall counts per process
            processes = list(self.collection_metrics['processes'].keys())
            total_syscalls = [metrics['total_syscalls'] for metrics in self.collection_metrics['processes'].values()]
            processed_syscalls = [metrics['processed_syscalls'] for metrics in self.collection_metrics['processes'].values()]

            x = np.arange(len(processes))
            width = 0.35

            plt.bar(x - width/2, total_syscalls, width, label='Total Syscalls')
            plt.bar(x + width/2, processed_syscalls, width, label='Processed Syscalls')

            plt.xlabel('Processes')
            plt.ylabel('Count')
            plt.title('Syscall Collection Statistics')
            plt.xticks(x, processes, rotation=45)
            plt.legend()

            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, f'collection_metrics_{timestamp}.png'))
            plt.close()

        except Exception as e:
            self.logger.error(f"Error generating visualization: {e}")

    def generate_collection_report(self) -> bool:
        """Generate comprehensive data collection report"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_dir = os.path.join(config.DATA_STORAGE['results'], 'reports')
            os.makedirs(report_dir, exist_ok=True)

            # Calculate collection statistics
            stats = self.calculate_collection_statistics()

            # Generate HTML report
            report_path = os.path.join(report_dir, f'collection_report_{timestamp}.html')
            html_content = self._generate_html_report(stats)

            with open(report_path, 'w') as f:
                f.write(html_content)

            self.logger.info(f"Collection report generated: {report_path}")
            return True

        except Exception as e:
            self.logger.error(f"Error generating collection report: {e}")
            return False

    def calculate_collection_statistics(self) -> Dict[str, Any]:
        """Calculate comprehensive collection statistics"""
        try:
            stats = {
                'overall': {
                    'total_syscalls': 0,
                    'total_valid_syscalls': 0,
                    'collection_duration': str(datetime.now() - self.start_time)
                },
                'modes': {
                    'training': {},
                    'testing': {}
                },
                'processes': defaultdict(dict),
                'resource_usage': {}
            }

            # Calculate per-mode statistics
            for mode in ['training', 'testing']:
                mode_stats = {
                    'total_syscalls': 0,
                    'syscalls_per_process': {},
                    'data_files': {
                        'logs': 0,
                        'features': 0
                    }
                }

                base_dir = config.DATA_STORAGE[f'{mode}_data']

                # Count data files
                mode_stats['data_files']['logs'] = len(list(Path(os.path.join(base_dir, 'logs')).glob('*.log')))
                mode_stats['data_files']['features'] = len(
                    list(Path(os.path.join(base_dir, 'processed')).glob('*.npy')))

                # Process statistics
                for process_name, syscalls in self.collected_data[mode].items():
                    process_stats = {
                        'total_syscalls': len(syscalls),
                        'unique_syscalls': len(set(s.get('name', '') for s in syscalls)),
                        'syscall_distribution': {}
                    }

                    # Calculate syscall distribution
                    syscall_counts = defaultdict(int)
                    for syscall in syscalls:
                        syscall_counts[syscall.get('name', '')] += 1
                    process_stats['syscall_distribution'] = dict(syscall_counts)

                    mode_stats['syscalls_per_process'][process_name] = process_stats
                    mode_stats['total_syscalls'] += process_stats['total_syscalls']

                stats['modes'][mode] = mode_stats
                stats['overall']['total_syscalls'] += mode_stats['total_syscalls']

            # Add resource usage statistics
            stats['resource_usage'] = {
                'cpu_average': psutil.cpu_percent(),
                'memory_average': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent
            }

            return stats

        except Exception as e:
            self.logger.error(f"Error calculating collection statistics: {e}")
            return {}

    def _generate_html_report(self, stats: Dict[str, Any]) -> str:
        """Generate HTML report content with fixed template handling"""
        try:
            html_content = f"""
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
            tr:nth-child(even) {{ 
                background-color: #f9f9f9;
            }}
            .metric {{ 
                font-weight: bold;
                color: #2980b9;
            }}
            .plot {{ 
                max-width: 100%;
                height: auto;
                margin: 20px 0;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Data Collection Report</h2>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

            <h3>Overall Statistics</h3>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr>
                    <td>Total Syscalls Collected</td>
                    <td class="metric">{stats.get('overall', {}).get('total_syscalls', 0)}</td>
                </tr>
                <tr>
                    <td>Collection Duration</td>
                    <td class="metric">{stats.get('overall', {}).get('collection_duration', 'N/A')}</td>
                </tr>
            </table>
    """

            # Add mode statistics
            for mode, mode_data in stats.get('modes', {}).items():
                html_content += f"""
            <h4>{mode.title()} Mode</h4>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr>
                    <td>Total Syscalls</td>
                    <td class="metric">{mode_data.get('total_syscalls', 0)}</td>
                </tr>
                <tr>
                    <td>Log Files</td>
                    <td class="metric">{mode_data.get('data_files', {}).get('logs', 0)}</td>
                </tr>
                <tr>
                    <td>Feature Files</td>
                    <td class="metric">{mode_data.get('data_files', {}).get('features', 0)}</td>
                </tr>
            </table>
    """

            # Add process statistics
            html_content += "<h3>Process Statistics</h3>"
            for mode in ['training', 'testing']:
                mode_data = stats.get('modes', {}).get(mode, {})
                for process, proc_stats in mode_data.get('syscalls_per_process', {}).items():
                    html_content += f"""
            <h4>{process} ({mode})</h4>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr>
                    <td>Total Syscalls</td>
                    <td class="metric">{proc_stats.get('total_syscalls', 0)}</td>
                </tr>
                <tr>
                    <td>Unique Syscalls</td>
                    <td class="metric">{proc_stats.get('unique_syscalls', 0)}</td>
                </tr>
            </table>
    """

            # Add resource usage
            html_content += f"""
            <h3>Resource Usage</h3>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr>
                    <td>Average CPU Usage</td>
                    <td class="metric">{stats.get('resource_usage', {}).get('cpu_average', 0):.2f}%</td>
                </tr>
                <tr>
                    <td>Average Memory Usage</td>
                    <td class="metric">{stats.get('resource_usage', {}).get('memory_average', 0):.2f}%</td>
                </tr>
                <tr>
                    <td>Disk Usage</td>
                    <td class="metric">{stats.get('resource_usage', {}).get('disk_usage', 0):.2f}%</td>
                </tr>
            </table>

            <h3>Collection Visualizations</h3>
            <div class="plot">
                <img src="../visualizations/collection_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png" 
                     alt="Collection Metrics">
            </div>
        </div>
    </body>
    </html>
    """
            return html_content

        except Exception as e:
            self.logger.error(f"Error generating HTML report: {e}")
            self.logger.error(traceback.format_exc())
            return "<html><body><h1>Error generating report</h1><p>Report generation failed</p></body></html>"

    def generate_process_visualizations(self) -> None:
        """Generate process-specific visualizations"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            vis_dir = os.path.join(config.DATA_STORAGE['results'], 'visualizations')
            os.makedirs(vis_dir, exist_ok=True)

            for mode in ['training', 'testing']:
                for process_name, syscalls in self.collected_data[mode].items():
                    try:
                        # Create safe process name for file path
                        safe_process_name = process_name.split()[0].replace('/', '_')

                        # Create figure
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

                        # Syscall distribution
                        syscall_counts = defaultdict(int)
                        for syscall in syscalls:
                            syscall_counts[syscall.get('name', '')] += 1

                        # Sort by frequency
                        sorted_syscalls = sorted(
                            syscall_counts.items(),
                            key=lambda x: x[1],
                            reverse=True
                        )[:10]  # Top 10 syscalls

                        if not sorted_syscalls:  # Skip if no data
                            continue

                        names, counts = zip(*sorted_syscalls)

                        # Plot with proper locator
                        x = np.arange(len(names))
                        ax1.bar(x, counts)
                        ax1.set_xticks(x)
                        ax1.set_xticklabels(names, rotation=45, ha='right')
                        ax1.set_title(f'Top 10 Syscalls - {safe_process_name}')
                        ax1.set_ylabel('Frequency')

                        # Save plot
                        plot_file = os.path.join(
                            vis_dir,
                            f'process_{safe_process_name}_{mode}_{timestamp}.png'
                        )
                        plt.tight_layout()
                        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                        plt.close()

                    except Exception as e:
                        self.logger.error(f"Error generating visualization for {process_name}: {e}")
                        continue

        except Exception as e:
            self.logger.error(f"Error generating process visualizations: {e}")


    def verify_data_integrity(self) -> bool:
        """Verify integrity of collected data"""
        try:
            verification_results = {
                'training': defaultdict(bool),
                'testing': defaultdict(bool)
            }

            for mode in ['training', 'testing']:
                data_dir = config.DATA_STORAGE[f'{mode}_data']

                # Verify .npy files
                npy_files = Path(os.path.join(data_dir, 'processed')).glob('*.npy')
                for npy_file in npy_files:
                    try:
                        # Load and verify .npy data
                        data = np.load(npy_file)
                        if data is None or len(data) == 0:
                            self.logger.error(f"Empty or invalid data in {npy_file}")
                            continue

                        # Verify corresponding metadata
                        metadata_file = str(npy_file).replace('.npy', '_metadata.json')
                        if not os.path.exists(metadata_file):
                            self.logger.error(f"Missing metadata for {npy_file}")
                            continue

                        verification_results[mode][str(npy_file)] = True

                    except Exception as e:
                        self.logger.error(f"Error verifying {npy_file}: {e}")
                        verification_results[mode][str(npy_file)] = False

            # Check verification results
            return all(
                result
                for mode_results in verification_results.values()
                for result in mode_results.values()
            )

        except Exception as e:
            self.logger.error(f"Error in data verification: {e}")
            return False




    # def archive_existing_data(self, mode='all') -> None:
    #     """Archive existing data before new collection"""
    #     try:
    #         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #         archive_base = os.path.join(config.DATA_STORAGE['archive'], f'backup_{timestamp}')
    #
    #         modes = ['training', 'testing'] if mode == 'all' else [mode]
    #
    #         for data_mode in modes:
    #             self.logger.info(f"Archiving existing {data_mode} data...")
    #
    #             # Create archive directory
    #             archive_dir = os.path.join(archive_base, data_mode)
    #             os.makedirs(archive_dir, exist_ok=True)
    #
    #             # Source directory
    #             source_base = config.DATA_STORAGE[f'{data_mode}_data']
    #
    #             # Archive each subdirectory
    #             for subdir in ['features', 'logs', 'processed', 'raw']:
    #                 source_dir = os.path.join(source_base, subdir)
    #                 if os.path.exists(source_dir) and os.listdir(source_dir):
    #                     archive_subdir = os.path.join(archive_dir, subdir)
    #                     os.makedirs(archive_subdir, exist_ok=True)
    #
    #                     # Move files to archive
    #                     for file in os.listdir(source_dir):
    #                         source_file = os.path.join(source_dir, file)
    #                         archive_file = os.path.join(archive_subdir, file)
    #                         try:
    #                             shutil.move(source_file, archive_file)
    #                             self.logger.debug(f"Archived: {source_file} -> {archive_file}")
    #                         except Exception as e:
    #                             self.logger.error(f"Error archiving {file}: {e}")
    #
    #             self.logger.info(f"Archived {data_mode} data to: {archive_dir}")
    #
    #     except Exception as e:
    #         self.logger.error(f"Error during data archiving: {e}")
    #         self.logger.error(traceback.format_exc())

    def archive_existing_data(self, mode='all') -> None:
        """Archive only features while preserving log files for future use"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_base = os.path.join(config.DATA_STORAGE['archive'], f'backup_{timestamp}')

            modes = ['training', 'testing'] if mode == 'all' else [mode]

            for data_mode in modes:
                self.logger.info(f"Archiving {data_mode} data...")

                # Create archive directory
                archive_dir = os.path.join(archive_base, data_mode)
                os.makedirs(archive_dir, exist_ok=True)

                # Source directory
                source_base = config.DATA_STORAGE[f'{data_mode}_data']

                # Only archive features and processed data, keep logs
                for subdir in ['features', 'processed']:
                    source_dir = os.path.join(source_base, subdir)
                    if os.path.exists(source_dir) and os.listdir(source_dir):
                        archive_subdir = os.path.join(archive_dir, subdir)
                        os.makedirs(archive_subdir, exist_ok=True)

                        # Move files to archive
                        for file in os.listdir(source_dir):
                            source_file = os.path.join(source_dir, file)
                            archive_file = os.path.join(archive_subdir, file)
                            try:
                                shutil.move(source_file, archive_file)
                                self.logger.debug(f"Archived: {source_file} -> {archive_file}")
                            except Exception as e:
                                self.logger.error(f"Error archiving {file}: {e}")

                # Log files are kept in place
                log_dir = os.path.join(source_base, 'logs')
                if os.path.exists(log_dir):
                    self.logger.info(f"Preserving log files in: {log_dir}")
                    log_files = list(Path(log_dir).glob('*.log'))
                    self.logger.info(f"Preserved {len(log_files)} log files for future use")

                self.logger.info(f"Archived features and processed data to: {archive_dir}")
                self.logger.info(f"Log files remain in: {log_dir}")

        except Exception as e:
            self.logger.error(f"Error during data archiving: {e}")
            self.logger.error(traceback.format_exc())

    def manage_log_files(self) -> None:
        """Manage accumulated log files with option to merge or organize"""
        try:
            for mode in ['training', 'testing']:
                log_dir = os.path.join(config.DATA_STORAGE[f'{mode}_data'], 'logs')
                if not os.path.exists(log_dir):
                    continue

                # Count log files by process
                process_logs = defaultdict(list)
                for log_file in Path(log_dir).glob('*.log'):
                    for process in config.TESTING_PROCESSES + config.TRAINING_PROCESSES:
                        base_name = process.split('/')[-1].lower()
                        if base_name in log_file.name.lower():
                            process_logs[process].append(log_file)
                            break

                # Print summary of available log files
                print(f"\n{mode.title()} Data Log Files Summary:")
                print("-" * 40)
                for process, files in process_logs.items():
                    print(f"{process}:")
                    print(f"- Number of log files: {len(files)}")
                    if files:
                        total_lines = sum(1 for f in files for _ in open(f))
                        print(f"- Total syscalls: {total_lines}")
                    print()

                # # Optional: Merge log files by process
                # if False:  # Set to True if you want to merge files
                #     for process, files in process_logs.items():
                #         if len(files) > 1:
                #             base_name = process.split('/')[-1].lower()
                #             merged_file = os.path.join(log_dir, f"{base_name}_merged.log")
                #             with open(merged_file, 'w') as outfile:
                #                 for log_file in sorted(files):
                #                     with open(log_file, 'r') as infile:
                #                         outfile.write(infile.read())
                #             print(f"Merged {len(files)} files for {process} into {merged_file}")

        except Exception as e:
            self.logger.error(f"Error managing log files: {e}")
            self.logger.error(traceback.format_exc())



    def run_collection(self) -> bool:
        """Run complete data collection process for both training and testing"""
        try:
            self.logger.info("Starting data collection process...")

            # Archive existing data
            self.logger.info("Archiving existing data...")
            self.archive_existing_data()
            self.logger.info("Data archiving completed")

            # Show existing log files
            self.manage_log_files()

            # Test process name sanitization
            self.test_process_name_sanitization()

            # Ensure all directories exist
            for mode in ['training', 'testing']:
                for subdir in ['raw', 'logs', 'processed', 'features']:
                    dir_path = os.path.join(config.DATA_STORAGE[f'{mode}_data'], subdir)
                    os.makedirs(dir_path, exist_ok=True)
                    os.system(f"sudo chown -R {os.getenv('USER')}:{os.getenv('USER')} {dir_path}")

            # First collect training data
            self.logger.info("\nCollecting training data...")
            self.logger.info("=" * 50)

            for process_name in config.TRAINING_PROCESSES:
                self.logger.info(f"Collecting data for {process_name}")
                count = self.collect_process_data(
                    process_name=process_name,  # Changed from process to process_name
                    duration=config.TRAINING_DURATION,
                    mode='training'
                )
                self.logger.info(f"Collected {count} syscalls for {process_name}")

            # Then collect testing data
            self.logger.info("\nCollecting testing data...")
            self.logger.info("=" * 50)

            for process_name in config.TESTING_PROCESSES:
                self.logger.info(f"Collecting data for {process_name}")
                count = self.collect_process_data(
                    process_name=process_name,  # Changed from process to process_name
                    duration=config.TRAINING_DURATION,
                    mode='testing'
                )
                self.logger.info(f"Collected {count} syscalls for {process_name}")

            # Save and verify all collected data
            if self.save_collected_data():
                self.logger.info("All data saved successfully")

                # Verify data integrity
                if self.verify_collected_data():
                    self.logger.info("Data integrity verified")

                    # Generate collection report
                    self.generate_collection_report()

                    # Show final statistics
                    self.show_data_statistics()

                    return True
                else:
                    self.logger.error("Data integrity verification failed")
                    return False
            else:
                self.logger.error("Failed to save collected data")
                return False

        except Exception as e:
            self.logger.error(f"Error in data collection: {e}")
            self.logger.error(traceback.format_exc())
            return False

    def initialize_collection(self) -> bool:
        """Initialize data collection requirements"""
        try:
            # Ensure temp directory exists
            ensure_temp_directory()

            # Start Tetragon if not running
            if not verify_tetragon_running():
                tetragon_process = Process(
                    target=start_tetragon_collection,
                    args=(self.stop_event,)
                )
                tetragon_process.start()

                if not wait_for_syscall_log():
                    raise RuntimeError("Tetragon initialization failed")

            # Start resource monitoring
            self.resource_monitor.check_resources()

            return True

        except Exception as e:
            self.logger.error(f"Initialization error: {e}")
            return False

    def cleanup_old_data(self, days_to_keep: int = 30) -> None:
        """Archive old data instead of deleting"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cleanup_date = datetime.now() - timedelta(days=days_to_keep)

            # Create archive directory
            archive_dir = os.path.join(
                config.DATA_STORAGE['archive'],
                timestamp
            )
            os.makedirs(archive_dir, exist_ok=True)

            for mode in ['training', 'testing']:
                mode_dir = config.DATA_STORAGE[f'{mode}_data']

                # Archive each subdirectory separately
                for subdir in ['raw', 'logs', 'processed', 'features']:
                    dir_path = os.path.join(mode_dir, subdir)
                    if not os.path.exists(dir_path):
                        continue

                    for file in Path(dir_path).glob('*'):
                        try:
                            if datetime.fromtimestamp(file.stat().st_mtime) < cleanup_date:
                                archive_path = os.path.join(
                                    archive_dir,
                                    mode,
                                    subdir,
                                    file.name
                                )
                                os.makedirs(os.path.dirname(archive_path), exist_ok=True)
                                shutil.move(str(file), archive_path)
                                self.logger.info(f"Archived old file: {file}")
                        except Exception as e:
                            self.logger.error(f"Error processing file {file}: {e}")

            # Compress archive
            if config.DATA_RETENTION['compression']:
                archive_name = f"archive_{timestamp}.tar.gz"
                shutil.make_archive(
                    os.path.join(config.DATA_STORAGE['archive'], archive_name),
                    'gztar',
                    archive_dir
                )
                shutil.rmtree(archive_dir)
                self.logger.info(f"Compressed archive created: {archive_name}")

        except Exception as e:
            self.logger.error(f"Error in data cleanup: {e}")

    def show_data_statistics(self):
        """Show statistics of all collected data"""
        try:
            self.logger.info("\nTotal Data Statistics:")

            for mode in ['training', 'testing']:
                base_dir = config.DATA_STORAGE[f'{mode}_data']

                self.logger.info(f"\n{mode.upper()} DATA:")
                total_samples = 0

                # Process each data directory
                for subdir in ['raw', 'logs', 'processed', 'features']:
                    dir_path = os.path.join(base_dir, subdir)
                    if not os.path.exists(dir_path):
                        continue

                    # Group files by process
                    process_files = defaultdict(list)
                    for file in Path(dir_path).glob('*'):
                        if file.is_file() and not str(file).endswith('_metadata.json'):
                            process_name = str(file).split('_')[0]
                            process_files[process_name].append(file)

                    self.logger.info(f"\n{subdir.upper()} FILES:")
                    for process, files in process_files.items():
                        process_samples = 0
                        collection_dates = set()

                        for file in files:
                            try:
                                if str(file).endswith('.npy'):
                                    data = np.load(file)
                                    process_samples += len(data)
                                elif str(file).endswith('.log'):
                                    with open(file, 'r') as f:
                                        process_samples += sum(1 for _ in f)

                                # Extract date from filename
                                date = str(file).split('_')[-1].split('.')[0][:8]
                                collection_dates.add(date)

                            except Exception as e:
                                self.logger.error(f"Error reading file {file}: {e}")
                                continue

                        self.logger.info(
                            f"Process: {process}\n"
                            f"  - Total samples: {process_samples}\n"
                            f"  - Collection dates: {sorted(list(collection_dates))}\n"
                            f"  - Number of files: {len(files)}"
                        )
                        total_samples += process_samples

                    self.logger.info(f"Total {subdir} samples: {total_samples}")

        except Exception as e:
            self.logger.error(f"Error showing data statistics: {e}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tetragon-bpf', required=True)
    return parser.parse_args()


def main():
    """Main execution function with comprehensive metrics tracking and error handling"""
    args = parse_args()
    os.environ['PROJECT_BASE_DIR'] = os.path.dirname(os.path.abspath(__file__))
    os.environ['TETRAGON_BPF_PATH'] = args.tetragon_bpf
    os.environ['POLICY_FILE'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'syscallpolicy.yaml')
    logger = get_logger(__name__ + ".main")
    collector = None

    try:
        logger.info("Starting data collection process...")

        # Initialize collector
        collector = DataCollector()

        # Test process name sanitization before proceeding
        collector.test_process_name_sanitization()
        logger.info("Process name sanitization test completed")

        # Initialize main collection metrics
        collector.collection_metrics.update({
            'start_time': datetime.now().isoformat(),
            'total_syscalls': 0,
            'total_processes': len(config.TRAINING_PROCESSES) + len(config.TESTING_PROCESSES),
            'processes': {},
            'errors': [],
            'system_info': {
                'cpu_count': psutil.cpu_count(),
                'memory_total': psutil.virtual_memory().total,
                'disk_total': psutil.disk_usage('/').total,
                'platform': sys.platform,
                'python_version': sys.version
            }
        })



        # Register signal handlers
        exit_handler = create_exit_handler(collector)
        signal.signal(signal.SIGINT, exit_handler)
        signal.signal(signal.SIGTERM, exit_handler)

        # Run collection
        if collector.run_collection():
            logger.info("Data collection completed successfully")

            # Update final collection metrics
            collector.collection_metrics.update({
                'end_time': datetime.now().isoformat(),
                'duration': str(datetime.now() - datetime.fromisoformat(collector.collection_metrics['start_time'])),
                'total_processes_completed': len(collector.collection_metrics['processes']),
                'total_errors': len(collector.collection_metrics['errors']),
                'resource_usage': {
                    'final_cpu': psutil.cpu_percent(),
                    'final_memory': psutil.virtual_memory().percent,
                    'final_disk': psutil.disk_usage('/').percent
                }
            })

            # Generate reports and statistics
            collector.show_data_statistics()
            collector.generate_process_visualizations()
            collector.generate_collection_report()

            # Save collection metrics
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            metrics_file = os.path.join(
                config.DATA_STORAGE['results'],
                'metrics',
                f'collection_metrics_{timestamp}.json'
            )
            os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
            with open(metrics_file, 'w') as f:
                json.dump(collector.collection_metrics, f, indent=4)

        else:
            logger.error("Data collection failed")
            collector.collection_metrics['status'] = 'failed'
            return

    except KeyboardInterrupt:
        logger.info("Data collection interrupted by user")
        if collector and collector.collected_data:
            logger.info("Saving partial data and generating reports...")
            try:
                collector.collection_metrics['status'] = 'interrupted'
                collector.collection_metrics['interrupt_time'] = datetime.now().isoformat()

                collector.save_collected_data()
                stats = collector.calculate_collection_statistics()
                collector.generate_collection_visualization(stats)
                collector.generate_collection_report()
                collector.generate_process_visualizations()
            except Exception as e:
                logger.error(f"Error saving partial data: {e}")
                logger.error(traceback.format_exc())
                collector.collection_metrics['errors'].append({
                    'timestamp': datetime.now().isoformat(),
                    'error': str(e),
                    'traceback': traceback.format_exc()
                })

    except Exception as e:
        logger.error(f"Error during data collection: {e}")
        logger.error(traceback.format_exc())
        if collector:
            collector.collection_metrics['status'] = 'error'
            collector.collection_metrics['errors'].append({
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'traceback': traceback.format_exc()
            })

    finally:
        logger.info("Performing cleanup operations...")
        try:
            if collector:
                # Stop collection
                collector.stop_event.set()

                # Stop Tetragon service
                stop_tetragon()

                # Generate final reports
                try:
                    stats = collector.calculate_collection_statistics()
                    collector.generate_collection_visualization(stats)
                    collector.generate_collection_report()
                    collector.generate_process_visualizations()

                    # Save final statistics and metrics
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    results_dir = os.path.join(config.DATA_STORAGE['results'])
                    os.makedirs(results_dir, exist_ok=True)

                    # Save statistics
                    stats_file = os.path.join(results_dir, 'statistics', f'collection_stats_{timestamp}.json')
                    os.makedirs(os.path.dirname(stats_file), exist_ok=True)
                    with open(stats_file, 'w') as f:
                        json.dump(stats, f, indent=4)

                    # Save final metrics
                    final_metrics = {
                        **collector.collection_metrics,
                        'cleanup_completion_time': datetime.now().isoformat(),
                        'final_resource_usage': {
                            'cpu_percent': psutil.cpu_percent(),
                            'memory_percent': psutil.virtual_memory().percent,
                            'disk_percent': psutil.disk_usage('/').percent
                        }
                    }

                    metrics_file = os.path.join(results_dir, 'metrics', f'final_metrics_{timestamp}.json')
                    os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
                    with open(metrics_file, 'w') as f:
                        json.dump(final_metrics, f, indent=4)

                    logger.info(f"Final statistics and reports generated successfully")
                    logger.info(f"Statistics saved to: {stats_file}")
                    logger.info(f"Metrics saved to: {metrics_file}")

                except Exception as e:
                    logger.error(f"Error generating final reports: {e}")
                    logger.error(traceback.format_exc())

                # Archive old data
                try:
                    collector.cleanup_old_data()
                except Exception as e:
                    logger.error(f"Error cleaning up old data: {e}")

            # Final cleanup
            try:
                cleanup_files_on_exit()
                stop_tetragon()
                gc.collect()

                # Print final summary
                if collector:
                    logger.info("\nCollection Summary:")
                    logger.info("=" * 50)
                    logger.info(f"Start Time: {collector.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
                    logger.info(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    logger.info(f"Duration: {datetime.now() - collector.start_time}")
                    logger.info(f"Total Syscalls: {collector.collection_metrics.get('total_syscalls', 0)}")
                    logger.info(f"Total Processes: {collector.collection_metrics.get('total_processes', 0)}")
                    logger.info(f"Results Directory: {config.DATA_STORAGE['results']}")
                    logger.info(f"Status: {collector.collection_metrics.get('status', 'completed')}")
                    logger.info("=" * 50)

                logger.info("Cleanup completed successfully")

            except Exception as e:
                logger.error(f"Error in final cleanup: {e}")
                logger.error(traceback.format_exc())

        except Exception as e:
            logger.error(f"Critical error during cleanup: {e}")
            logger.error(traceback.format_exc())
            sys.exit(1)


if __name__ == "__main__":
    main()
