import threading
import os
import time
import sys
import gc
import shutil
import subprocess
import signal
import logging
import json
from datetime import datetime
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from pathlib import Path
from queue import Empty
import pandas as pd
import seaborn as sns
import psutil
import numpy as np
from .config import config
import matplotlib.pyplot as plt
from .logging_setup import get_logger
from multiprocessing import Process, Queue, Event, Manager, Lock
import itertools
from collections import defaultdict



# ANSI color codes
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

in_memory_queue = Queue()

logger = logging.getLogger(__name__)

__all__ = [
    'collect_syscalls',
    'clean_up_files',
    'verify_tetragon_running',
    'rotate_tetragon_log',
    'flush_in_memory_data',
    'schedule_flush',
    'get_performance_metrics',
    'calculate_resource_usage',
    'start_tetragon_collection',
    'rotate_logs',
    'cleanup_old_archives',
    'wait_for_syscall_log',
    'stop_tetragon',
    'verify_running_processes',
    'get_process_pids',
    'process_syscall_log',
    'handle_exit',
    'cleanup_files_on_exit',
    'cleanup_shared_resources',
    'print_confusion_matrix_and_metrics',
    'validate_syscall',
    'start_background_tasks',
    'cleanup_old_files',
    'ensure_temp_directory',
    'cleanup_old_files',
    'rotate_logs'

]


# Utility function to ensure file exists with retries
def ensure_file_exists(log_path, retries=5, delay=1):
    """Ensure the syscall log file exists before attempting to process it."""
    for _ in range(retries):
        if os.path.exists(log_path):
            return True
        logger.warning(f"File {log_path} not found. Retrying in {delay} seconds...")
        time.sleep(delay)
    logger.error(f"File {log_path} could not be found after {retries} retries.")
    return False

def process_syscall_log(log_file):
    """Process the syscall log file."""
    log_path = os.path.join(config.TEMP_DIR, log_file)

    if ensure_file_exists(log_path):
        logger.info(f"Processing syscall log: {log_path}")
        try:
            with open(log_path, 'r') as f:
                data = f.read()
                logger.info(f"Successfully processed the log file: {log_file}")
        except Exception as e:
            logger.error(f"Error processing the log file: {e}")
    else:
        logger.error(f"Failed to process syscalls, {log_file} is missing.")


def verify_tetragon_running():
    """Verify if Tetragon is running and collecting data"""
    try:
        # First verify the process
        try:
            result = subprocess.run(['pgrep', '-f', 'tetragon'], capture_output=True, text=True)
            if result.returncode != 0:
                # If Tetragon is not running, start it
                logger.info("Tetragon not running, starting it...")
                start_cmd = (
                    f"sudo tetragon "
                    f"--bpf-lib {config.TETRAGON_BPF_LIB} "
                    f"--tracing-policy {config.SYSCALL_POLICY_FILE} "
                    f"--export-filename {config.SYSCALL_LOG_FILE} "
                    f"--log-level {config.TETRAGON_LOG_LEVEL} "
                )
                subprocess.Popen(start_cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                time.sleep(5)  # Wait for startup

                # Check again
                result = subprocess.run(['pgrep', '-f', 'tetragon'], capture_output=True, text=True)
                if result.returncode != 0:
                    return False

        except Exception as e:
            logger.error(f"Error checking Tetragon process: {e}")
            return False

        # Get all Tetragon PIDs
        tetragon_pids = [int(pid) for pid in result.stdout.strip().split()]
        if not tetragon_pids:
            return False

        # Check if each process is actually running
        for pid in tetragon_pids:
            try:
                process = psutil.Process(pid)
                if process.status() not in [psutil.STATUS_RUNNING, psutil.STATUS_SLEEPING]:
                    return False
            except psutil.NoSuchProcess:
                return False

        # Check if log directory exists and is writable
        if not os.path.exists(config.LOG_DIR):
            os.makedirs(config.LOG_DIR, exist_ok=True)
            os.system(f"sudo chown -R {os.getenv('USER')}:{os.getenv('USER')} {config.LOG_DIR}")

        # Verify log file exists or can be created
        syscall_log_dir = os.path.dirname(config.SYSCALL_LOG_FILE)
        if not os.path.exists(syscall_log_dir):
            os.makedirs(syscall_log_dir, exist_ok=True)
            os.system(f"sudo chown -R {os.getenv('USER')}:{os.getenv('USER')} {syscall_log_dir}")

        # Wait for log file to appear
        for _ in range(5):  # Wait up to 5 seconds
            if os.path.exists(config.SYSCALL_LOG_FILE):
                return True
            time.sleep(1)

        return False

    except Exception as e:
        logger.error(f"Error verifying Tetragon: {e}")
        return False


def verify_syscall_data(log_file: str, max_retries: int = 3) -> bool:
    """Verify Tetragon is collecting data properly"""
    for attempt in range(max_retries):
        try:
            if os.path.exists(log_file):
                initial_size = os.path.getsize(log_file)
                time.sleep(1)  # Wait a second
                current_size = os.path.getsize(log_file)

                if current_size > initial_size:
                    # Check if file contains valid data
                    with open(log_file, 'r') as f:
                        first_line = f.readline().strip()
                        if first_line and first_line.startswith('{'):
                            try:
                                data = json.loads(first_line)
                                if 'process_tracepoint' in data:
                                    return True
                            except json.JSONDecodeError:
                                pass
            time.sleep(1)
        except Exception as e:
            logger.error(f"Error verifying syscall data: {e}")

    return False

# def start_tetragon_collection(stop_event):
#     """Start Tetragon collection with simplified configuration"""
#     logger.info("Starting Tetragon collection...")
#
#     try:
#         # Create necessary directories
#         os.makedirs(config.LOG_DIR, exist_ok=True)
#
#         # Get absolute paths
#         bpf_lib_path = os.path.abspath(config.TETRAGON_BPF_LIB)
#         policy_path = os.path.abspath(config.SYSCALL_POLICY_FILE)
#         log_path = os.path.abspath(config.SYSCALL_LOG_FILE)
#
#         # Clear existing log file
#         if os.path.exists(log_path):
#             os.remove(log_path)
#
#         # Command for running Tetragon - simpler configuration
#         command = (
#             f"sudo tetragon "
#             f"--bpf-lib {bpf_lib_path} "
#             f"--tracing-policy {policy_path} "
#             f"--export-filename {log_path} "
#             f"--log-level {config.TETRAGON_LOG_LEVEL} "
#             f"> {config.TEMP_LOG} 2>&1"
#         )
#
#         process = None
#         try:
#             process = subprocess.Popen(command, shell=True, preexec_fn=os.setsid)
#             logger.info(f"Started Tetragon process with PID: {process.pid}")
#
#             # Wait for initialization
#             time.sleep(config.TETRAGON_CHECK_INTERVAL)
#
#             # Monitor the process
#             while not stop_event.is_set():
#                 if not verify_tetragon_running():
#                     logger.error("Tetragon not running properly, restarting...")
#                     if process:
#                         os.killpg(os.getpgid(process.pid), signal.SIGTERM)
#                         process.wait(timeout=5)
#                     process = subprocess.Popen(command, shell=True, preexec_fn=os.setsid)
#                     logger.info(f"Restarted Tetragon with PID: {process.pid}")
#                     time.sleep(config.TETRAGON_CHECK_INTERVAL)
#
#                 # Check log file size
#                 if os.path.exists(log_path) and os.path.getsize(log_path) > config.MAX_LOG_SIZE:
#                     rotate_tetragon_log(log_path)
#
#                 time.sleep(1)
#
#         except Exception as e:
#             logger.error(f"Error in Tetragon process management: {e}")
#             raise
#
#     except Exception as e:
#         logger.error(f"Critical error in Tetragon collection: {e}")
#         raise
#
#     finally:
#         if process and process.poll() is None:
#             logger.info("Cleaning up Tetragon process...")
#             try:
#                 os.killpg(os.getpgid(process.pid), signal.SIGTERM)
#                 process.wait(timeout=5)
#             except:
#                 os.killpg(os.getpgid(process.pid), signal.SIGKILL)

def start_tetragon_collection(stop_event):
    """Start Tetragon collection with proper startup verification"""
    logger.info("Starting Tetragon collection...")

    try:
        # Stop any existing Tetragon processes
        print("11111111111111111111111111")
        #stop_tetragon()
        print("1111111111111111111111111122222222222")
        time.sleep(2)  # Wait for cleanup

        # Ensure directories exist
        os.makedirs(os.path.dirname(config.SYSCALL_LOG_FILE), exist_ok=True)
        os.system(f"sudo chown -R {os.getenv('USER')}:{os.getenv('USER')} {os.path.dirname(config.SYSCALL_LOG_FILE)}")
        print("111111111111111111111111113333333333333333333")

        tetrag = os.getenv('TETRAGON_BPF_PATH')

        print(f"111111144444444444444444444444444 {tetrag} bpf-lib: {config.TETRAGON_BPF_LIB} policy {config.SYSCALL_POLICY_FILE} log {config.SYSCALL_LOG_FILE} level {config.TETRAGON_LOG_LEVEL}")
        # Start Tetragon with sudo
        start_cmd = (
            f"sudo {tetrag} "
            f"--bpf-lib {config.TETRAGON_BPF_LIB} "
            f"--tracing-policy {config.SYSCALL_POLICY_FILE} "
            f"--export-filename {config.SYSCALL_LOG_FILE} "
            f"--log-level {config.TETRAGON_LOG_LEVEL} "
        )

        process = subprocess.Popen(
            start_cmd.split(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid
        )

        # Wait for startup
        time.sleep(5)

        # Verify startup was successful
        if verify_tetragon_running():
            logger.info("Tetragon started successfully")
            return True
        else:
            logger.error("Failed to start Tetragon properly")
            return False

    except Exception as e:
        logger.error(f"Error starting Tetragon: {e}")
        return False


def verify_running_processes(processes_list, phase='train'):
    """Enhanced process verification with detailed status checking"""
    running_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'status']):
        try:
            cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ""
            for target in processes_list:
                if target in cmdline:
                    status = proc.info['status']
                    logger.info(
                        f"Found {phase} process: {proc.info['name']} (PID: {proc.info['pid']}, Status: {status})")
                    running_processes.append(proc.info['pid'])
                    break
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    return running_processes


def collect_syscalls(log_file, retry_attempts=5, retry_delay=2):
    """Collect syscalls with retries and proper verification"""
    try:
        if not log_file:
            logger.error("Log file path is empty")
            return False

        log_file = os.path.join(config.TEMP_DIR, log_file)
        syscall_log = os.path.abspath(config.SYSCALL_LOG_FILE)

        # Ensure directories exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        os.makedirs(os.path.dirname(syscall_log), exist_ok=True)

        for attempt in range(retry_attempts):
            try:
                # First verify Tetragon is running properly
                if not verify_tetragon_running():
                    logger.warning(f"Waiting for Tetragon to start (attempt {attempt + 1})")
                    time.sleep(retry_delay)
                    continue

                # Then verify data collection
                if not verify_syscall_data(syscall_log):
                    logger.warning(f"Waiting for syscall data (attempt {attempt + 1})")
                    time.sleep(retry_delay)
                    continue

                # Copy file if verification passed
                shutil.copy2(syscall_log, log_file)
                os.chmod(log_file, 0o666)

                if os.path.exists(log_file) and os.path.getsize(log_file) > 0:
                    with open(log_file, 'r') as f:
                        if f.readline().strip():
                            return True

                logger.warning(f"Invalid data in copied file (attempt {attempt + 1})")
                time.sleep(retry_delay)

            except Exception as e:
                logger.error(f"Error collecting syscalls (attempt {attempt + 1}): {e}")
                time.sleep(retry_delay)

        logger.error(f"Failed to collect syscalls after {retry_attempts} attempts")
        return False

    except Exception as e:
        logger.error(f"Critical error in collect_syscalls: {e}")
        return False


def get_performance_metrics(pid):
    """Enhanced performance metrics collection"""
    try:
        process = psutil.Process(pid)
        with process.oneshot():
            cpu_times = process.cpu_times()
            memory_info = process.memory_info()
            io_counters = process.io_counters()
            num_threads = process.num_threads()

        return {
            'cpu_times': cpu_times,
            'memory_info': memory_info,
            'io_counters': io_counters,
            'num_threads': num_threads
        }
    except psutil.NoSuchProcess:
        logger.error(f"Process with PID {pid} no longer exists.")
        return None


# def get_process_pids(process_name):
#     """Get PIDs for a process with enhanced matching"""
#     pids = []
#     try:
#         for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
#             try:
#                 if process_name in proc.info['name'] or \
#                    (proc.info['cmdline'] and process_name in ' '.join(proc.info['cmdline'])):
#                     pids.append(proc.info['pid'])
#             except (psutil.NoSuchProcess, psutil.AccessDenied):
#                 continue
#     except Exception as e:
#         logger.error(f"Error getting PIDs for {process_name}: {e}")
#     return pids

def get_process_pids(process_name):
    """Get the PIDs for the process with the given name."""
    pids = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = " ".join(proc.info['cmdline']) if proc.info['cmdline'] else ""
            if process_name in proc.info['name'] or process_name in cmdline:
                pids.append(proc.info['pid'])
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return pids


def clean_up_files(log_file, text_file):
    if os.path.exists(log_file):
        os.remove(log_file)
    if os.path.exists(text_file):
        os.remove(text_file)


def cleanup_files_on_exit():
    """Enhanced cleanup with archiving and better error handling."""
    logger.info("Performing cleanup...")

    # try:
    #     # Create timestamp for archive
    #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #     archive_subdir = os.path.join(config.ARCHIVE_DIR, timestamp)
    #     os.makedirs(archive_subdir, exist_ok=True)
    #
    #     # Ensure a delay before cleanup
    #     time.sleep(2)  # Introduce a delay to ensure all processes have finished with the files
    #
    #     # Cleanup temporary files
    #     cleanup_patterns = [
    #         '*.log',
    #         'SSG_*.png',
    #         'training_*.png',
    #         'temp_*.npy',
    #         'in_memory_syscalls_*.log'
    #     ]
    #
    #     for pattern in cleanup_patterns:
    #         for filepath in Path(config.TEMP_DIR).glob(pattern):
    #             try:
    #                 if 'training' in str(filepath) or 'SSG' in str(filepath):
    #                     # Archive important files
    #                     archive_path = os.path.join(archive_subdir, filepath.name)
    #                     shutil.move(str(filepath), archive_path)
    #                     logger.debug(f"Archived: {filepath} -> {archive_path}")
    #                 else:
    #                     # Remove temporary files
    #                     os.remove(filepath)
    #                     logger.debug(f"Removed: {filepath}")
    #             except Exception as e:
    #                 logger.error(f"Error handling file {filepath}: {e}")
    #
    #     # Cleanup old archives
    #     cleanup_old_archives()
    #
    #     logger.info("Cleanup completed successfully")
    #
    # except Exception as e:
    #     logger.error(f"Error during cleanup: {e}")

    """Modified cleanup to preserve training data"""
    try:
        # Only clean temporary files
        temp_patterns = [
            'temp_*.log',
            'temp_*.txt',
            'temp_*.npy'
        ]

        for pattern in temp_patterns:
            for filepath in Path(config.TEMP_DIR).glob(pattern):
                try:
                    os.remove(filepath)
                except Exception as e:
                    logging.error(f"Error removing temp file {filepath}: {e}")

    except Exception as e:
        logging.error(f"Error in cleanup: {e}")


def flush_in_memory_data():
    """Flush in-memory data to log files"""
    global in_memory_queue
    while not in_memory_queue.empty():
        syscall_data = in_memory_queue.get()
        log_file = f"in_memory_syscalls_{int(time.time())}.log"
        with open(log_file, 'w') as f:
            f.write(str(syscall_data))
        logging.info(f"Flushed in-memory syscall data to log file: {log_file}")


# def schedule_flush(in_memory_queue):
#     """Schedule periodic data flushing with monitoring"""
#     try:
#         success = flush_in_memory_data()
#         if success:
#             logger.info("Scheduled flush completed successfully")
#         else:
#             logger.warning("Scheduled flush completed with issues")
#     except Exception as e:
#         logger.error(f"Error in scheduled flush: {e}")
#     finally:
#         threading.Timer(config.FLUSH_INTERVAL, schedule_flush, args=[in_memory_queue]).start()

def schedule_flush():
    """Schedule periodic flushing of in-memory data."""
    flush_in_memory_data()
    threading.Timer(config.FLUSH_INTERVAL, schedule_flush).start()


def start_background_tasks():
    """Start all background tasks"""
    try:
        schedule_flush()
        logger.info("Background tasks started successfully")
    except Exception as e:
        logger.error(f"Error starting background tasks: {e}")
        raise


class DataArchiver:
    def __init__(self):
        self.logger = get_logger(__name__)

    def archive_old_data(self):
        """Archive old data files"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_dir = os.path.join(config.DATA_STORAGE['archive'], timestamp)
            os.makedirs(archive_dir, exist_ok=True)

            # Archive threshold (e.g., 7 days old)
            threshold = datetime.now() - timedelta(
                seconds=config.DATA_RETENTION['archive_interval']
            )

            for data_type, directory in config.DATA_STORAGE.items():
                if data_type == 'archive':
                    continue

                for file_path in Path(directory).glob('*'):
                    file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_time < threshold:
                        # Archive file
                        archive_path = os.path.join(
                            archive_dir,
                            f"{data_type}_{file_path.name}"
                        )
                        if config.DATA_RETENTION['compression']:
                            with tarfile.open(f"{archive_path}.tar.gz", "w:gz") as tar:
                                tar.add(file_path)
                            os.remove(file_path)
                        else:
                            shutil.move(file_path, archive_path)

        except Exception as e:
            self.logger.error(f"Error archiving data: {e}")

class DataManager:
    def __init__(self):
        self.logger = get_logger(__name__)
        self.archiver = DataArchiver()

    def get_data_statistics(self):
        """Get statistics about stored data"""
        stats = {}
        for data_type, directory in config.DATA_STORAGE.items():
            if data_type == 'archive':
                continue

            files = list(Path(directory).glob('*'))
            stats[data_type] = {
                'file_count': len(files),
                'total_size': sum(f.stat().st_size for f in files),
                'oldest_file': min(files, key=lambda x: x.stat().st_mtime).name,
                'newest_file': max(files, key=lambda x: x.stat().st_mtime).name
            }
        return stats

    def cleanup_old_data(self, force=False):
        """Archive old data based on retention policy"""
        if force or self._should_archive():
            self.archiver.archive_old_data()

    def _should_archive(self):
        """Check if archiving is needed based on storage usage"""
        total_size = 0
        for directory in config.DATA_STORAGE.values():
            if os.path.exists(directory):
                total_size += sum(
                    f.stat().st_size for f in Path(directory).glob('*')
                )

        # Archive if total size exceeds threshold (e.g., 1GB)
        return total_size > (1024 * 1024 * 1024)  # 1GB in bytes

def stop_tetragon():
    """Enhanced Tetragon shutdown with forced termination if necessary."""
    try:
        result = subprocess.run(['pgrep', '-f', 'tetragon'], capture_output=True, text=True)
        if result.returncode == 0:
            pids = result.stdout.strip().split()

            for pid in pids:
                try:
                    os.kill(int(pid), signal.SIGTERM)
                except ProcessLookupError:
                    continue

            time.sleep(config.TETRAGON_RESTART_DELAY)

            result = subprocess.run(['pgrep', '-f', 'tetragon'], capture_output=True, text=True)
            if result.returncode == 0:
                logger.warning("Tetragon didn't stop gracefully, forcing termination...")
                subprocess.run(['sudo', 'pkill', '-9', '-f', 'tetragon'])

        logger.info("Tetragon stopped successfully.")
    except Exception as e:
        logger.error(f"Error stopping Tetragon: {e}")


def calculate_resource_usage(start_metrics, end_metrics):
    """Enhanced resource usage calculation"""
    if not (start_metrics and end_metrics):
        return 0, 0, 0

    cpu_usage = (end_metrics['cpu_times'].user + end_metrics['cpu_times'].system) - \
                (start_metrics['cpu_times'].user + start_metrics['cpu_times'].system)

    memory_usage = (end_metrics['memory_info'].rss - start_metrics['memory_info'].rss) / (1024 * 1024)  # MB

    io_usage = sum([
        end_metrics['io_counters'].read_bytes - start_metrics['io_counters'].read_bytes,
        end_metrics['io_counters'].write_bytes - start_metrics['io_counters'].write_bytes
    ]) / (1024 * 1024)  # MB

    return cpu_usage, memory_usage, io_usage

def rotate_tetragon_log(log_path):
    """Rotate Tetragon log file"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{log_path}.{timestamp}"
        shutil.move(log_path, backup_path)
        logger.info(f"Rotated Tetragon log to: {backup_path}")

        log_dir = os.path.dirname(log_path)
        log_files = sorted(
            Path(log_dir).glob(f"{os.path.basename(log_path)}.*"),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )

        for old_log in log_files[config.LOG_ROTATION_COUNT:]:
            try:
                os.remove(old_log)
                logger.debug(f"Removed old Tetragon log: {old_log}")
            except Exception as e:
                logger.error(f"Error removing old Tetragon log {old_log}: {e}")

    except Exception as e:
        logger.error(f"Error rotating Tetragon log: {e}")

def rotate_logs():
    """Rotate log files to prevent excessive size"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if os.path.exists(config.LOG_FILE) and os.path.getsize(config.LOG_FILE) > config.MAX_LOG_SIZE:
            backup_file = f"{config.LOG_FILE}.{timestamp}"
            shutil.move(config.LOG_FILE, backup_file)
            logger.info(f"Rotated log file to: {backup_file}")

        cleanup_old_logs()
    except Exception as e:
        logger.error(f"Error rotating logs: {e}")

def cleanup_old_logs():
    """Clean up old log files while keeping recent ones"""
    try:
        log_files = []
        for pattern in ['*.log', '*.log.*']:
            log_files.extend(Path(config.LOG_DIR).glob(pattern))

        log_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        for log_file in log_files[config.LOG_ROTATION_COUNT:]:
            try:
                os.remove(log_file)
                logger.debug(f"Removed old log file: {log_file}")
            except Exception as e:
                logger.error(f"Error removing old log file {log_file}: {e}")

    except Exception as e:
        logger.error(f"Error cleaning up old logs: {e}")

def wait_for_syscall_log(timeout=30, check_interval=1):
    """Wait for the syscall log file to be created and contain data."""
    start_time = time.time()
    attempts = 0
    max_attempts = timeout // check_interval

    while attempts < max_attempts:
        try:
            if os.path.exists(config.SYSCALL_LOG_FILE):
                if os.path.getsize(config.SYSCALL_LOG_FILE) > 0:
                    with open(config.SYSCALL_LOG_FILE, 'r') as f:
                        first_line = f.readline().strip()
                        if first_line:
                            logger.info(f"Syscall log file found and contains data after {attempts * check_interval} seconds")
                            return True

            logger.debug(f"Waiting for syscall log file... Attempt {attempts + 1}/{max_attempts}")
            time.sleep(check_interval)
            attempts += 1

        except Exception as e:
            logger.error(f"Error checking syscall log file: {e}")
            return False


def cleanup_shared_resources(in_memory_queue):
    """Clean up shared resources and memory"""
    try:
        # Clear queues
        while not in_memory_queue.empty():
            try:
                in_memory_queue.get_nowait()
            except Empty:
                break

        # Release locks if any are held
        if hasattr(config.model_lock, '_semlock'):
            config.model_lock._semlock.release()

    except Exception as e:
        logger.error(f"Error cleaning up shared resources: {e}")



def print_confusion_matrix_and_metrics(y_true, y_pred, save_dir='results'):
    """Enhanced metrics visualization with proper handling of single-class predictions"""
    if not y_true or not y_pred:
        logger.error("No data available for metrics calculation")
        return

    try:
        # Convert ListProxy objects or lists to numpy arrays
        y_true_arr = np.array(list(y_true))
        y_pred_arr = np.array(list(y_pred))

        # Create results directory
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Ensure we have predictions from both classes
        unique_true = np.unique(y_true_arr)
        unique_pred = np.unique(y_pred_arr)

        print("\n" + "=" * 50)
        print("Detection Performance Results")
        print("=" * 50)

        if len(unique_true) < 2 or len(unique_pred) < 2:
            logger.warning("Only single class predictions available. Limited metrics will be shown.")

            # Calculate basic metrics
            accuracy = accuracy_score(y_true_arr, y_pred_arr)

            # Print limited metrics
            print("\nMetrics Summary (Limited - Single Class Predictions):")
            print("-" * 50)
            print(f"Total samples: {len(y_true_arr)}")
            print(f"Accuracy: {accuracy:.4f}")

            # Enhanced class distribution display
            pred_dist = np.bincount(y_pred_arr)
            true_dist = np.bincount(y_true_arr)

            print("\nClass Distribution:")
            print("-" * 30)
            print("Predicted classes:")
            for i, count in enumerate(pred_dist):
                print(f"  Class {i}: {count} samples ({count / len(y_pred_arr) * 100:.2f}%)")

            print("\nTrue classes:")
            for i, count in enumerate(true_dist):
                print(f"  Class {i}: {count} samples ({count / len(y_true_arr) * 100:.2f}%)")

        else:
            # Calculate full metrics
            conf_matrix = confusion_matrix(y_true_arr, y_pred_arr)
            accuracy = accuracy_score(y_true_arr, y_pred_arr)
            precision = precision_score(y_true_arr, y_pred_arr, zero_division=0)
            recall = recall_score(y_true_arr, y_pred_arr, zero_division=0)
            f1 = f1_score(y_true_arr, y_pred_arr, zero_division=0)

            # Calculate additional metrics
            tn = conf_matrix[0, 0]
            fp = conf_matrix[0, 1]
            fn = conf_matrix[1, 0]
            tp = conf_matrix[1, 1]

            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

            # Print comprehensive metrics
            print("\nClassification Metrics:")
            print("-" * 30)
            print(f"Accuracy:    {accuracy:.4f}")
            print(f"Precision:   {precision:.4f}")
            print(f"Recall:      {recall:.4f}")
            print(f"F1 Score:    {f1:.4f}")
            print(f"Specificity: {specificity:.4f}")

            print("\nError Rates:")
            print("-" * 30)
            print(f"False Positive Rate: {fpr:.4f}")
            print(f"False Negative Rate: {fnr:.4f}")

            print("\nConfusion Matrix:")
            print("-" * 30)
            print(f"True Negatives (TN):  {tn}")
            print(f"False Positives (FP): {fp}")
            print(f"False Negatives (FN): {fn}")
            print(f"True Positives (TP):  {tp}")

            # Class distribution
            print("\nClass Distribution:")
            print("-" * 30)
            print("Predicted classes:")
            pred_dist = np.bincount(y_pred_arr)
            for i, count in enumerate(pred_dist):
                print(f"  Class {i}: {count} samples ({count / len(y_pred_arr) * 100:.2f}%)")

            print("\nTrue classes:")
            true_dist = np.bincount(y_true_arr)
            for i, count in enumerate(true_dist):
                print(f"  Class {i}: {count} samples ({count / len(y_true_arr) * 100:.2f}%)")

            # Create visualization
            plt.figure(figsize=(12, 8))

            # Plot confusion matrix
            plt.subplot(221)
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')

            # Plot prediction distribution
            plt.subplot(222)
            pd.Series(y_pred_arr).value_counts().plot(kind='bar')
            plt.title('Prediction Distribution')
            plt.xlabel('Class')
            plt.ylabel('Count')

            plt.tight_layout()
            metrics_file = os.path.join(save_dir, f'performance_metrics_{timestamp}.png')
            plt.savefig(metrics_file, dpi=300, bbox_inches='tight')
            plt.close()

        # Save detailed metrics
        metrics_dict = {
            'timestamp': timestamp,
            'total_samples': len(y_true_arr),
            'accuracy': float(accuracy),
            'class_distribution': {
                'predictions': np.bincount(y_pred_arr).tolist(),
                'true_labels': np.bincount(y_true_arr).tolist()
            }
        }

        if len(unique_true) >= 2 and len(unique_pred) >= 2:
            metrics_dict.update({
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'specificity': float(specificity),
                'confusion_matrix': {
                    'tn': int(tn),
                    'fp': int(fp),
                    'fn': int(fn),
                    'tp': int(tp)
                }
            })

        metrics_file = os.path.join(save_dir, f'metrics_{timestamp}.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics_dict, f, indent=4)

        logger.info(f"Metrics saved to {metrics_file}")

    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        import traceback
        logger.error(traceback.format_exc())



def handle_exit(signum, frame):
    """Enhanced exit handler with comprehensive cleanup and metrics saving"""
    logger = get_logger(__name__)

    if os.getpid() == config.main_process_pid:
        logger.info("Initiating graceful shutdown...")

        # Set stop event for all processes
        config.stop_event.set()

        try:
            # Create results directory with timestamp
            results_dir = os.path.join(config.BASE_DIR, 'results',
                                     datetime.now().strftime("%Y%m%d_%H%M%S"))
            os.makedirs(results_dir, exist_ok=True)

            # Stop Tetragon with timeout
            stop_tetragon()

            # Print and save final results if available
            if hasattr(config, 'y_true') and hasattr(config, 'y_pred'):
                print("\nFinal Detection Results:")
                print("=" * 50)
                print_confusion_matrix_and_metrics(config.y_true, config.y_pred,
                                                save_dir=results_dir)

            # Stop all monitoring processes
            for proc in config.processes:
                try:
                    proc.terminate()
                    proc.join(timeout=config.PROCESS_TIMEOUT)
                    if proc.is_alive():
                        logger.warning(f"Force killing process {proc.name}")
                        proc.kill()
                except Exception as e:
                    logger.error(f"Error terminating process {proc.name}: {e}")

            # Clear shared resources
            try:
                cleanup_shared_resources(config.in_memory_queue)
            except Exception as e:
                logger.error(f"Error cleaning shared resources: {e}")

            # Cleanup files based on mode
            try:
                if hasattr(config, 'mode'):
                    cleanup_files_on_exit(mode=config.mode)
                else:
                    cleanup_files_on_exit(mode='test')
            except Exception as e:
                logger.error(f"Error in file cleanup: {e}")

            # Save monitoring statistics if available
            if hasattr(config, 'monitor'):
                try:
                    stats_file = os.path.join(results_dir, 'monitoring_stats.json')
                    config.monitor.save_statistics(stats_file)
                    logger.info(f"Monitoring statistics saved to {stats_file}")
                except Exception as e:
                    logger.error(f"Error saving monitoring stats: {e}")

            # Save system resource usage
            try:
                resource_stats = {
                    'cpu_percent': psutil.cpu_percent(),
                    'memory_percent': psutil.virtual_memory().percent,
                    'disk_usage': psutil.disk_usage('/').percent,
                }
                with open(os.path.join(results_dir, 'resource_usage.json'), 'w') as f:
                    json.dump(resource_stats, f, indent=4)
            except Exception as e:
                logger.error(f"Error saving resource stats: {e}")

            # Save execution summary
            try:
                summary = {
                    'execution_time': str(datetime.now()),
                    'total_processes_monitored': len(config.TESTING_PROCESSES),
                    'exit_status': 'graceful',
                    'results_location': results_dir
                }
                with open(os.path.join(results_dir, 'execution_summary.json'), 'w') as f:
                    json.dump(summary, f, indent=4)
            except Exception as e:
                logger.error(f"Error saving execution summary: {e}")

            # Final garbage collection
            gc.collect()

            logger.info(f"Shutdown completed successfully. Results saved in: {results_dir}")
            print(f"\n{GREEN}Results and metrics saved in: {results_dir}{RESET}")
            sys.exit(0)

        except Exception as e:
            logger.error(f"Critical error during shutdown: {e}")
            import traceback
            logger.error(traceback.format_exc())
            sys.exit(1)

    else:
        logger.info(f"Child process {os.getpid()} exiting")
        sys.exit(0)

def cleanup_old_archives():
    """Clean up old archive directories while keeping recent ones"""
    try:
        archives = sorted(Path(config.ARCHIVE_DIR).glob('*'))
        if len(archives) > config.LOG_ROTATION_COUNT:
            for archive in archives[:-config.LOG_ROTATION_COUNT]:
                try:
                    shutil.rmtree(archive)
                    logger.debug(f"Removed old archive: {archive}")
                except Exception as e:
                    logger.error(f"Error removing old archive {archive}: {e}")

    except Exception as e:
        logger.error(f"Error cleaning up old archives: {e}")


def validate_syscall(syscall):
    """Validate syscall structure and content"""
    required_fields = ['name', 'binary', 'args']

    # Check required fields exist
    if not all(field in syscall for field in required_fields):
        return False

    # Validate syscall name
    if not syscall['name'] or not isinstance(syscall['name'], str):
        return False

    # Validate binary path
    if not syscall['binary'] or not isinstance(syscall['binary'], str):
        return False

    # Validate arguments
    if not isinstance(syscall['args'], (list, tuple)):
        return False

    return True


def ensure_temp_directory():
    """Ensure temp directory exists with proper permissions"""
    try:
        os.makedirs(config.TEMP_DIR, exist_ok=True)
        os.chmod(config.TEMP_DIR, 0o755)
        # Ensure write permissions for the current user
        os.system(f"sudo chown -R {os.getenv('USER')}:{os.getenv('USER')} {config.TEMP_DIR}")
        return True
    except Exception as e:
        logging.error(f"Error setting up temp directory: {e}")
        return False


def ensure_file_permissions(filepath: str) -> bool:
    """Ensure file exists and has proper permissions"""
    try:
        directory = os.path.dirname(filepath)
        os.makedirs(directory, exist_ok=True)
        if os.path.exists(filepath):
            os.chmod(filepath, 0o644)
        return True
    except Exception as e:
        logging.error(f"Error setting file permissions for {filepath}: {e}")
        return False

def cleanup_old_files():
    """Cleanup old temporary files"""
    try:
        for pattern in ['*.log', '*.tmp']:
            for filepath in Path(config.TEMP_DIR).glob(pattern):
                try:
                    if filepath.is_file():
                        filepath.unlink()
                except Exception as e:
                    logging.error(f"Error removing file {filepath}: {e}")
    except Exception as e:
        logging.error(f"Error during cleanup: {e}")
