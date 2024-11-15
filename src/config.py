import os
import logging
from pathlib import Path
from multiprocessing import Event, Manager
import yaml
import traceback
import shutil

def is_nixos():
    """Check if running on NixOS"""
    try:
        with open('/etc/os-release') as f:
            return 'NixOS' in f.read()
    except:
        return False


class Config:

    @staticmethod
    def get_process_paths():
        """Get correct process paths based on environment"""
        if is_nixos():
            return {
                #'chrome': "/run/current-system/sw/bin/chrome",
                #'teams': "/run/current-system/sw/bin/teams",
                #'python3': "/run/current-system/sw/bin/python3",
                #'firefox': "/run/current-system/sw/bin/firefox"
                'teams': "/nix/store/qaiwka94wmqskr6rfyjxzgxh56ha22rx-electron-unwrapped-29.4.5/libexec/electron/electron",
                'firefox':"/nix/store/dm3kb1zqz4i8cjn39ybphzk0ny2lj2vr-firefox-129.0.2/lib/firefox/firefox"
            }
        return {
            'chrome': "/opt/google/chrome/chrome",
            'teams': "/usr/share/teams/teams",
            'python3': "/usr/bin/python3",
            'firefox': "/usr/bin/firefox"
        }

    def validate_environment(self):
        """Validate environment setup"""
        required_env = ['PROJECT_BASE_DIR']
        if is_nixos():
            required_env.extend(['TETRAGON_BPF_PATH', 'POLICY_FILE'])

        missing = [env for env in required_env if not os.getenv(env)]
        if missing:
            raise EnvironmentError(f"Missing required environment variables: {missing}")

    def _get_resource_limits(self):
        """Get appropriate resource limits based on environment"""
        if is_nixos():
            return {
                'max_memory_percent': 85,  # Slightly lower for NixOS
                'max_cpu_percent': 75,
                'max_disk_percent': 90,
                'max_processes': 800
            }
        return {
            'max_memory_percent': 90,
            'max_cpu_percent': 80,
            'max_disk_percent': 96,
            'max_processes': 1000
        }

    def __init__(self):
        # Base Directories, Use XDG directories

        # Validate environment first
        #self.validate_environment()

        # Initialize resource limits using the method
        self.RESOURCE_LIMITS = self._get_resource_limits()

        # Keep existing directory structure
        #self.BASE_DIR = os.getenv('PROJECT_BASE_DIR', os.path.expanduser('~/.local/share/hids'))
        self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))


        # Better Tetragon path handling
        if is_nixos():
            if not os.getenv('TETRAGON_BPF_PATH'):
                raise EnvironmentError("TETRAGON_BPF_PATH not set in NixOS environment")
            self.TETRAGON_BPF_LIB = os.getenv('TETRAGON_BPF_PATH')
        else:
            self.TETRAGON_BPF_LIB = '/nix/store/tetragon/bpf'

        # Load policy from YAML
        #policy_path = os.getenv('POLICY_FILE', os.path.join(self.BASE_DIR, 'syscallpolicy.yaml'))
        policy_path = os.getenv('POLICY_FILE', os.path.join(self.BASE_DIR, 'syscallpolicy.yaml'))

        if not os.path.exists(policy_path):
            raise FileNotFoundError(f"Policy file not found at {policy_path}")
        with open(policy_path) as f:
            self.POLICY = yaml.safe_load(f)


        self.SYSCALL_POLICY_FILE = policy_path

        self.LOG_DIR = os.path.join(os.getcwd(), 'logs')
        print("TTTTTTTTTTTTTTTTTTTTTTTTTTT" + self.LOG_DIR)
        self.TEMP_DIR = os.path.join(os.getcwd(), 'temp')
        self.ARCHIVE_DIR = os.path.join(os.getcwd(), 'archive')
        self.GRAPHS_DIR = os.path.join(os.getcwd(), 'graphs')

        # Shared resources (will be set by Manager)
        self.main_process_pid = None
        self.stop_event = None
        self.processes = None
        self.y_true = None
        self.y_pred = None
        self.performance_metrics = None

        # Process monitoring
        self.PROCESS_MONITOR = {
            'check_interval': 5,
            'max_retries': 3,
            'retry_delay': 2
        }
        # Add these to existing PROCESS_MONITOR
        self.PROCESS_MONITOR.update({
            'collection_batch_size': 1000,
            'collection_timeout': 30,
            'json_verify_timeout': 5
        })

        # Add Data Collection Stats
        self.COLLECTION_STATS = {
            'save_interval': 300,  # Save stats every 5 minutes
            'metrics_file': 'collection_metrics.json',
            'plot_metrics': True,
            'track_memory_usage': True
        }

        # SSG Configuration
        self.SSG_CONFIG = {
            'plot_interval': 10,
            'plot_size': (20, 12),
            'node_sizes': {
                'normal': 1000,
                'special': 2000
            },
            'node_colors': {
                'seen': 'lightblue',
                'unseen_syscall': 'orange',
                'unseen_arg': 'red'
            },
            'edge_alpha': 0.4,
            'node_alpha': 0.7,
            'font_sizes': {
                'node_labels': 10,
                'edge_labels': 8,
                'title': 14,
                'legend': 10
            },
            'special_nodes': ['USN', 'UAN'],
            'special_syscalls': ['open', 'stat', 'execve', 'clone'],
            'layout': 'spring',
            'dpi': 300,
            'save_format': 'png'
        }

        # Performance Monitoring Configuration
        self.PERFORMANCE_MONITORING = {
            'log_interval': 60,  # Log performance metrics every 60 seconds
            'plot_interval': 300,  # Generate plots every 5 minutes
            'max_history_size': 1000,  # Maximum number of metrics to store
            'critical_thresholds': {
                'cpu': 80,  # CPU usage threshold (%)
                'memory': 85,  # Memory usage threshold (%)
                'detection_time': 1.0  # Detection time threshold (seconds)
            }
        }

        # Add Data Storage Paths

        # Initialize DATA_STORAGE with all paths as strings
        self.DATA_STORAGE = {
            'raw_logs': os.path.join(os.getcwd(), 'data/raw_logs'),
            'processed_logs': os.path.join(os.getcwd(), 'data/processed_logs'),
            'training_data': os.path.join(os.getcwd(), 'data/training'),
            'testing_data': os.path.join(os.getcwd(), 'data/testing'),
            'archive': os.path.join(os.getcwd(), 'data/archive'),
            'models': os.path.join(self.BASE_DIR, 'models'),
            'results': os.path.join(os.getcwd(), 'results'),
            # Realtime paths
            'realtime_base': os.path.join(os.getcwd(), 'data/realtime'),
            'realtime_results': os.path.join(os.getcwd(), 'results/realtime'),
            'realtime_alerts': os.path.join(os.getcwd(), 'results/realtime/alerts'),
            'realtime_plots': os.path.join(os.getcwd(), 'results/realtime/plots'),
            'realtime_metrics': os.path.join(os.getcwd(), 'results/realtime/metrics'),
            'realtime_temp': os.path.join(os.getcwd(), 'temp/realtime')
        }

        # Use systemd for process management
        self.USE_SYSTEMD = True

        # Initialize RESOURCE_LIMITS
        self.RESOURCE_LIMITS = {
            'max_processes': 1000,
            'max_memory_percent': 90,
            'max_cpu_percent': 80,
            'max_disk_percent': 95,
            'process_timeout': 5,
            'max_log_size': 1000 * 1024 * 1024  # 1GB
        }


        # Update TEST_DATA_DIR to match the structure
        self.TEST_DATA_DIR = os.path.join(os.getcwd(), 'data', 'testing')

        # Add Syscall Collection Configuration
        self.SYSCALL_COLLECTION = {
            'batch_size': 1000,
            'json_buffer_size': 100 * 1024 * 1024,  # 10MB buffer for JSON
            'max_batch_time': 60,  # Maximum time to collect a batch (seconds)
            'min_syscalls_per_batch': 100,
            'retry_delay': 2,
            'max_retries': 5,
            'file_patterns': {
                'json': 'syscalls_raw_{process}_{timestamp}.json',
                'text': 'syscalls_processed_{process}_{timestamp}.log',
                'features': 'features_{process}_{timestamp}.npy'
            }
        }

        # Data Processing Paths
        self.DATA_PATHS = {
            'raw_json': os.path.join(self.TEMP_DIR, 'raw_json'),
            'processed_text': os.path.join(self.TEMP_DIR, 'processed_text'),
            'features': os.path.join(self.TEMP_DIR, 'features')
        }

        # Tetragon JSON Configuration
        self.TETRAGON_JSON = {
            'format_version': '1.0',
            'required_fields': ['process_tracepoint', 'process', 'binary'],
            'max_line_size': 1024 * 1024,  # 1MB max line size
            'encoding': 'utf-8'
        }



        # Initialize RESOURCE_LIMITS first
        self.RESOURCE_LIMITS = {
            'max_processes': 1000,
            'max_memory_percent': 90,
            'max_cpu_percent': 80,
            'max_disk_percent': 95,
            'process_timeout': 5,
            'max_log_size': 1000 * 1024 * 1024  # 1GB
        }


        # In your config.py
        self.DEBUG_MODE = True  # Set to True for development

        # Add graphs directory
        self.GRAPHS_DIR = os.path.join(os.getcwd(), 'graphs')

        # Data Retention Configuration
        self.DATA_RETENTION = {
            'keep_raw_logs': True,
            'keep_processed_logs': True,
            'archive_interval': 7 * 24 * 3600,  # 7 days
            'compression': True,
            'max_storage_size': 50 * 1024 * 1024 * 1024  # 50GB
        }

        # Shared resources (will be set by Manager)
        self.main_process_pid = None
        self.stop_event = None
        self.processes = None
        self.y_true = None
        self.y_pred = None
        self.performance_metrics = None

        # Export the log interval as a top-level constant
        self.PERFORMANCE_LOG_INTERVAL = self.PERFORMANCE_MONITORING['log_interval']

        # Performance Metrics Storage
        self.METRICS_HISTORY = {
            'max_samples': 10000,
            'cleanup_threshold': 0.8,
            'save_interval': 3600  # Save metrics to disk every hour
        }

        # In config.py
        self.SSG_CONFIG = {
            'plot_interval': 10000,  # Plot every 10000th interval
            'plot_size': (20, 12),  # Figure size
            'node_sizes': {
                'normal': 1000,
                'special': 2000
            },
            'node_colors': {
                'seen': 'lightblue',
                'unseen_syscall': 'orange',
                'unseen_arg': 'red'
            },
            'edge_alpha': 0.4,
            'node_alpha': 0.7,
            'font_sizes': {
                'node_labels': 10,
                'edge_labels': 8,
                'title': 14,
                'legend': 10
            },
            'special_nodes': ['USN', 'UAN'],
            'layout': 'spring',
            'dpi': 300,
            'save_format': 'png'
        }


        # Required directories
        self.REQUIRED_DIRS = [
            self.LOG_DIR,
            self.TEMP_DIR,
            self.ARCHIVE_DIR,
            self.GRAPHS_DIR
        ]

        # Graph Storage Settings
        self.GRAPH_STORAGE = {
            'max_stored_graphs': 10000,
            'cleanup_threshold': 0.9,
            'compression': True,
            'archive_interval': 24 * 3600
        }

        # Logging Configuration
        self.LOG_LEVEL = 'DEBUG'
        self.LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
        self.LOG_FILE = os.path.join(self.LOG_DIR, 'hids.log')
        self.SYSCALL_LOG_FILE = os.path.join(self.LOG_DIR, 'syscalls.log')
        self.TEMP_LOG = os.path.join(self.TEMP_DIR, 'temp.log')


        self.MODEL_FILE = os.path.join(self.BASE_DIR,'models','final', 'best_autoencoder_model.pth')
        self.SCALER_FILE = os.path.join(self.BASE_DIR, 'models','final','scaler.pkl')
        self.THRESHOLD_FILE = os.path.join(self.BASE_DIR,'models','final', 'thresholds.npy')
        self.PROCESSED_LOG_TEMPLATE = 'syscalls_processed_{}.log'

        # Tetragon Configuration
        #self.TETRAGON_BPF_LIB = '/home/sivaprasad/Downloads/Tatraon/tetragon-v1.1.0-amd64/usr/local/lib/tetragon/bpf/'
        #self.SYSCALL_POLICY_FILE = os.path.join(self.BASE_DIR, 'syscallpolicy.yaml')
        self.TETRAGON_CHECK_INTERVAL = 20
        self.TETRAGON_RESTART_DELAY = 5
        self.TETRAGON_MAX_RETRIES = 3
        self.TETRAGON_LOG_LEVEL = 'info'

        self.TEST_DATA_DIR = os.path.join(self.BASE_DIR, 'data', 'testing')
        #self.TEST_DATA_DIR = '/home/sivaprasad/PycharmProjects/HIDS_Modules/'


        # System Configuration
        self.MAX_RETRIES = 3
        self.COLLECTION_BACKOFF = 2
        self.MAX_PROCESSES = 1000
        self.PROCESS_TIMEOUT = 5
        self.QUEUE_TIMEOUT = 30


        # Time Intervals (in seconds)
        self.TIME_INTERVAL = 20
        self.TRAINING_DURATION = 600
        self.COLLECTION_INTERVAL = 0.2  # Minimum interval between collections
        self.FLUSH_INTERVAL = 60
        self.CLEANUP_INTERVAL = 3600
        self.SLEEP_INTERVAL = 0.01  # Sleep time in collection loop
        self.TETRAGON_INIT_WAIT = 3  # Wait time for Tetragon initialization
        self.MONITOR_INTERVAL = 1
        self.HEALTH_CHECK_INTERVAL = 60
        self.MIN_SLEEP_TIME = 0.01  # Minimum sleep duration
        self.MAX_COLLECTION_ATTEMPTS = 100  # Maximum collection attempts per duration
        self.COLLECTION_TIMEOUT = 30  # Maximum time for single collection

        self.SSG_FEATURES = {
            'min_syscalls_per_window': 50000,
            'expected_feature_dim': 11,  # Fixed feature dimension
            'batch_size': 64,
            'feature_types': [
                'usi',  # Unseen Syscall Influence
                'uai',  # Unseen Argument Influence
                'graph_size',  # Graph Size
                'node_count',  # Node Count
                'edge_count',  # Edge Count
                'avg_degree',  # Average Degree
                'context_influence',  # Context Influence
                'frequency_increase',  # Frequency Increase
                'unique_syscalls',  # Unique Syscalls Count
                'unique_args',  # Unique Arguments Count
                'syscall_entropy'  # Syscall Entropy
            ],
            'edge_weight_threshold': 0.1,
            'max_edges_per_node': 500,
            'clustering_threshold': 0.3
        }

        # Update Feature Extraction settings
        self.FEATURE_EXTRACTION = {
            'max_args': 5,
            'feature_dimension': 11,
            'hash_mod': 100,
            'window_size': 64,  # Match batch size
            'min_samples': 256,  # 4 * batch_size
            'stride_ratio': 0.25,
            'min_window_ratio': 0.5,
            'max_windows_per_batch': 1000
        }

        # Instead, consolidate under MODEL_CONFIG:
        self.MODEL_CONFIG = {
            'training': {
                'batch_size': 64,
                'learning_rate': 1e-5,
                'weight_decay': 1e-6,
                'num_epochs': 400,
                'early_stopping_patience': 30,
                'validation_split': 0.2,
                'min_epochs': 100,
                'warmup_epochs': 5,
                'validation_frequency':1
            },
            'scheduler': {
                'T_0': 50,
                'T_mult': 1,
                'eta_min': 1e-6,
                'warmup_factor': 0.1
            },
            'architecture': {
                'encoder_dims': [128, 64, 32],
                'dropout_rate': 0.3,
                'batch_norm': True,
                'leaky_relu_slope': 0.2
            },
            'thresholds': {
                'trained_percentile': 95,
                'unseen_multiplier': 1.2,
                'min_samples': 1000,
                'threshold_multiplier': 1.5
            }
        }

        #self.config.NUM_EPOCHS = 200

        # Training Configuration
        self.BATCH_SIZE = self.MODEL_CONFIG['training']['batch_size']
        self.MIN_BATCH_SIZE = 32
        self.MAX_BATCH_SIZE = 128
        self.VALIDATION_SPLIT = self.MODEL_CONFIG['training']['validation_split']
        self.MIN_SAMPLES_REQUIRED = self.BATCH_SIZE * 4
        self.COLLECTION_TIMEOUT = 300



        self.TRAINING_CONFIG = {
                'checkpoints_dir': os.path.join(self.BASE_DIR, 'checkpoints'),
                'model_save_dir': os.path.join(self.BASE_DIR, 'models'),
                'results_dir': os.path.join(self.BASE_DIR, 'results'),
                'plots_dir': os.path.join(self.BASE_DIR, 'plots'),
        }

        # Updated REALTIME_CONFIG
        self.REALTIME_CONFIG = {
            'dynamic_window_size': 50,
            'plot_interval': 60,
            'metrics_save_interval': 300,
            'alert_thresholds': {
                'anomaly_rate': 0.3,
                'consecutive_anomalies': 5,
                'mse_multiplier': 2.0
            },
            'process_timeout': 5,
            'process_check_interval': 60,
            'minimum_syscalls': self.MIN_SAMPLES_REQUIRED  # Use the already defined value
        }

        # Threshold Parameters
        self.TRAINED_PERCENTILE = 95
        self.UNSEEN_MULTIPLIER = 1.2
        self.THRESHOLD_MULTIPLIER = 1.2


        # Threshold Parameters
        self.TRAINED_PERCENTILE = 90
        self.UNSEEN_MULTIPLIER = 1.2
        self.DYNAMIC_WINDOW_SIZE = 50
        self.THRESHOLD_MULTIPLIER = 1.5
        self.MIN_SAMPLES_FOR_THRESHOLD = 10000

        # self.REALTIME_CONFIG = {
        #     'dynamic_window_size': 50,
        #     'plot_interval': 10,  # Plot every 10 intervals
        #     'metrics_save_interval': 300,  # 5 minutes
        #     'alert_thresholds': {
        #         'anomaly_rate': 0.3,
        #         'consecutive_anomalies': 5,
        #         'mse_multiplier': 2.0
        #     },
        #     'process_timeout': 5
        # }

        process_paths = self.get_process_paths()
        self.TRAINING_PROCESSES = [
            process_paths['firefox']
        ]
        self.TESTING_PROCESSES = [
            process_paths['teams']
        ]

        # # Process Lists
        # self.TRAINING_PROCESSES = [
        #     "/opt/google/chrome/chrome"
        #     #"/usr/share/teams/teams"
        # ]
        # self.TESTING_PROCESSES = [
        #     #"/usr/lib/firefox/firefox",
        #     "/usr/bin/python3"
        #     #"/usr/lib/virtualbox/VirtualBoxVM --comment TestVM1 --startvm 39985ac2-e75e-43a3-b6e7-ce2bfecb9c1c --no-startvm-errormsgbox",
        #     #"/usr/lib/virtualbox/VirtualBoxVM --comment TestVM2 --startvm 01dd3de2-4548-469a-935f-ba16f828c415 --no-startvm-errormsgbox"
        #
        # ] + self.TRAINING_PROCESSES

        # self.TESTING_PROCESSES = [
        # "/usr/bin/python3",
        # #"/usr/bin/VirtualBoxVM"
        # "/usr/bin/atom"
        #
        # #"/usr/lib/virtualbox/VirtualBoxVM --comment TestVM1 --startvm 39985ac2-e75e-43a3-b6e7-ce2bfecb9c1c --no-startvm-errormsgbox"
        # #"/opt/google/chrome/chrome",
        # #"/usr/share/teams/teams"
        # ]

        # # In config.py - Update only this list
        # self.TESTING_PROCESSES = [
        #     #"/home/sivaprasad/Downloads/IDS_Test/Slips_IDS/StratosphereLinuxIPS/slips/bin/python3",
        #     #"/usr/bin/atom",
        #     #"/usr/bin/python3",
        #     #"/usr/bin/firefox"
        #     "/usr/share/teams/teams"
        # ]



        # Resource Management
        self.RESOURCE_LIMITS = {
            'max_memory_percent': 90,
            'max_cpu_percent': 80,
            'max_disk_percent': 96,
            'max_processes': 1000
        }
        self.MAX_MEMORY_PERCENT = self.RESOURCE_LIMITS['max_memory_percent']
        self.MAX_CPU_PERCENT = self.RESOURCE_LIMITS['max_cpu_percent']
        self.MAX_LOG_SIZE = 1000 * 1024 * 1024  # 1GB
        self.LOG_ROTATION_COUNT = 5
        self.MEMORY_CHECK_INTERVAL = 300
        self.RESOURCE_WARNING_THRESHOLD = 0.8

        # Performance Monitoring
        self.METRICS_INTERVAL = 60
        self.METRICS_HISTORY_SIZE = 1000

        # Anomaly Detection
        self.ANOMALY_PARAMS = {
            'base_threshold_percentile': 99,
            'dynamic_adjustment_factor': 1.2,
            'min_samples_for_threshold': 100,
            'score_window_size': 50
        }

        # Process Monitoring
        self.PROCESS_RESTART_DELAY = 5
        self.MAX_RESTART_ATTEMPTS = 3

        # Debug Configuration
        self.DEBUG_MODE = False
        self.VERBOSE_LOGGING = True
        self.PROFILING_ENABLED = False

        # Error Handling
        self.ERROR_BACKOFF_BASE = 2
        self.ERROR_BACKOFF_MAX = 30
        self.MAX_ERROR_RETRIES = 3

        # Visualization
        self.PLOT_DPI = 300
        self.MAX_GRAPH_NODES = 1000
        self.GRAPH_LAYOUT = 'kamada_kawai'

        # Initialize directories and verify configuration
        self._initialize_directories()
        self._verify_configuration()

    # def _initialize_directories(self):
    #     """Initialize all required directories with proper permissions"""
    #     try:
    #         for directory in self.REQUIRED_DIRS:
    #             os.makedirs(directory, mode=0o755, exist_ok=True)
    #             os.system(f"sudo chown -R {os.getenv('USER')}:{os.getenv('USER')} {directory}")
    #             if not os.access(directory, os.W_OK):
    #                 logging.error(f"No write permission for directory: {directory}")
    #     except Exception as e:
    #         logging.error(f"Error initializing directories: {e}")

    # def _get_resource_limits(self):
    #     """Get appropriate resource limits based on environment"""
    #     if is_nixos():
    #         return {
    #             'max_memory_percent': 85,  # Slightly lower for NixOS
    #             'max_cpu_percent': 75,
    #             'max_disk_percent': 90,
    #             'max_processes': 800
    #         }
    #     return self.RESOURCE_LIMITS

    def _initialize_directories(self):
        """Initialize all required directories with proper permissions"""
        try:
            for directory in self.REQUIRED_DIRS:
                os.makedirs(directory, mode=0o755, exist_ok=True)
                if not os.access(directory, os.W_OK):
                    logging.error(f"No write permission for directory: {directory}")
        except Exception as e:
            logging.error(f"Error initializing directories: {e}")

    def _verify_configuration(self):
        """Verify critical paths and configuration values"""
        try:

            # Verify all critical directories
            for directory in self.REQUIRED_DIRS:
                if not os.path.exists(directory):
                    try:
                        os.makedirs(directory, mode=0o755, exist_ok=True)
                        os.system(f"sudo chown -R {os.getenv('USER')}:{os.getenv('USER')} {directory}")
                        logging.info(f"Created directory at: {directory}")
                    except Exception as e:
                        logging.error(f"Failed to create directory at {directory}: {e}")

            # Verify data storage paths
            for storage_type, path in self.DATA_STORAGE.items():
                if not os.path.exists(path):
                    try:
                        os.makedirs(path, mode=0o755, exist_ok=True)
                        os.system(f"sudo chown -R {os.getenv('USER')}:{os.getenv('USER')} {path}")
                        logging.info(f"Created {storage_type} directory at: {path}")
                    except Exception as e:
                        logging.error(f"Failed to create {storage_type} directory at {path}: {e}")

            # Verify all critical directories
            critical_paths = [
                (self.LOG_DIR, "logs directory"),
                (self.TEMP_DIR, "temp directory"),
                (self.ARCHIVE_DIR, "archive directory"),
                (self.GRAPHS_DIR, "graphs directory")
            ]

            # Verify all critical directories
            for directory in self.REQUIRED_DIRS:
                if not os.path.exists(directory):
                    try:
                        os.makedirs(directory, mode=0o755, exist_ok=True)
                        os.system(f"sudo chown -R {os.getenv('USER')}:{os.getenv('USER')} {directory}")
                        logging.info(f"Created directory at: {directory}")
                    except Exception as e:
                        logging.error(f"Failed to create directory at {directory}: {e}")

            # Verify data storage paths
            for storage_type, path in self.DATA_STORAGE.items():
                if not os.path.exists(path):
                    try:
                        os.makedirs(path, mode=0o755, exist_ok=True)
                        os.system(f"sudo chown -R {os.getenv('USER')}:{os.getenv('USER')} {path}")
                        logging.info(f"Created {storage_type} directory at: {path}")
                    except Exception as e:
                        logging.error(f"Failed to create {storage_type} directory at {path}: {e}")



            # # Verify all critical directories
            # for directory in self.REQUIRED_DIRS:
            #     if not os.path.exists(directory):
            #         try:
            #             os.makedirs(directory, mode=0o755, exist_ok=True)
            #             os.system(f"sudo chown -R sivaprasad:sivaprasad {directory}")
            #             logging.info(f"Created directory at: {directory}")
            #         except Exception as e:
            #             logging.error(f"Failed to create directory at {directory}: {e}")



            # Add verification for GRAPHS_DIR
            if not os.access(self.GRAPHS_DIR, os.W_OK):
                logging.error(f"No write permission for graphs directory: {self.GRAPHS_DIR}")

            for path, name in critical_paths:
                if not os.path.exists(path):
                    try:
                        os.makedirs(path, mode=0o755, exist_ok=True)
                        os.system(f"sudo chown -R sivaprasad:sivaprasad {path}")
                        logging.info(f"Created {name} at: {path}")
                    except Exception as e:
                        logging.error(f"Failed to create {name} at {path}: {e}")

            # Verify Tetragon paths
            if not os.path.exists(self.TETRAGON_BPF_LIB):
                logging.error(f"Tetragon BPF library not found at: {self.TETRAGON_BPF_LIB}")

            if not os.path.exists(self.SYSCALL_POLICY_FILE):
                logging.error(f"Syscall policy file not found at: {self.SYSCALL_POLICY_FILE}")

            # Verify file permissions
            for file_path in [self.SYSCALL_LOG_FILE, self.TEMP_LOG]:
                directory = os.path.dirname(file_path)
                if not os.path.exists(directory):
                    os.makedirs(directory, mode=0o755, exist_ok=True)
                    os.system(f"sudo chown -R {os.getenv('USER')}:{os.getenv('USER')} {directory}")

            # Verify SSG configuration
            if self.SSG_CONFIG['plot_interval'] <= 0:
                logging.warning("SSG plot interval must be positive, setting to default (10)")
                self.SSG_CONFIG['plot_interval'] = 10

            # Verify feature dimensions
            if self.SSG_FEATURES['expected_feature_dim'] < len(self.SSG_FEATURES['feature_types']):
                logging.warning("Feature dimension is less than number of feature types")

            # Verify graph storage limits
            if self.GRAPH_STORAGE['max_stored_graphs'] < 100:
                logging.warning("Graph storage limit is very low, setting to minimum (100)")
                self.GRAPH_STORAGE['max_stored_graphs'] = 100

            # Verify configuration values
            if self.BATCH_SIZE < self.MIN_BATCH_SIZE:
                logging.warning(f"Batch size ({self.BATCH_SIZE}) is less than minimum ({self.MIN_BATCH_SIZE})")

            if self.MAX_MEMORY_PERCENT > 90:
                logging.warning("Maximum memory percentage is set very high")

            # Verify queue timeout
            if not hasattr(self, 'QUEUE_TIMEOUT'):
                self.QUEUE_TIMEOUT = 30
            elif self.QUEUE_TIMEOUT <= 0:
                logging.warning("QUEUE_TIMEOUT must be positive, setting to default 30 seconds")
                self.QUEUE_TIMEOUT = 30

        except Exception as e:
            logging.error(f"Error in configuration verification: {e}")



# Create global config instance
config = Config()

# Export commonly used variables
BASE_DIR = config.BASE_DIR
MODEL_FILE = config.MODEL_FILE
SCALER_FILE = config.SCALER_FILE
THRESHOLD_FILE = config.THRESHOLD_FILE
PROCESSED_LOG_TEMPLATE = config.PROCESSED_LOG_TEMPLATE
SYSCALL_LOG_FILE = config.SYSCALL_LOG_FILE
TEMP_LOG = config.TEMP_LOG
LOG_DIR = config.LOG_DIR
TEMP_DIR = config.TEMP_DIR
ARCHIVE_DIR = config.ARCHIVE_DIR
GRAPHS_DIR = config.GRAPHS_DIR
SSG_CONFIG = config.SSG_CONFIG
SSG_FEATURES = config.SSG_FEATURES
GRAPH_STORAGE = config.GRAPH_STORAGE
DATA_STORAGE = config.DATA_STORAGE
DATA_RETENTION = config.DATA_RETENTION
RESOURCE_LIMITS = config.RESOURCE_LIMITS

# System configuration exports
MAX_RETRIES = config.MAX_RETRIES
COLLECTION_BACKOFF = config.COLLECTION_BACKOFF
TRAINING_PROCESSES = config.TRAINING_PROCESSES
TESTING_PROCESSES = config.TESTING_PROCESSES
FLUSH_INTERVAL = config.FLUSH_INTERVAL
MAX_LOG_SIZE = config.MAX_LOG_SIZE
LOG_ROTATION_COUNT = config.LOG_ROTATION_COUNT
CLEANUP_INTERVAL = config.CLEANUP_INTERVAL

# Resource limits
RESOURCE_LIMITS = config.RESOURCE_LIMITS

# Tetragon configuration exports
TETRAGON_BPF_LIB = config.TETRAGON_BPF_LIB
SYSCALL_POLICY_FILE = config.SYSCALL_POLICY_FILE
TETRAGON_CHECK_INTERVAL = config.TETRAGON_CHECK_INTERVAL

# Export commonly used variables
PERFORMANCE_LOG_INTERVAL = config.PERFORMANCE_LOG_INTERVAL

# Logging configuration
LOGGING_CONFIG = {
    'default_level': 'INFO',
    'file': {
        'enabled': True,
        'path': 'logs/hids.log',
        'max_bytes': 1000 * 1024 * 1024,
        'backup_count': 5,
        'level': 'DEBUG'
    },
    'console': {
        'enabled': True,
        'level': 'INFO',
        'colored': True
    }
}

# SSG Default Configurations
SSG_DEFAULTS = {
    'min_edge_weight': 0.1,
    'max_nodes': 1000,
    'plot_interval': 10,
    'special_nodes': ['USN', 'UAN'],
    'colors': {
        'normal': 'lightblue',
        'unseen': 'orange',
        'special': 'red'
    }
}
