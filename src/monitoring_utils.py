import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, TypedDict, Union, Tuple
import numpy as np
import sys
from collections import deque
import json
from datetime import datetime
import psutil
from .config import config


# Type definitions
class ResourceMetrics(TypedDict):
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_io: Dict[str, int]
    process_count: int
    timestamp: str


class AnomalyMetrics(TypedDict):
    timestamp: datetime
    process: str
    score: float
    threshold: float


class MonitoringStats(TypedDict):
    total_samples: int
    detection_time_avg: float
    cpu_usage_avg: float
    memory_usage_avg: float
    start_time: str
    end_time: str


class ResourceMonitor:
    def __init__(self):
        self.logger = logging.getLogger('resource_monitor')
        self.start_time = datetime.now()
        self.resource_history = []

    def check_resources(self) -> Dict[str, Any]:
        """Check current system resource usage"""
        try:
            current_usage = {
                'timestamp': datetime.now(),
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': psutil.disk_usage('/').percent,
                'network_io': psutil.net_io_counters()._asdict(),
                'process_count': len(psutil.pids())
            }

            self.resource_history.append(current_usage)
            return current_usage

        except Exception as e:
            self.logger.error(f"Error checking resources: {e}")
            return {}



    def get_resource_summary(self) -> Dict[str, Any]:
        """Get summary of resource usage"""
        try:
            if not self.resource_history:
                return {}

            df = pd.DataFrame(self.resource_history)

            return {
                'monitoring_duration': str(datetime.now() - self.start_time),
                'cpu_usage': {
                    'current': df['cpu_percent'].iloc[-1],
                    'average': df['cpu_percent'].mean(),
                    'max': df['cpu_percent'].max()
                },
                'memory_usage': {
                    'current': df['memory_percent'].iloc[-1],
                    'average': df['memory_percent'].mean(),
                    'max': df['memory_percent'].max()
                },
                'disk_usage': {
                    'current': df['disk_percent'].iloc[-1],
                    'average': df['disk_percent'].mean(),
                    'max': df['disk_percent'].max()
                },
                'process_count': {
                    'current': df['process_count'].iloc[-1],
                    'average': df['process_count'].mean(),
                    'max': df['process_count'].max()
                }
            }

        except Exception as e:
            self.logger.error(f"Error generating resource summary: {e}")
            return {}


class SystemMonitor:
    def __init__(self):
        """Initialize system monitor with enhanced tracking capabilities"""
        self.logger = logging.getLogger('monitoring')

        # Initialize metrics storage
        self.cpu_history: List[float] = []
        self.memory_history: List[float] = []
        self.disk_history: List[float] = []
        self.detection_times: List[float] = []
        self.anomaly_scores: List[float] = []
        self.timestamps: List[datetime] = []

        # Process-specific metrics
        self.process_metrics: Dict[str, Dict[str, List[float]]] = {}

        # Resource metrics
        self.resource_history: List[ResourceMetrics] = []

        # Initialize monitoring directories
        self._initialize_directories()

        # Start time of monitoring
        self.start_time = datetime.now()

        # Alert thresholds
        self.thresholds = {
            'cpu_percent': config.RESOURCE_LIMITS['max_cpu_percent'],
            'memory_percent': config.RESOURCE_LIMITS['max_memory_percent'],
            'disk_percent': config.RESOURCE_LIMITS['max_disk_percent']
        }

    def get_timestamps(self):
        """Return monitoring timestamps"""
        return self.timestamps if hasattr(self, 'timestamps') else []

    # def get_metrics(self) -> Dict[str, List[float]]:
    #     """Get current system metrics"""
    #     return {
    #         'cpu_usage': self.cpu_history if hasattr(self, 'cpu_history') else [],
    #         'memory_usage': self.memory_history if hasattr(self, 'memory_history') else []
    #     }

    def get_metrics(self) -> Dict[str, List[float]]:
        """Get current system metrics with validation"""
        try:
            metrics = {
                'cpu_usage': self.cpu_history if hasattr(self, 'cpu_history') else [],
                'memory_usage': self.memory_history if hasattr(self, 'memory_history') else []
            }

            # Validate metrics
            for key in metrics:
                if not metrics[key]:
                    metrics[key] = [0.0]  # Default to 0 if no data
                metrics[key] = [x for x in metrics[key] if not np.isnan(x)]  # Remove NaN values

            return metrics
        except Exception as e:
            self.logger.error(f"Error getting metrics: {e}")
            return {'cpu_usage': [0.0], 'memory_usage': [0.0]}

    def get_cpu_usage(self):
        """Return CPU usage history"""
        return self.cpu_history if hasattr(self, 'cpu_history') else []

    def get_memory_usage(self):
        """Return memory usage history"""
        return self.memory_history if hasattr(self, 'memory_history') else []

    def _initialize_directories(self) -> None:
        """Initialize required directories for monitoring data"""
        try:
            self.monitoring_dir = os.path.join(config.DATA_STORAGE['results'], 'monitoring')
            self.plots_dir = os.path.join(self.monitoring_dir, 'plots')
            self.stats_dir = os.path.join(self.monitoring_dir, 'stats')

            for directory in [self.monitoring_dir, self.plots_dir, self.stats_dir]:
                os.makedirs(directory, exist_ok=True)

        except Exception as e:
            self.logger.error(f"Error initializing directories: {e}")
            raise

    def _json_serialize(self, obj):
        """Custom JSON serializer for handling numpy and other complex types"""
        try:
            if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                                np.int16, np.int32, np.int64, np.uint8,
                                np.uint16, np.uint32, np.uint64)):
                return int(obj)
            elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, deque):
                return list(obj)
            return str(obj)
        except Exception as e:
            self.logger.error(f"Error in JSON serialization: {e}")
            return str(obj)

    def update(self, cpu_usage: float, memory_usage: float,
               detection_time: float, anomaly_score: Optional[float] = None,
               process_name: Optional[str] = None) -> None:
        """Update monitoring metrics with new measurements"""
        try:
            current_time = datetime.now()

            # Update general metrics
            self.cpu_history.append(cpu_usage)
            self.memory_history.append(memory_usage)
            self.detection_times.append(detection_time)
            self.timestamps.append(current_time)

            # Get disk usage
            disk_usage = psutil.disk_usage('/').percent
            self.disk_history.append(disk_usage)

            if anomaly_score is not None:
                self.anomaly_scores.append(anomaly_score)

            # Update process-specific metrics
            if process_name:
                if process_name not in self.process_metrics:
                    self.process_metrics[process_name] = {
                        'cpu_usage': [],
                        'memory_usage': [],
                        'detection_times': [],
                        'anomaly_scores': []
                    }
                metrics = self.process_metrics[process_name]
                metrics['cpu_usage'].append(cpu_usage)
                metrics['memory_usage'].append(memory_usage)
                metrics['detection_times'].append(detection_time)
                if anomaly_score is not None:
                    metrics['anomaly_scores'].append(anomaly_score)

            # Check resource thresholds
            self._check_resource_thresholds({
                'cpu_percent': cpu_usage,
                'memory_percent': memory_usage,
                'disk_percent': disk_usage
            })

            # Save metrics periodically
            if len(self.timestamps) % config.PERFORMANCE_MONITORING['log_interval'] == 0:
                self.save_metrics()

        except Exception as e:
            self.logger.error(f"Error updating metrics: {e}")

    def _check_resource_thresholds(self, metrics: Dict[str, float]) -> None:
        """Check if any resource metrics exceed thresholds"""
        try:
            for metric_name, value in metrics.items():
                if value > self.thresholds.get(metric_name, float('inf')):
                    self._handle_threshold_violation(metric_name, value)

        except Exception as e:
            self.logger.error(f"Error checking resource thresholds: {e}")

    def _handle_threshold_violation(self, metric_name: str, value: float) -> None:
        """Handle resource threshold violations"""
        try:
            violation_data = {
                'timestamp': datetime.now().isoformat(),
                'metric': metric_name,
                'value': value,
                'threshold': self.thresholds[metric_name],
                'system_info': self._get_system_info()
            }

            # Save violation data
            violation_file = os.path.join(
                self.stats_dir,
                f'violation_{metric_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            )

            with open(violation_file, 'w') as f:
                json.dump(violation_data, f, indent=4)

            # Log violation
            self.logger.warning(
                f"Resource threshold violation - {metric_name}: {value:.2f} "
                f"(threshold: {self.thresholds[metric_name]})"
            )

        except Exception as e:
            self.logger.error(f"Error handling threshold violation: {e}")

    def _get_system_info(self) -> Dict[str, Any]:
        """Get current system information"""
        try:
            return {
                'cpu_count': psutil.cpu_count(),
                'total_memory': psutil.virtual_memory().total,
                'disk_usage': psutil.disk_usage('/').percent,
                'load_average': psutil.getloadavg(),
                'boot_time': datetime.fromtimestamp(psutil.boot_time()).isoformat(),
                'process_count': len(psutil.pids())
            }
        except Exception as e:
            self.logger.error(f"Error getting system info: {e}")
            return {}

    def save_statistics(self) -> None:
        """Save monitoring statistics to disk"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            stats_dir = os.path.join(config.DATA_STORAGE['results'], 'monitoring', 'stats')
            os.makedirs(stats_dir, exist_ok=True)

            stats_file = os.path.join(
                stats_dir,
                f'monitoring_stats_{timestamp}.json'
            )

            stats = {
                'cpu_usage': {
                    'history': self.cpu_history,
                    'average': np.mean(self.cpu_history) if self.cpu_history else 0,
                    'max': np.max(self.cpu_history) if self.cpu_history else 0
                },
                'memory_usage': {
                    'history': self.memory_history,
                    'average': np.mean(self.memory_history) if self.memory_history else 0,
                    'max': np.max(self.memory_history) if self.memory_history else 0
                },
                'disk_usage': {
                    'history': self.disk_history,
                    'average': np.mean(self.disk_history) if self.disk_history else 0,
                    'max': np.max(self.disk_history) if self.disk_history else 0
                },
                'detection_times': {
                    'history': self.detection_times,
                    'average': np.mean(self.detection_times) if self.detection_times else 0,
                    'max': np.max(self.detection_times) if self.detection_times else 0
                },
                'process_metrics': self.process_metrics,
                'timestamp': timestamp,
                'monitoring_duration': str(datetime.now() - self.start_time),
                'system_info': {
                    'cpu_count': psutil.cpu_count(),
                    'total_memory': psutil.virtual_memory().total,
                    'platform': sys.platform,
                    'python_version': sys.version
                }
            }

            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=4, default=self._json_serialize)

            self.logger.info(f"Statistics saved to: {stats_file}")

        except Exception as e:
            self.logger.error(f"Error saving statistics: {e}")

    def save_metrics(self) -> None:
        """Save current monitoring metrics to disk"""
        try:
            current_time = datetime.now()
            metrics_data = {
                'timestamp': current_time.isoformat(),
                'monitoring_duration': str(current_time - self.start_time),
                'general_metrics': {
                    'cpu_history': self.cpu_history,
                    'memory_history': self.memory_history,
                    'disk_history': self.disk_history,
                    'detection_times': self.detection_times,
                    'anomaly_scores': self.anomaly_scores,
                    'timestamps': [t.isoformat() for t in self.timestamps]
                },
                'process_metrics': {
                    process: {
                        metric: values
                        for metric, values in metrics.items()
                    }
                    for process, metrics in self.process_metrics.items()
                },
                'system_info': self._get_system_info()
            }

            # Save metrics
            metrics_file = os.path.join(
                self.stats_dir,
                f'metrics_{current_time.strftime("%Y%m%d_%H%M%S")}.json'
            )

            with open(metrics_file, 'w') as f:
                json.dump(metrics_data, f, indent=4)

            self.logger.info(f"Metrics saved to: {metrics_file}")

        except Exception as e:
            self.logger.error(f"Error saving metrics: {e}")

    def plot_metrics(self, save_path: Optional[str] = None) -> None:
        """Generate and save monitoring metrics visualization"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if save_path is None:
                save_path = os.path.join(self.plots_dir, f'monitoring_metrics_{timestamp}.png')

            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))

            # CPU Usage Plot
            ax1.plot(self.timestamps, self.cpu_history, 'b-', label='CPU Usage')
            ax1.axhline(
                y=self.thresholds['cpu_percent'],
                color='r',
                linestyle='--',
                label='CPU Threshold'
            )
            ax1.set_title('CPU Usage Over Time')
            ax1.set_ylabel('CPU %')
            ax1.legend()
            ax1.grid(True)

            # Memory Usage Plot
            ax2.plot(self.timestamps, self.memory_history, 'g-', label='Memory Usage')
            ax2.axhline(
                y=self.thresholds['memory_percent'],
                color='r',
                linestyle='--',
                label='Memory Threshold'
            )
            ax2.set_title('Memory Usage Over Time')
            ax2.set_ylabel('Memory %')
            ax2.legend()
            ax2.grid(True)

            # Disk Usage Plot
            ax3.plot(self.timestamps, self.disk_history, 'm-', label='Disk Usage')
            ax3.axhline(
                y=self.thresholds['disk_percent'],
                color='r',
                linestyle='--',
                label='Disk Threshold'
            )
            ax3.set_title('Disk Usage Over Time')
            ax3.set_ylabel('Disk %')
            ax3.legend()
            ax3.grid(True)

            # Detection Time Plot
            ax4.plot(self.timestamps, self.detection_times, 'r-', label='Detection Time')
            ax4.set_title('Detection Time Over Time')
            ax4.set_ylabel('Time (seconds)')
            ax4.legend()
            ax4.grid(True)

            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

            self.logger.info(f"Metrics plot saved to: {save_path}")

        except Exception as e:
            self.logger.error(f"Error plotting metrics: {e}")

    def generate_monitoring_report(self, save_path: Optional[str] = None) -> None:
        """Generate comprehensive HTML monitoring report"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if save_path is None:
                save_path = os.path.join(self.stats_dir, f'monitoring_report_{timestamp}.html')

            # Calculate statistics
            monitoring_stats = self._calculate_monitoring_stats()

            # Generate plots
            plots_data = self._generate_report_plots()

            # Generate HTML report
            html_content = self._generate_html_report(monitoring_stats, plots_data)

            with open(save_path, 'w') as f:
                f.write(html_content)

            self.logger.info(f"Monitoring report saved to: {save_path}")

        except Exception as e:
            self.logger.error(f"Error generating monitoring report: {e}")

    def _calculate_monitoring_stats(self) -> MonitoringStats:
        """Calculate monitoring statistics"""
        return MonitoringStats(
            total_samples=len(self.timestamps),
            detection_time_avg=np.mean(self.detection_times) if self.detection_times else 0,
            cpu_usage_avg=np.mean(self.cpu_history) if self.cpu_history else 0,
            memory_usage_avg=np.mean(self.memory_history) if self.memory_history else 0,
            start_time=self.start_time.isoformat(),
            end_time=datetime.now().isoformat()
        )

    def _generate_report_plots(self) -> Dict[str, str]:
        """Generate plots for the report"""
        plots = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Generate and save plots
        for plot_type in ['metrics', 'process_metrics', 'resource_usage']:
            plot_path = os.path.join(self.plots_dir, f'{plot_type}_{timestamp}.png')
            if plot_type == 'metrics':
                self.plot_metrics(plot_path)
            elif plot_type == 'process_metrics':
                self._plot_process_metrics(plot_path)
            elif plot_type == 'resource_usage':
                self._plot_resource_usage(plot_path)
            plots[plot_type] = plot_path

        return plots

    def _plot_process_metrics(self, save_path: str) -> None:
        """Plot process-specific metrics"""
        try:
            if not self.process_metrics:
                return

            num_processes = len(self.process_metrics)
            fig, axes = plt.subplots(num_processes, 1, figsize=(15, 5 * num_processes))
            if num_processes == 1:
                axes = [axes]

            for ax, (process_name, metrics) in zip(axes, self.process_metrics.items()):
                ax.plot(self.timestamps[:len(metrics['cpu_usage'])],
                        metrics['cpu_usage'],
                        label='CPU Usage')
                ax.plot(self.timestamps[:len(metrics['memory_usage'])],
                        metrics['memory_usage'],
                        label='Memory Usage')
                ax.set_title(f'Resource Usage - {process_name}')
                ax.set_ylabel('Usage %')
                ax.legend()
                ax.grid(True)

            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            self.logger.error(f"Error plotting process metrics: {e}")

    def _plot_resource_usage(self, save_path: str) -> None:
        """Plot detailed resource usage"""
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

            # Resource usage distribution
            data = {
                'CPU Usage': self.cpu_history,
                'Memory Usage': self.memory_history,
                'Disk Usage': self.disk_history
            }
            df = pd.DataFrame(data)
            sns.boxplot(data=df, ax=ax1)
            ax1.set_title('Resource Usage Distribution')
            ax1.set_ylabel('Usage %')

            # Detection time distribution
            sns.histplot(self.detection_times, ax=ax2, bins=30)
            ax2.set_title('Detection Time Distribution')
            ax2.set_xlabel('Time (seconds)')

            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            self.logger.error(f"Error plotting resource usage: {e}")

    def _generate_html_report(self, stats: MonitoringStats, plots: Dict[str, str]) -> str:
        """Generate HTML content for monitoring report"""
        html_template = """
                        <html>
                        <head>
                            <style>
                                body {
                                    font-family: Arial, sans-serif;
                                    margin: 20px;
                                    background-color: #f5f5f5;
                                }
                                .container {
                                    max-width: 1200px;
                                    margin: 0 auto;
                                    background-color: white;
                                    padding: 20px;
                                    border-radius: 8px;
                                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                                }
                                h2, h3 {
                                    color: #2c3e50;
                                    border-bottom: 2px solid #3498db;
                                    padding-bottom: 10px;
                                }
                                table {
                                    width: 100%;
                                    border-collapse: collapse;
                                    margin: 20px 0;
                                }
                                th, td {
                                    padding: 12px;
                                    text-align: left;
                                    border: 1px solid #ddd;
                                }
                                th {
                                    background-color: #3498db;
                                    color: white;
                                }
                                tr:nth-child(even) {
                                    background-color: #f9f9f9;
                                }
                                .metric {
                                    font-weight: bold;
                                    color: #2980b9;
                                }
                                .warning {
                                    color: #e74c3c;
                                }
                                .success {
                                    color: #27ae60;
                                }
                                img {
                                    max-width: 100%;
                                    height: auto;
                                    margin: 20px 0;
                                }
                                .plot-container {
                                    margin: 20px 0;
                                    padding: 10px;
                                    background-color: white;
                                    border-radius: 4px;
                                    box-shadow: 0 0 5px rgba(0,0,0,0.1);
                                }
                                .timestamp {
                                    color: #7f8c8d;
                                    font-style: italic;
                                }
                            </style>
                        </head>
                        <body>
                            <div class="container">
                                <h2>System Monitoring Report</h2>
                                <p class="timestamp">Generated: {timestamp}</p>

                                <h3>Monitoring Summary</h3>
                                <table>
                                    <tr>
                                        <th>Metric</th>
                                        <th>Value</th>
                                    </tr>
                                    <tr>
                                        <td>Total Samples</td>
                                        <td class="metric">{total_samples}</td>
                                    </tr>
                                    <tr>
                                        <td>Average CPU Usage</td>
                                        <td class="metric">{cpu_usage_avg:.2f}%</td>
                                    </tr>
                                    <tr>
                                        <td>Average Memory Usage</td>
                                        <td class="metric">{memory_usage_avg:.2f}%</td>
                                    </tr>
                                    <tr>
                                        <td>Average Detection Time</td>
                                        <td class="metric">{detection_time_avg:.4f}s</td>
                                    </tr>
                                    <tr>
                                        <td>Monitoring Duration</td>
                                        <td class="metric">{monitoring_duration}</td>
                                    </tr>
                                </table>

                                <h3>System Resource Plots</h3>
                                <div class="plot-container">
                                    <h4>Resource Usage Over Time</h4>
                                    <img src="{metrics_plot}" alt="Resource Metrics">
                                </div>

                                <div class="plot-container">
                                    <h4>Process-Specific Metrics</h4>
                                    <img src="{process_plot}" alt="Process Metrics">
                                </div>

                                <div class="plot-container">
                                    <h4>Resource Usage Distribution</h4>
                                    <img src="{resource_plot}" alt="Resource Usage">
                                </div>

                                <h3>Process Details</h3>
                                {process_tables}
                            </div>
                        </body>
                        </html>
                        """

        # Generate process-specific tables
        process_tables = ""
        for process_name, metrics in self.process_metrics.items():
            process_tables += f"""
                                            <h4>{process_name}</h4>
                                            <table>
                                                <tr>
                                                    <th>Metric</th>
                                                    <th>Average</th>
                                                    <th>Maximum</th>
                                                </tr>
                                                <tr>
                                                    <td>CPU Usage</td>
                                                    <td class="metric">{np.mean(metrics['cpu_usage']):.2f}%</td>
                                                    <td class="metric">{np.max(metrics['cpu_usage']):.2f}%</td>
                                                </tr>
                                                <tr>
                                                    <td>Memory Usage</td>
                                                    <td class="metric">{np.mean(metrics['memory_usage']):.2f}%</td>
                                                    <td class="metric">{np.max(metrics['memory_usage']):.2f}%</td>
                                                </tr>
                                                <tr>
                                                    <td>Detection Time</td>
                                                    <td class="metric">{np.mean(metrics['detection_times']):.4f}s</td>
                                                    <td class="metric">{np.max(metrics['detection_times']):.4f}s</td>
                                                </tr>
                                            </table>
                                        """

        # Calculate monitoring duration
        monitoring_duration = str(datetime.now() - datetime.fromisoformat(stats['start_time']))

        return html_template.format(
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            total_samples=stats['total_samples'],
            cpu_usage_avg=stats['cpu_usage_avg'],
            memory_usage_avg=stats['memory_usage_avg'],
            detection_time_avg=stats['detection_time_avg'],
            monitoring_duration=monitoring_duration,
            metrics_plot=plots.get('metrics', ''),
            process_plot=plots.get('process_metrics', ''),
            resource_plot=plots.get('resource_usage', ''),
            process_tables=process_tables
        )


def cleanup_old_data(self, days_to_keep: int = 7) -> None:
    """Cleanup old monitoring data"""
    try:
        cleanup_time = datetime.now() - timedelta(days=days_to_keep)

        # Cleanup old plots
        for file_path in Path(self.plots_dir).glob('*.png'):
            if datetime.fromtimestamp(file_path.stat().st_mtime) < cleanup_time:
                os.remove(file_path)
                self.logger.debug(f"Removed old plot: {file_path}")

        # Cleanup old stats
        for file_path in Path(self.stats_dir).glob('*.json'):
            if datetime.fromtimestamp(file_path.stat().st_mtime) < cleanup_time:
                os.remove(file_path)
                self.logger.debug(f"Removed old stats: {file_path}")

        self.logger.info("Completed cleanup of old monitoring data")

    except Exception as e:
        self.logger.error(f"Error cleaning up old data: {e}")


class AnomalyVisualizer:
    def __init__(self):
        self.logger = logging.getLogger('visualization')
        self.anomaly_data: List[AnomalyMetrics] = []

        # Initialize visualization directories
        self.vis_dir = os.path.join(config.DATA_STORAGE['results'], 'visualizations')
        os.makedirs(self.vis_dir, exist_ok=True)

    def add_anomaly(self, timestamp: datetime, process_name: str,
                    score: float, threshold: float) -> None:
        """Add anomaly data point"""
        try:
            self.anomaly_data.append(AnomalyMetrics(
                timestamp=timestamp,
                process=process_name,
                score=score,
                threshold=threshold
            ))

            # Save data periodically
            if len(self.anomaly_data) % 100 == 0:
                self.save_anomaly_data()

        except Exception as e:
            self.logger.error(f"Error adding anomaly data: {e}")

    def save_anomaly_data(self) -> None:
        """Save anomaly data to disk"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            data_file = os.path.join(self.vis_dir, f'anomaly_data_{timestamp}.json')

            serializable_data = [
                {
                    **d,
                    'timestamp': d['timestamp'].isoformat()
                }
                for d in self.anomaly_data
            ]

            with open(data_file, 'w') as f:
                json.dump(serializable_data, f, indent=4)

            self.logger.info(f"Anomaly data saved to: {data_file}")

        except Exception as e:
            self.logger.error(f"Error saving anomaly data: {e}")


def initialize_monitoring() -> Tuple[SystemMonitor, AnomalyVisualizer]:
    """Initialize all monitoring components"""
    try:
        system_monitor = SystemMonitor()
        anomaly_visualizer = AnomalyVisualizer()

        return system_monitor, anomaly_visualizer

    except Exception as e:
        logging.error(f"Error initializing monitoring: {e}")
        return None, None
