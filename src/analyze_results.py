import os
import sys
import numpy as np
import pandas as pd
import psutil
import traceback
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score,
    roc_curve,
    auc,
    precision_recall_curve
)
import json
import logging
from collections import defaultdict

from .config import config
from .logging_setup import get_logger
from .monitoring_utils import SystemMonitor, AnomalyVisualizer


class ResultAnalyzer:
    def __init__(self):
        """Initialize result analyzer"""
        self.logger = get_logger(__name__)

        # Initialize storage paths
        self.results_dir = os.path.join(config.DATA_STORAGE['results'], 'analysis')
        self.plots_dir = os.path.join(self.results_dir, 'plots')
        self.metrics_dir = os.path.join(self.results_dir, 'metrics')
        self.reports_dir = os.path.join(self.results_dir, 'reports')

        # Create directories
        for directory in [self.results_dir, self.plots_dir, self.metrics_dir, self.reports_dir]:
            os.makedirs(directory, exist_ok=True)

        # Initialize result storage
        self.results = {
            'test': defaultdict(dict),
            'realtime': defaultdict(dict)
        }

        # Initialize metric storage
        self.metrics = defaultdict(dict)

    def load_results(self, mode: str = 'test') -> bool:
        """Load results from files"""
        try:
            self.logger.info(f"Loading {mode} results...")
            results_path = os.path.join(config.DATA_STORAGE['results'], mode)

            # Load predictions and true labels
            predictions_file = list(Path(results_path).glob('test_predictions_*.npy'))
            labels_file = list(Path(results_path).glob('test_labels_*.npy'))

            if not predictions_file or not labels_file:
                self.logger.error("Results files not found")
                return False

            # Use most recent results
            predictions_file = sorted(predictions_file)[-1]
            labels_file = sorted(labels_file)[-1]

            self.results[mode]['predictions'] = np.load(predictions_file)
            self.results[mode]['true_labels'] = np.load(labels_file)

            # Load MSE scores if available
            mse_file = list(Path(results_path).glob('test_mse_*.npy'))
            if mse_file:
                self.results[mode]['mse_scores'] = np.load(sorted(mse_file)[-1])

            # Load process-specific results
            for process in config.TESTING_PROCESSES:
                process_file = list(Path(results_path).glob(f'process_{process.replace("/", "_")}*.npz'))
                if process_file:
                    data = np.load(sorted(process_file)[-1])
                    self.results[mode]['processes'][process] = {
                        'predictions': data['predictions'],
                        'true_labels': data['true_labels'],
                        'mse_scores': data['mse_scores']
                    }

            self.logger.info(f"Successfully loaded {mode} results")
            return True

        except Exception as e:
            self.logger.error(f"Error loading results: {e}")
            return False

    def calculate_metrics(self, mode: str = 'test') -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        try:
            results = self.results[mode]
            metrics = {}

            # Overall metrics
            y_true = results['true_labels']
            y_pred = results['predictions']

            metrics['overall'] = self._calculate_basic_metrics(y_true, y_pred)

            if 'mse_scores' in results:
                metrics['overall'].update(self._calculate_threshold_metrics(
                    y_true, results['mse_scores']
                ))

            # Process-specific metrics
            metrics['processes'] = {}
            for process, process_results in results.get('processes', {}).items():
                process_metrics = self._calculate_basic_metrics(
                    process_results['true_labels'],
                    process_results['predictions']
                )

                if 'mse_scores' in process_results:
                    process_metrics.update(self._calculate_threshold_metrics(
                        process_results['true_labels'],
                        process_results['mse_scores']
                    ))

                metrics['processes'][process] = process_metrics

            self.metrics[mode] = metrics
            return metrics

        except Exception as e:
            self.logger.error(f"Error calculating metrics: {e}")
            return {}



    # def _calculate_basic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    #     """Calculate basic classification metrics with proper handling for trained/untrained processes"""
    #     metrics = {}
    #     try:
    #         # Basic metrics
    #         metrics['accuracy'] = accuracy_score(y_true, y_pred)
    #
    #         # Calculate confusion matrix first
    #         cm = confusion_matrix(y_true, y_pred)
    #         tn, fp, fn, tp = cm.ravel()
    #
    #         # Store raw confusion matrix values
    #         metrics['true_negatives'] = int(tn)
    #         metrics['false_positives'] = int(fp)
    #         metrics['false_negatives'] = int(fn)
    #         metrics['true_positives'] = int(tp)
    #
    #         # Calculate precision with handling for trained/untrained processes
    #         if fp + tp > 0:  # If we have any positive predictions
    #             metrics['precision'] = float(tp / (tp + fp))
    #         else:
    #             metrics['precision'] = 0.0
    #
    #         # Calculate recall with handling for trained/untrained processes
    #         if tp + fn > 0:  # If we have any actual positives
    #             metrics['recall'] = float(tp / (tp + fn))
    #         else:
    #             metrics['recall'] = 0.0
    #
    #         # Calculate F1 score only if precision and recall are valid
    #         if metrics['precision'] + metrics['recall'] > 0:
    #             metrics['f1_score'] = float(2 * (metrics['precision'] * metrics['recall']) /
    #                                         (metrics['precision'] + metrics['recall']))
    #         else:
    #             metrics['f1_score'] = 0.0
    #
    #         # Calculate additional metrics with proper handling
    #         metrics['specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    #         metrics['false_positive_rate'] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    #         metrics['false_negative_rate'] = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    #
    #         # Add new validation metrics
    #         total_samples = tn + fp + fn + tp
    #         if total_samples > 0:
    #             metrics['total_accuracy'] = float((tp + tn) / total_samples)
    #             metrics['balanced_accuracy'] = float((tp / (tp + fn) if (tp + fn) > 0 else 0) +
    #                                                  (tn / (tn + fp) if (tn + fp) > 0 else 0)) / 2
    #         else:
    #             metrics['total_accuracy'] = 0.0
    #             metrics['balanced_accuracy'] = 0.0
    #
    #         # Handle zero metrics gracefully
    #         for key in metrics:
    #             if np.isnan(metrics[key]) or np.isinf(metrics[key]):
    #                 metrics[key] = 0.0
    #
    #         return metrics
    #
    #     except Exception as e:
    #         self.logger.error(f"Error calculating basic metrics: {e}")
    #         # Return default metrics
    #         return {
    #             'accuracy': 0.0,
    #             'precision': 0.0,
    #             'recall': 0.0,
    #             'f1_score': 0.0,
    #             'specificity': 0.0,
    #             'true_negatives': 0,
    #             'false_positives': 0,
    #             'false_negatives': 0,
    #             'true_positives': 0,
    #             'false_positive_rate': 0.0,
    #             'false_negative_rate': 0.0,
    #             'total_accuracy': 0.0,
    #             'balanced_accuracy': 0.0
    #         }

    # def _calculate_basic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    #     """Calculate basic classification metrics"""
    #     metrics = {}
    #     try:
    #         # Basic metrics
    #         metrics['accuracy'] = accuracy_score(y_true, y_pred)
    #
    #         # Calculate confusion matrix first
    #         cm = confusion_matrix(y_true, y_pred)
    #         tn, fp, fn, tp = cm.ravel()
    #
    #         # Store raw confusion matrix values
    #         metrics['true_negatives'] = int(tn)
    #         metrics['false_positives'] = int(fp)
    #         metrics['false_negatives'] = int(fn)
    #         metrics['true_positives'] = int(tp)
    #
    #         # Calculate metrics for normal class (class 0)
    #         total = tn + fp + fn + tp
    #         if total > 0:
    #             # Normal class metrics
    #             metrics['accuracy'] = float((tn + tp) / total)
    #             metrics['normal_precision'] = float(tn / (tn + fn)) if (tn + fn) > 0 else 1.0
    #             metrics['normal_recall'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 1.0
    #             metrics['specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 1.0
    #
    #             # Anomaly class metrics (class 1)
    #             metrics['precision'] = float(tp / (tp + fp)) if (tp + fp) > 0 else 1.0
    #             metrics['recall'] = float(tp / (tp + fn)) if (tp + fn) > 0 else 1.0
    #
    #             # F1 scores
    #             if metrics['precision'] + metrics['recall'] > 0:
    #                 metrics['f1_score'] = float(2 * (metrics['precision'] * metrics['recall']) /
    #                                             (metrics['precision'] + metrics['recall']))
    #             else:
    #                 metrics['f1_score'] = 0.0
    #
    #             if metrics['normal_precision'] + metrics['normal_recall'] > 0:
    #                 metrics['normal_f1'] = float(2 * (metrics['normal_precision'] * metrics['normal_recall']) /
    #                                              (metrics['normal_precision'] + metrics['normal_recall']))
    #             else:
    #                 metrics['normal_f1'] = 0.0
    #
    #             # Additional metrics
    #             metrics['total_accuracy'] = float((tp + tn) / total)
    #             metrics['balanced_accuracy'] = float((metrics['recall'] + metrics['specificity']) / 2)
    #             metrics['false_positive_rate'] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    #             metrics['false_negative_rate'] = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    #
    #         # Handle zero metrics gracefully
    #         for key in metrics:
    #             if isinstance(metrics[key], (int, float)):
    #                 if np.isnan(metrics[key]) or np.isinf(metrics[key]):
    #                     metrics[key] = 0.0
    #
    #         return metrics
    #
    #     except Exception as e:
    #         self.logger.error(f"Error calculating basic metrics: {e}")
    #         # Return default metrics
    #         return {
    #             'accuracy': 0.0,
    #             'precision': 0.0,
    #             'recall': 0.0,
    #             'f1_score': 0.0,
    #             'specificity': 0.0,
    #             'normal_precision': 0.0,
    #             'normal_recall': 0.0,
    #             'normal_f1': 0.0,
    #             'total_accuracy': 0.0,
    #             'balanced_accuracy': 0.0,
    #             'false_positive_rate': 0.0,
    #             'false_negative_rate': 0.0,
    #             'true_negatives': 0,
    #             'false_positives': 0,
    #             'false_negatives': 0,
    #             'true_positives': 0
    #         }

    def _calculate_basic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate basic classification metrics"""
        metrics = {}
        try:
            # Basic metrics
            metrics['accuracy'] = accuracy_score(y_true, y_pred)

            # Calculate confusion matrix first
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()

            # Store raw confusion matrix values
            metrics['true_negatives'] = int(tn)
            metrics['false_positives'] = int(fp)
            metrics['false_negatives'] = int(fn)
            metrics['true_positives'] = int(tp)

            # Calculate metrics for normal class (class 0)
            total = tn + fp + fn + tp
            if total > 0:
                # Normal class metrics
                metrics['accuracy'] = float((tn + tp) / total)
                metrics['normal_precision'] = float(tn / (tn + fn)) if (tn + fn) > 0 else 1.0
                metrics['normal_recall'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 1.0
                metrics['specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 1.0

                # Anomaly class metrics (class 1)
                metrics['precision'] = float(tp / (tp + fp)) if (tp + fp) > 0 else 1.0
                metrics['recall'] = float(tp / (tp + fn)) if (tp + fn) > 0 else 1.0

                # F1 scores
                if metrics['precision'] + metrics['recall'] > 0:
                    metrics['f1_score'] = float(2 * (metrics['precision'] * metrics['recall']) /
                                                (metrics['precision'] + metrics['recall']))
                else:
                    metrics['f1_score'] = 0.0

                if metrics['normal_precision'] + metrics['normal_recall'] > 0:
                    metrics['normal_f1'] = float(2 * (metrics['normal_precision'] * metrics['normal_recall']) /
                                                 (metrics['normal_precision'] + metrics['normal_recall']))
                else:
                    metrics['normal_f1'] = 0.0

                # Additional metrics
                metrics['total_accuracy'] = float((tp + tn) / total)
                metrics['balanced_accuracy'] = float((metrics['recall'] + metrics['specificity']) / 2)
                metrics['false_positive_rate'] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
                metrics['false_negative_rate'] = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

            # Handle zero metrics gracefully
            for key in metrics:
                if isinstance(metrics[key], (int, float)):
                    if np.isnan(metrics[key]) or np.isinf(metrics[key]):
                        metrics[key] = 0.0

            return metrics

        except Exception as e:
            self.logger.error(f"Error calculating basic metrics: {e}")
            # Return default metrics
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'specificity': 0.0,
                'normal_precision': 0.0,
                'normal_recall': 0.0,
                'normal_f1': 0.0,
                'total_accuracy': 0.0,
                'balanced_accuracy': 0.0,
                'false_positive_rate': 0.0,
                'false_negative_rate': 0.0,
                'true_negatives': 0,
                'false_positives': 0,
                'false_negatives': 0,
                'true_positives': 0
            }



    def _calculate_threshold_metrics(self, y_true: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
        """Calculate threshold-based metrics with enhanced robustness"""
        metrics = {}
        try:
            if len(scores) == 0:
                return metrics

            # ROC curve metrics with validation
            try:
                fpr, tpr, _ = roc_curve(y_true, scores)
                metrics['roc_auc'] = float(auc(fpr, tpr))
            except Exception as e:
                self.logger.warning(f"Error calculating ROC metrics: {e}")
                metrics['roc_auc'] = 0.0

            # Precision-Recall curve metrics with validation
            try:
                precision, recall, _ = precision_recall_curve(y_true, scores)
                metrics['pr_auc'] = float(auc(recall, precision))
            except Exception as e:
                self.logger.warning(f"Error calculating PR metrics: {e}")
                metrics['pr_auc'] = 0.0

            # Score distribution metrics with validation
            valid_scores = scores[~np.isnan(scores) & ~np.isinf(scores)]
            if len(valid_scores) > 0:
                metrics['score_mean'] = float(np.mean(valid_scores))
                metrics['score_std'] = float(np.std(valid_scores))
                metrics['score_median'] = float(np.median(valid_scores))
                metrics['score_min'] = float(np.min(valid_scores))
                metrics['score_max'] = float(np.max(valid_scores))

                # Add percentile metrics for better threshold analysis
                percentiles = [25, 50, 75, 90, 95, 99]
                for p in percentiles:
                    metrics[f'percentile_{p}'] = float(np.percentile(valid_scores, p))

                # Calculate IQR and outlier bounds
                q1, q3 = np.percentile(valid_scores, [25, 75])
                iqr = q3 - q1
                metrics['iqr'] = float(iqr)
                metrics['outlier_bound_lower'] = float(q1 - 1.5 * iqr)
                metrics['outlier_bound_upper'] = float(q3 + 1.5 * iqr)

                # Calculate percentage of outliers
                outliers = (valid_scores < metrics['outlier_bound_lower']) | (
                            valid_scores > metrics['outlier_bound_upper'])
                metrics['outlier_percentage'] = float(np.mean(outliers) * 100)

            return metrics

        except Exception as e:
            self.logger.error(f"Error calculating threshold metrics: {e}")
            return {}

    def plot_results(self, mode: str = 'test') -> None:
        """Generate comprehensive result plots"""
        try:
            results = self.results[mode]
            metrics = self.metrics[mode]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Overall performance plots
            self._plot_overall_performance(results, metrics['overall'], mode, timestamp)

            # Process-specific plots
            self._plot_process_performance(results, metrics['processes'], mode, timestamp)

            # Score distribution plots
            if 'mse_scores' in results:
                self._plot_score_distributions(results, mode, timestamp)

            # Time series plots
            self._plot_time_series(results, mode, timestamp)

        except Exception as e:
            self.logger.error(f"Error plotting results: {e}")

    def _plot_overall_performance(self, results: Dict, metrics: Dict, mode: str, timestamp: str) -> None:
        """Plot overall performance metrics"""
        try:
            fig = plt.figure(figsize=(20, 15))

            # Confusion Matrix
            ax1 = plt.subplot(221)
            cm = confusion_matrix(results['true_labels'], results['predictions'])
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                ax=ax1,
                cmap='Blues'
            )
            ax1.set_title('Confusion Matrix')
            ax1.set_xlabel('Predicted')
            ax1.set_ylabel('True')

            # ROC Curve
            if 'mse_scores' in results:
                ax2 = plt.subplot(222)
                fpr, tpr, _ = roc_curve(results['true_labels'], results['mse_scores'])
                ax2.plot(
                    fpr,
                    tpr,
                    label=f'ROC curve (AUC = {metrics["roc_auc"]:.2f})'
                )
                ax2.plot([0, 1], [0, 1], 'k--')
                ax2.set_title('ROC Curve')
                ax2.set_xlabel('False Positive Rate')
                ax2.set_ylabel('True Positive Rate')
                ax2.legend()
                ax2.grid(True)

                # Precision-Recall Curve
                ax3 = plt.subplot(223)
                precision, recall, _ = precision_recall_curve(
                    results['true_labels'],
                    results['mse_scores']
                )
                ax3.plot(
                    recall,
                    precision,
                    label=f'PR curve (AUC = {metrics["pr_auc"]:.2f})'
                )
                ax3.set_title('Precision-Recall Curve')
                ax3.set_xlabel('Recall')
                ax3.set_ylabel('Precision')
                ax3.legend()
                ax3.grid(True)

            # Metrics Summary
            ax4 = plt.subplot(224)
            ax4.axis('off')
            metrics_text = (
                f"Accuracy: {metrics['accuracy']:.4f}\n"
                f"Precision: {metrics['precision']:.4f}\n"
                f"Recall: {metrics['recall']:.4f}\n"
                f"F1 Score: {metrics['f1_score']:.4f}\n"
                f"Specificity: {metrics['specificity']:.4f}\n"
                f"False Positive Rate: {metrics['false_positive_rate']:.4f}\n"
                f"False Negative Rate: {metrics['false_negative_rate']:.4f}"
            )
            ax4.text(0.1, 0.5, metrics_text, fontsize=12)

            plt.tight_layout()
            plt.savefig(
                os.path.join(self.plots_dir, f'overall_performance_{mode}_{timestamp}.png'),
                dpi=300,
                bbox_inches='tight'
            )
            plt.close()

        except Exception as e:
            self.logger.error(f"Error plotting overall performance: {e}")

    def _plot_process_performance(self, results: Dict, process_metrics: Dict,
                                  mode: str, timestamp: str) -> None:
        """Plot process-specific performance"""
        try:
            if not process_metrics:
                return

            # Create metrics comparison plot
            metrics_df = pd.DataFrame(process_metrics).T

            plt.figure(figsize=(15, 10))
            metrics_df[['precision', 'recall', 'f1_score', 'accuracy']].plot(kind='bar')
            plt.title('Performance Metrics by Process')
            plt.xlabel('Process')
            plt.ylabel('Score')
            plt.legend()
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()

            plt.savefig(
                os.path.join(self.plots_dir, f'process_performance_{mode}_{timestamp}.png'),
                dpi=300,
                bbox_inches='tight'
            )
            plt.close()

            # Create score distribution plots per process
            if 'processes' in results:
                n_processes = len(results['processes'])
                fig, axes = plt.subplots(
                    n_processes,
                    2,
                    figsize=(15, 5 * n_processes)
                )

                for idx, (process, process_results) in enumerate(results['processes'].items()):
                    if 'mse_scores' in process_results:
                        # Score distribution
                        sns.histplot(
                            data=process_results['mse_scores'],
                            ax=axes[idx, 0],
                            bins=50
                        )
                        axes[idx, 0].set_title(f'Score Distribution - {process}')

                        # Prediction distribution
                        sns.countplot(
                            x=process_results['predictions'],
                            ax=axes[idx, 1]
                        )
                        axes[idx, 1].set_title(f'Prediction Distribution - {process}')

                plt.tight_layout()
                plt.savefig(
                    os.path.join(self.plots_dir, f'process_distributions_{mode}_{timestamp}.png'),
                    dpi=300,
                    bbox_inches='tight'
                )
                plt.close()

        except Exception as e:
            self.logger.error(f"Error plotting process performance: {e}")

    def _plot_score_distributions(self, results: Dict, mode: str, timestamp: str) -> None:
        """Plot score distributions"""
        try:
            plt.figure(figsize=(15, 10))

            # Overall score distribution
            plt.subplot(211)
            sns.histplot(
                data=pd.DataFrame({
                    'MSE': results['mse_scores'],
                    'Label': results['true_labels']
                }),
                x='MSE',
                hue='Label',
                bins=50
            )
            plt.title('Score Distribution by Label')

            # Box plot
            plt.subplot(212)
            sns.boxplot(
                data=pd.DataFrame({
                    'MSE': results['mse_scores'],
                    'Label': results['true_labels']
                }),
                x='Label',
                y='MSE'
            )
            plt.title('Score Distribution Box Plot')

            plt.tight_layout()
            plt.savefig(
                os.path.join(self.plots_dir, f'score_distributions_{mode}_{timestamp}.png'),
                dpi=300,
                bbox_inches='tight'
            )
            plt.close()

        except Exception as e:
            self.logger.error(f"Error plotting score distributions: {e}")

    def _plot_time_series(self, results: Dict, mode: str, timestamp: str) -> None:
        """Plot time series analysis"""
        try:
            if 'mse_scores' not in results:
                return

            plt.figure(figsize=(15, 10))

            # Score evolution
            plt.subplot(211)
            plt.plot(results['mse_scores'])
            plt.axhline(
                y=np.mean(results['mse_scores']),
                color='r',
                linestyle='--',
                label='Mean'
            )
            plt.title('Score Evolution Over Time')
            plt.xlabel('Sample')
            plt.ylabel('MSE Score')
            plt.legend()
            plt.grid(True)

            # Moving average of anomaly rate
            plt.subplot(212)
            window_size = 100
            moving_avg = pd.Series(results['predictions']).rolling(window_size).mean()
            plt.plot(moving_avg, label=f'Moving Average (window={window_size})')
            plt.title('Anomaly Rate Evolution')
            plt.xlabel('Sample')
            plt.ylabel('Anomaly Rate')
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plt.savefig(
                os.path.join(self.plots_dir, f'time_series_{mode}_{timestamp}.png'),
                dpi=300,
                bbox_inches='tight'
            )
            plt.close()

        except Exception as e:
            self.logger.error(f"Error plotting time series: {e}")



    def generate_report(self, mode: str = 'test') -> None:
        """Generate comprehensive analysis report with enhanced metrics"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = os.path.join(self.reports_dir, f'analysis_report_{mode}_{timestamp}.html')

            # Gather comprehensive metrics
            metrics = self.calculate_metrics(mode)
            system_metrics = self._get_system_metrics()
            runtime_stats = self._get_runtime_statistics()

            # Generate enhanced HTML content
            html_content = self._generate_html_report(
                mode=mode,
                timestamp=timestamp,
                metrics=metrics,
                system_metrics=system_metrics,
                runtime_stats=runtime_stats
            )

            # Save report
            with open(report_file, 'w') as f:
                f.write(html_content)

            # Save additional artifacts
            self._save_metric_plots(metrics, mode, timestamp)
            self._save_metric_summary(metrics, mode, timestamp)

            self.logger.info(f"Analysis report generated: {report_file}")

        except Exception as e:
            self.logger.error(f"Error generating report: {e}")
            self.logger.error(traceback.format_exc())

    def _get_system_metrics(self) -> Dict[str, float]:
        """Get current system metrics"""
        try:
            return {
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent,
                'process_memory': psutil.Process().memory_info().rss / 1024 / 1024  # MB
            }
        except Exception as e:
            self.logger.error(f"Error getting system metrics: {e}")
            return {'cpu_percent': 0, 'memory_percent': 0, 'disk_usage': 0, 'process_memory': 0}

    def _get_default_metrics(self) -> Dict[str, float]:
        """Return default metrics when calculation fails"""
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'specificity': 0.0,
            'true_positives': 0,
            'true_negatives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'false_positive_rate': 0.0,
            'false_negative_rate': 0.0
        }

    def _generate_html_report(self, mode: str, timestamp: str) -> str:
        """Generate HTML report content"""
        try:
            metrics = self.metrics[mode]

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
                    .plot {
                        max-width: 100%;
                        height: auto;
                        margin: 20px 0;
                    }
                    .process-section {
                        margin: 20px 0;
                        padding: 15px;
                        background-color: #f8f9fa;
                        border-radius: 4px;
                    }
                    .timestamp {
                        color: #7f8c8d;
                        font-style: italic;
                    }
                </style>
            </head>
            <body>
                <div class="container">
                    <h2>Performance Analysis Report ({mode})</h2>
                    <p class="timestamp">Generated: {timestamp}</p>

                    <h3>Overall Performance</h3>
                    <table>
                        <tr><th>Metric</th><th>Value</th></tr>
                        {overall_metrics}
                    </table>

                    <div class="plot">
                        <h3>Performance Visualizations</h3>
                        <img src="../plots/overall_performance_{mode}_{timestamp}.png" 
                             alt="Overall Performance">
                    </div>

                    <h3>Process-Specific Performance</h3>
                    {process_sections}

                    <div class="plot">
                        <h3>Score Distributions</h3>
                        <img src="../plots/score_distributions_{mode}_{timestamp}.png" 
                             alt="Score Distributions">
                    </div>

                    <div class="plot">
                        <h3>Time Series Analysis</h3>
                        <img src="../plots/time_series_{mode}_{timestamp}.png" 
                             alt="Time Series Analysis">
                    </div>

                    <h3>Statistical Analysis</h3>
                    {statistical_analysis}
                </div>
            </body>
            </html>
            """

            # Generate overall metrics table
            overall_metrics = ""
            for metric, value in metrics['overall'].items():
                if isinstance(value, float):
                    overall_metrics += f"""
                        <tr>
                            <td>{metric.replace('_', ' ').title()}</td>
                            <td class="metric">{value:.4f}</td>
                        </tr>
                    """
                else:
                    overall_metrics += f"""
                        <tr>
                            <td>{metric.replace('_', ' ').title()}</td>
                            <td class="metric">{value}</td>
                        </tr>
                    """

            # Generate process-specific sections
            process_sections = ""
            for process, process_metrics in metrics.get('processes', {}).items():
                process_sections += f"""
                    <div class="process-section">
                        <h4>{process}</h4>
                        <table>
                            <tr><th>Metric</th><th>Value</th></tr>
                """

                for metric, value in process_metrics.items():
                    if isinstance(value, float):
                        process_sections += f"""
                            <tr>
                                <td>{metric.replace('_', ' ').title()}</td>
                                <td class="metric">{value:.4f}</td>
                            </tr>
                        """
                    else:
                        process_sections += f"""
                            <tr>
                                <td>{metric.replace('_', ' ').title()}</td>
                                <td class="metric">{value}</td>
                            </tr>
                        """

                process_sections += "</table></div>"

            # Generate statistical analysis
            statistical_analysis = self._generate_statistical_analysis(mode)

            # Format the template
            return html_template.format(
                mode=mode,
                timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                overall_metrics=overall_metrics,
                process_sections=process_sections,
                statistical_analysis=statistical_analysis
            )

        except Exception as e:
            self.logger.error(f"Error generating HTML report: {e}")
            return ""

    def _generate_statistical_analysis(self, mode: str) -> str:
        """Generate statistical analysis section"""
        try:
            results = self.results[mode]

            if 'mse_scores' not in results:
                return "<p>No score data available for statistical analysis.</p>"

            # Calculate statistics
            normal_scores = results['mse_scores'][results['true_labels'] == 0]
            anomaly_scores = results['mse_scores'][results['true_labels'] == 1]

            stats = {
                'Normal Samples': {
                    'Count': len(normal_scores),
                    'Mean': np.mean(normal_scores),
                    'Std': np.std(normal_scores),
                    'Median': np.median(normal_scores),
                    'Min': np.min(normal_scores),
                    'Max': np.max(normal_scores)
                },
                'Anomaly Samples': {
                    'Count': len(anomaly_scores),
                    'Mean': np.mean(anomaly_scores),
                    'Std': np.std(anomaly_scores),
                    'Median': np.median(anomaly_scores),
                    'Min': np.min(anomaly_scores),
                    'Max': np.max(anomaly_scores)
                }
            }

            # Generate HTML table
            html = """
                <table>
                    <tr>
                        <th>Statistic</th>
                        <th>Normal Samples</th>
                        <th>Anomaly Samples</th>
                    </tr>
            """

            for stat in ['Count', 'Mean', 'Std', 'Median', 'Min', 'Max']:
                html += f"""
                    <tr>
                        <td>{stat}</td>
                        <td class="metric">
                            {stats['Normal Samples'][stat]:.4f if stat != 'Count' 
                             else stats['Normal Samples'][stat]}
                        </td>
                        <td class="metric">
                            {stats['Anomaly Samples'][stat]:.4f if stat != 'Count' 
                             else stats['Anomaly Samples'][stat]}
                        </td>
                    </tr>
                """

            html += "</table>"

            # Add additional statistics
            detection_rate = len(anomaly_scores) / len(results['mse_scores'])
            false_positive_rate = np.sum(results['predictions'][results['true_labels'] == 0]) / len(normal_scores)

            html += f"""
                <h4>Additional Statistics</h4>
                <table>
                    <tr>
                        <td>Detection Rate</td>
                        <td class="metric">{detection_rate:.2%}</td>
                    </tr>
                    <tr>
                        <td>False Positive Rate</td>
                        <td class="metric">{false_positive_rate:.2%}</td>
                    </tr>
                </table>
            """

            return html

        except Exception as e:
            self.logger.error(f"Error generating statistical analysis: {e}")
            return "<p>Error generating statistical analysis.</p>"


def main():
    """Main execution function"""
    try:
        # Initialize analyzer
        analyzer = ResultAnalyzer()

        try:
            # Load and analyze test results
            if analyzer.load_results(mode='test'):
                analyzer.calculate_metrics(mode='test')
                analyzer.plot_results(mode='test')
                analyzer.generate_report(mode='test')

            # Load and analyze real-time results
            if analyzer.load_results(mode='realtime'):
                analyzer.calculate_metrics(mode='realtime')
                analyzer.plot_results(mode='realtime')
                analyzer.generate_report(mode='realtime')

        except Exception as e:
            analyzer.logger.error(f"Error in analysis: {e}")
            import traceback
            analyzer.logger.error(traceback.format_exc())

    except Exception as e:
        logging.error(f"Critical error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
