import os
import numpy as np
import networkx as nx
import traceback
import logging
import time
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from .syscall_utils import (
    known_syscalls,
    known_arguments,
    is_unseen_syscall,
    is_unseen_argument,
    update_known_entities,
    distinct_syscalls_with_unseen_args
)
from .model_utils import detect_anomalies_with_fixed_threshold
from datetime import datetime
from .config import config, GRAPHS_DIR
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
plt.ioff()  # Turn off interactive mode
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import logging
from datetime import datetime
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any, Optional
import traceback

# Module level logger
logger = logging.getLogger(__name__)




# def preprocess_data(collected_syscalls, interval_counter=None, mode='test', autoencoder=None,
#                     threshold_trained=None, threshold_unseen=None, scaler=None):
#     """Modified preprocessing with comprehensive statistics, metrics, and SSG plotting"""
#     try:
#         batch_start_time = time.time()
#
#         # Single input validation
#         if not collected_syscalls or not isinstance(collected_syscalls, list):
#             logger.error("Invalid input data structure")
#             return None
#
#         # Get all syscalls
#         all_syscalls = collected_syscalls[0]
#         total_syscalls = len(all_syscalls)
#
#         # Check minimum requirements
#         if total_syscalls < config.MIN_SAMPLES_REQUIRED // 2:
#             logger.info(f"Insufficient syscalls: {total_syscalls} < {config.MIN_SAMPLES_REQUIRED // 2}")
#             return None
#
#         # Initialize interval counter if not provided
#         if interval_counter is None:
#             interval_counter = 0
#
#         #logger.info(f"Starting preprocessing of {total_syscalls} syscalls")
#
#         # Calculate window parameters once
#         window_size = config.MIN_SAMPLES_REQUIRED
#         stride = window_size // 4  # 75% overlap
#
#         # Calculate number of complete windows
#         num_complete_windows = (total_syscalls - window_size) // stride + 1
#
#         # Initialize features list
#         ssg_features = []
#         processed_syscalls = 0
#
#         # Process complete windows
#         for i in range(num_complete_windows):
#             start_idx = i * stride
#             end_idx = start_idx + window_size
#             if end_idx <= total_syscalls:
#                 window = all_syscalls[start_idx:end_idx]
#                 if len(window) == window_size:
#                     try:
#                         # Build SSG for window
#                         G = build_ssg(window)
#
#                         # Plot SSG based on mode and interval
#                         current_interval = interval_counter + i
#                         if mode == 'train' or (mode == 'test' and current_interval % config.SSG_CONFIG['plot_interval'] == 0):
#                             plot_title = f"SSG_Window_{i}_Interval_{current_interval}_{mode}"
#                             logger.info(f"Plotting SSG for {plot_title}")
#                             plot_ssg(G, title=plot_title)
#
#                         # Extract features
#                         usi, uai, graph_size, context_metrics = calculate_ssg_features(G, window)
#
#                         # Calculate graph metrics
#                         node_count = G.number_of_nodes()
#                         edge_count = G.number_of_edges()
#                         avg_degree = float(edge_count) / node_count if node_count > 0 else 0
#
#                         # Get context metrics
#                         context_influence = context_metrics['context_influence']
#                         frequency_increase = context_metrics['frequency_increase']
#
#                         # Calculate syscall diversity metrics
#                         syscall_counts = defaultdict(int)
#                         arg_counts = defaultdict(int)
#                         for syscall in window:
#                             syscall_counts[syscall['name']] += 1
#                             for arg in syscall.get('args', []):
#                                 arg_counts[str(arg)] += 1
#
#                         unique_syscalls = len(syscall_counts)
#                         unique_args = len(arg_counts)
#
#                         # Calculate syscall entropy
#                         total_syscalls_window = sum(syscall_counts.values())
#                         syscall_entropy = -sum((count / total_syscalls_window) * np.log2(count / total_syscalls_window)
#                                              for count in syscall_counts.values())
#
#                         # Compile feature vector
#                         feature_vector = [
#                             float(usi),                    # 1. Unseen Syscall Influence
#                             float(uai),                    # 2. Unseen Argument Influence
#                             float(graph_size),             # 3. Graph size
#                             float(node_count),             # 4. Node count
#                             float(edge_count),             # 5. Edge count
#                             float(avg_degree),             # 6. Average degree
#                             float(context_influence),      # 7. Context influence
#                             float(frequency_increase),     # 8. Frequency increase
#                             float(unique_syscalls),        # 9. Unique syscalls
#                             float(unique_args),            # 10. Unique arguments
#                             float(syscall_entropy)         # 11. Syscall entropy
#                         ]
#
#                         ssg_features.append(feature_vector)
#                         processed_syscalls = end_idx
#
#                     except Exception as e:
#                         logger.error(f"Error processing window {i}: {e}")
#                         continue
#
#         # Handle remaining syscalls
#         remaining_syscalls = total_syscalls - processed_syscalls
#         if remaining_syscalls >= window_size // 2:
#             window = all_syscalls[-window_size:]
#             try:
#                 G = build_ssg(window)
#                 if mode == 'train' or (mode == 'test' and num_complete_windows % config.SSG_CONFIG['plot_interval'] == 0):
#                     plot_title = f"SSG_Window_Final_Interval_{interval_counter + num_complete_windows}_{mode}"
#                     plot_ssg(G, title=plot_title)
#
#                 # Extract features
#                 usi, uai, graph_size, context_metrics = calculate_ssg_features(G, window)
#
#                 # Calculate graph metrics
#                 node_count = G.number_of_nodes()
#                 edge_count = G.number_of_edges()
#                 avg_degree = float(edge_count) / node_count if node_count > 0 else 0
#
#                 # Get context metrics
#                 context_influence = context_metrics['context_influence']
#                 frequency_increase = context_metrics['frequency_increase']
#
#                 # Calculate syscall diversity metrics
#                 syscall_counts = defaultdict(int)
#                 arg_counts = defaultdict(int)
#                 for syscall in window:
#                     syscall_counts[syscall['name']] += 1
#                     for arg in syscall.get('args', []):
#                         arg_counts[str(arg)] += 1
#
#                 unique_syscalls = len(syscall_counts)
#                 unique_args = len(arg_counts)
#
#                 # Calculate syscall entropy
#                 total_syscalls_window = sum(syscall_counts.values())
#                 syscall_entropy = -sum((count / total_syscalls_window) * np.log2(count / total_syscalls_window)
#                                        for count in syscall_counts.values())
#
#                 # Compile feature vector
#                 feature_vector = [
#                     float(usi),  # 1. Unseen Syscall Influence
#                     float(uai),  # 2. Unseen Argument Influence
#                     float(graph_size),  # 3. Graph size
#                     float(node_count),  # 4. Node count
#                     float(edge_count),  # 5. Edge count
#                     float(avg_degree),  # 6. Average degree
#                     float(context_influence),  # 7. Context influence
#                     float(frequency_increase),  # 8. Frequency increase
#                     float(unique_syscalls),  # 9. Unique syscalls
#                     float(unique_args),  # 10. Unique arguments
#                     float(syscall_entropy)  # 11. Syscall entropy
#                 ]
#
#                 ssg_features.append(feature_vector)
#                 processed_syscalls = end_idx
#
#             except Exception as e:
#                 logger.error(f"Error processing final window: {e}")
#
#         if not ssg_features:
#             logger.error("No features extracted")
#             return None
#
#         # Convert to numpy array
#         ssg_features = np.array(ssg_features, dtype=np.float32)
#
#         # Calculate processing metrics
#         processing_time = time.time() - batch_start_time
#         processed_percentage = (processed_syscalls / total_syscalls) * 100 if total_syscalls > 0 else 0
#
#         # Log summary
#         # logger.info("\nProcessing Summary:")
#         # logger.info(f"- Total syscalls: {total_syscalls}")
#         # logger.info(f"- Processed syscalls: {processed_syscalls}")
#         # logger.info(f"- Features extracted: {len(ssg_features)}")
#         # logger.info(f"- Processing time: {processing_time:.2f}s")
#         # logger.info(f"- Processing rate: {total_syscalls/processing_time:.2f} syscalls/second")
#         # logger.info(f"- Coverage: {processed_percentage:.1f}%")
#
#         return ssg_features
#
#     except Exception as e:
#         logger.error(f"Error in preprocessing: {e}")
#         logger.error(traceback.format_exc())
#         return None

def preprocess_data(collected_syscalls, interval_counter=None, mode='test', autoencoder=None,
                    threshold_trained=None, threshold_unseen=None, scaler=None):
    """Modified preprocessing with comprehensive statistics, metrics, and SSG plotting"""
    try:
        batch_start_time = time.time()

        # Single input validation
        if not collected_syscalls or not isinstance(collected_syscalls, list):
            logger.error("Invalid input data structure")
            return None

        # Get all syscalls
        all_syscalls = collected_syscalls[0]
        total_syscalls = len(all_syscalls)

        # Check minimum requirements
        if total_syscalls < config.MIN_SAMPLES_REQUIRED // 2:
            logger.info(f"Insufficient syscalls: {total_syscalls} < {config.MIN_SAMPLES_REQUIRED // 2}")
            return None

        # Initialize interval counter if not provided
        if interval_counter is None:
            interval_counter = 0

        #logger.info(f"Starting preprocessing of {total_syscalls} syscalls")

        # Add new windowing summary logging
        # logger.info(f"\nWindowing Process Summary:")
        # logger.info("=" * 50)
        # logger.info(f"Total syscalls collected: {total_syscalls}")

        # Calculate window parameters once
        window_size = config.MIN_SAMPLES_REQUIRED
        stride = window_size // 4  # 75% overlap

        # Calculate number of complete windows
        num_complete_windows = (total_syscalls - window_size) // stride + 1

        # Add window configuration logging
        # logger.info(f"\nWindow configuration:")
        # logger.info(f"- Window size: {window_size}")
        # logger.info(f"- Stride size: {stride}")
        # logger.info(f"- Overlap percentage: 75%")
        # logger.info(f"- Expected complete windows: {num_complete_windows}")

        # Initialize features list
        ssg_features = []
        processed_syscalls = 0
        window_count = 0  # Add window counter

        # Process complete windows
        for i in range(num_complete_windows):
            start_idx = i * stride
            end_idx = start_idx + window_size

            # # Add window progress logging
            # if i % 5 == 0:  # Log every 5th window
            #     logger.info(f"\nProcessing window {i+1}/{num_complete_windows}:")
            #     logger.info(f"- Start index: {start_idx}")
            #     logger.info(f"- End index: {end_idx}")
            #     logger.info(f"- Window size: {window_size}")

            if end_idx <= total_syscalls:
                window = all_syscalls[start_idx:end_idx]
                if len(window) == window_size:
                    try:
                        # Build SSG for window
                        G = build_ssg(window)

                        # Plot SSG based on mode and interval
                        current_interval = interval_counter + i
                        if mode == 'train' or (mode == 'test' and current_interval % config.SSG_CONFIG['plot_interval'] == 0):
                            plot_title = f"SSG_Window_{i}_Interval_{current_interval}_{mode}"
                            logger.info(f"Plotting SSG for {plot_title}")
                            #plot_ssg(G, title=plot_title)

                        # Extract features
                        usi, uai, graph_size, context_metrics = calculate_ssg_features(G, window)

                        # Calculate graph metrics
                        node_count = G.number_of_nodes()
                        edge_count = G.number_of_edges()
                        avg_degree = float(edge_count) / node_count if node_count > 0 else 0

                        # Get context metrics
                        context_influence = context_metrics['context_influence']
                        frequency_increase = context_metrics['frequency_increase']

                        # Calculate syscall diversity metrics
                        syscall_counts = defaultdict(int)
                        arg_counts = defaultdict(int)
                        for syscall in window:
                            syscall_counts[syscall['name']] += 1
                            for arg in syscall.get('args', []):
                                arg_counts[str(arg)] += 1

                        unique_syscalls = len(syscall_counts)
                        unique_args = len(arg_counts)

                        # Calculate syscall entropy
                        total_syscalls_window = sum(syscall_counts.values())
                        syscall_entropy = -sum((count / total_syscalls_window) * np.log2(count / total_syscalls_window)
                                             for count in syscall_counts.values())

                        # Compile feature vector
                        feature_vector = [
                            float(usi),                    # 1. Unseen Syscall Influence
                            float(uai),                    # 2. Unseen Argument Influence
                            float(graph_size),             # 3. Graph size
                            float(node_count),             # 4. Node count
                            float(edge_count),             # 5. Edge count
                            float(avg_degree),             # 6. Average degree
                            float(context_influence),      # 7. Context influence
                            float(frequency_increase),     # 8. Frequency increase
                            float(unique_syscalls),        # 9. Unique syscalls
                            float(unique_args),            # 10. Unique arguments
                            float(syscall_entropy)         # 11. Syscall entropy
                        ]

                        ssg_features.append(feature_vector)
                        processed_syscalls = end_idx
                        window_count += 1  # Increment window counter

                    except Exception as e:
                        logger.error(f"Error processing window {i}: {e}")
                        continue

        # Handle remaining syscalls
        remaining_syscalls = total_syscalls - processed_syscalls
        if remaining_syscalls >= window_size // 2:
            # Add remaining syscalls logging
            logger.info(f"\nProcessing final partial window:")
            logger.info(f"- Remaining syscalls: {remaining_syscalls}")
            logger.info(f"- Using last {window_size} syscalls")

            window = all_syscalls[-window_size:]
            try:
                G = build_ssg(window)
                if mode == 'train' or (mode == 'test' and num_complete_windows % config.SSG_CONFIG['plot_interval'] == 0):
                    plot_title = f"SSG_Window_Final_Interval_{interval_counter + num_complete_windows}_{mode}"
                    #plot_ssg(G, title=plot_title)

                # Extract features
                usi, uai, graph_size, context_metrics = calculate_ssg_features(G, window)

                # Calculate graph metrics
                node_count = G.number_of_nodes()
                edge_count = G.number_of_edges()
                avg_degree = float(edge_count) / node_count if node_count > 0 else 0

                # Get context metrics
                context_influence = context_metrics['context_influence']
                frequency_increase = context_metrics['frequency_increase']

                # Calculate syscall diversity metrics
                syscall_counts = defaultdict(int)
                arg_counts = defaultdict(int)
                for syscall in window:
                    syscall_counts[syscall['name']] += 1
                    for arg in syscall.get('args', []):
                        arg_counts[str(arg)] += 1

                unique_syscalls = len(syscall_counts)
                unique_args = len(arg_counts)

                # Calculate syscall entropy
                total_syscalls_window = sum(syscall_counts.values())
                syscall_entropy = -sum((count / total_syscalls_window) * np.log2(count / total_syscalls_window)
                                       for count in syscall_counts.values())

                # Compile feature vector
                feature_vector = [
                    float(usi),  # 1. Unseen Syscall Influence
                    float(uai),  # 2. Unseen Argument Influence
                    float(graph_size),  # 3. Graph size
                    float(node_count),  # 4. Node count
                    float(edge_count),  # 5. Edge count
                    float(avg_degree),  # 6. Average degree
                    float(context_influence),  # 7. Context influence
                    float(frequency_increase),  # 8. Frequency increase
                    float(unique_syscalls),  # 9. Unique syscalls
                    float(unique_args),  # 10. Unique arguments
                    float(syscall_entropy)  # 11. Syscall entropy
                ]

                ssg_features.append(feature_vector)
                processed_syscalls = end_idx
                window_count += 1  # Increment window counter for final window

            except Exception as e:
                logger.error(f"Error processing final window: {e}")

        if not ssg_features:
            logger.error("No features extracted")
            return None

        # Convert to numpy array
        ssg_features = np.array(ssg_features, dtype=np.float32)

        # Calculate processing metrics
        processing_time = time.time() - batch_start_time
        processed_percentage = (processed_syscalls / total_syscalls) * 100 if total_syscalls > 0 else 0

        #Enhanced logging summary
        # logger.info("\nWindowing Process Complete:")
        # logger.info("=" * 50)
        # logger.info(f"Total windows created: {window_count}")
        # logger.info(f"Syscalls per window: {window_size}")
        # logger.info(f"Total syscalls processed: {processed_syscalls}")
        # logger.info(f"Coverage: {processed_percentage:.2f}%")
        # #logger.info(f"Processing time: {processing_time:.2f}s")
        # logger.info(f"Processing rate: {total_syscalls/processing_time:.2f} syscalls/second")

        return ssg_features

    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
        logger.error(traceback.format_exc())
        return None


def extract_window_features(G, window):
    """Extract features for a single window"""
    try:
        # Extract basic features
        usi, uai, graph_size, context_metrics = calculate_ssg_features(G, window)

        # Calculate graph metrics
        node_count = G.number_of_nodes()
        edge_count = G.number_of_edges()
        avg_degree = float(edge_count) / node_count if node_count > 0 else 0

        # Get context metrics
        context_influence = context_metrics['context_influence']
        frequency_increase = context_metrics['frequency_increase']

        # Calculate syscall diversity
        unique_syscalls = len(set(syscall['name'] for syscall in window))
        unique_args = len(set(
            str(arg) for syscall in window
            for arg in syscall.get('args', [])
        ))

        # Add temporal features
        syscall_rates = calculate_syscall_rates(window)

        return [
            float(usi),
            float(uai),
            float(graph_size),
            float(node_count),
            float(edge_count),
            float(avg_degree),
            float(context_influence),
            float(frequency_increase),
            float(unique_syscalls),
            float(unique_args),
            float(syscall_rates['avg_rate']),
            float(syscall_rates['peak_rate'])
        ]

    except Exception as e:
        logger.error(f"Error extracting window features: {e}")
        return None


def calculate_syscall_rates(window):
    """Calculate syscall rate metrics"""
    timestamps = [syscall.get('timestamp', 0) for syscall in window]
    if len(timestamps) < 2:
        return {'avg_rate': 0, 'peak_rate': 0}

    time_diffs = np.diff(timestamps)
    rates = 1 / time_diffs[time_diffs > 0]  # syscalls per second

    return {
        'avg_rate': np.mean(rates) if len(rates) > 0 else 0,
        'peak_rate': np.max(rates) if len(rates) > 0 else 0
    }

def build_ssg(captured_syscalls):
    """Build the System Call Sequence Graph (SSG) with enhanced edge weight calculation"""
    try:
        G = nx.DiGraph()
        edge_weights = defaultdict(int)

        # Add special nodes for unseen syscalls and arguments
        G.add_node("USN")
        G.add_node("UAN")

        prev_syscall = None
        for syscall in captured_syscalls:
            try:
                call_name = syscall['name']

                # Handle unseen syscalls
                if is_unseen_syscall(syscall['name']):
                    call_name = "USN"

                G.add_node(call_name)

                if prev_syscall:
                    edge_weights[(prev_syscall, call_name)] += 1

                # Handle unseen arguments
                unseen_args_found = any(is_unseen_argument(arg) for arg in syscall.get('args', []))
                if unseen_args_found:
                    G.add_edge("UAN", call_name)
                    edge_weights[("UAN", call_name)] += 1

                prev_syscall = call_name

                # Update known entities
                update_known_entities(syscall['name'], syscall.get('args', []))

            except Exception as e:
                logger.error(f"Error processing syscall in build_ssg: {e}")
                continue

        for (u, v), weight in edge_weights.items():
            G.add_edge(u, v, weight=weight)

        return G

    except Exception as e:
        logger.error(f"Error in build_ssg: {e}")
        return nx.DiGraph()


def calculate_ssg_features(G, all_captured_syscalls):
    """
    Calculate USI, UAI, graph size, and context metrics based on the syscall sequence graph (SSG).
    """
    all_syscall_names = [syscall['name'] for syscall in all_captured_syscalls]
    unseen_syscalls = set(all_syscall_names) - known_syscalls

    all_arguments = [arg for syscall in all_captured_syscalls for arg in syscall.get('args', [])]
    unseen_args = set(all_arguments) - known_arguments

    in_degree_centrality = nx.in_degree_centrality(G)
    out_degree_centrality = nx.out_degree_centrality(G)

    # Calculate Unseen Syscall Influence (USI)
    usi = (in_degree_centrality.get("USN", 0) + out_degree_centrality.get("USN", 0)) * len(unseen_syscalls)

    # Calculate Unseen Argument Influence (UAI)
    uai = (in_degree_centrality.get("UAN", 0) + out_degree_centrality.get("UAN", 0)) * len(unseen_args) * len(distinct_syscalls_with_unseen_args)

    # Calculate graph size (frequency of syscalls)
    graph_size = sum([G[u][v].get('weight', 1) for u, v in G.edges])

    # Contextual influence and frequency increase calculations
    context_influence = calculate_context_influence(G, unseen_syscalls)
    frequency_increase = calculate_frequency_increase(all_captured_syscalls, beta=1.5)

    context_metrics = {
        'context_influence': context_influence,
        'frequency_increase': frequency_increase
    }

    return usi, uai, graph_size, context_metrics

def calculate_context_influence(G, unseen_syscalls):
    """Calculate influence of unseen syscalls in the local context"""
    try:
        influence_score = 0
        for unseen_syscall in unseen_syscalls:
            if unseen_syscall in G:
                neighbors = list(G.neighbors(unseen_syscall))
                influence_score += len(neighbors) * (G.in_degree(unseen_syscall) + G.out_degree(unseen_syscall))
        return influence_score
    except Exception as e:
        logger.error(f"Error calculating context influence: {e}")
        return 0

def calculate_frequency_increase(all_captured_syscalls, beta=1.5):
    """Calculate the frequency increase metric"""
    try:
        syscall_counts = defaultdict(int)
        for syscall in all_captured_syscalls:
            syscall_counts[syscall['name']] += 1

        if not syscall_counts:
            return 0

        avg_frequency = np.mean(list(syscall_counts.values()))
        frequency_increase_count = sum(1 for count in syscall_counts.values() if count > beta * avg_frequency)

        return frequency_increase_count
    except Exception as e:
        logger.error(f"Error calculating frequency increase: {e}")
        return 0


def verify_directories():
    """Verify all required directories exist with proper permissions"""
    try:
        # Check graphs directory
        if not os.path.exists(config.GRAPHS_DIR):
            os.makedirs(config.GRAPHS_DIR, mode=0o755, exist_ok=True)
            os.system(f"sudo chown -R sivaprasad:sivaprasad {config.GRAPHS_DIR}")

        # Test write permissions
        test_file = os.path.join(config.GRAPHS_DIR, 'test.txt')
        try:
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
        except Exception as e:
            logging.error(f"Write permission test failed: {e}")
            return False

        return True

    except Exception as e:
        logging.error(f"Error verifying directories: {e}")
        return False


def plot_ssg(G, title="System Call Sequence Graph"):
    """Plot the System Call Sequence Graph with enhanced visualization"""
    try:
        # Extract interval number if present
        try:
            interval_num = int(''.join(filter(str.isdigit, title)))
            logger.info(f"Attempting to plot SSG for interval {interval_num}")
        except ValueError:
            interval_num = 0
            logger.info("Plotting SSG without interval number")

        # Skip plotting if interval doesn't match plot interval
        if interval_num % config.SSG_CONFIG['plot_interval'] != 0 and interval_num != 0:
            logger.debug(f"Skipping plot for interval {interval_num}")
            return

        # Clear any existing plots
        plt.close('all')

        # Create figure
        plt.figure(figsize=config.SSG_CONFIG['plot_size'])

        # Use spring layout for graph visualization
        spring_pos = nx.spring_layout(G, k=1.5, iterations=50)
        pos = spring_pos.copy()

        # Position special nodes
        if "UAN" in G.nodes:
            pos["UAN"] = np.array([1, 0])
        if "USN" in G.nodes:
            pos["USN"] = np.array([-1, 0])

        # Separate nodes by type
        normal_nodes = [node for node in G.nodes if node not in ["USN", "UAN"]]
        usn_nodes = ["USN"] if "USN" in G.nodes else []
        uan_nodes = ["UAN"] if "UAN" in G.nodes else []

        # Draw different node categories separately
        # Normal nodes
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=normal_nodes,
            node_color=config.SSG_CONFIG['node_colors']['seen'],
            node_size=config.SSG_CONFIG['node_sizes']['normal'],
            alpha=config.SSG_CONFIG['node_alpha']
        )

        # Unseen syscall nodes
        if usn_nodes:
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=usn_nodes,
                node_color=config.SSG_CONFIG['node_colors']['unseen_syscall'],
                node_size=config.SSG_CONFIG['node_sizes']['special'],
                alpha=config.SSG_CONFIG['node_alpha']
            )

        # Unseen argument nodes
        if uan_nodes:
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=uan_nodes,
                node_color=config.SSG_CONFIG['node_colors']['unseen_arg'],
                node_size=config.SSG_CONFIG['node_sizes']['special'],
                alpha=config.SSG_CONFIG['node_alpha']
            )

        # Draw edges
        nx.draw_networkx_edges(
            G, pos,
            alpha=config.SSG_CONFIG['edge_alpha'],
            edge_color='gray'
        )

        # Add labels with different sizes for special nodes
        normal_labels = {node: node for node in normal_nodes}
        special_labels = {node: node for node in usn_nodes + uan_nodes}

        nx.draw_networkx_labels(
            G, pos,
            labels=normal_labels,
            font_size=config.SSG_CONFIG['font_sizes']['node_labels']
        )
        nx.draw_networkx_labels(
            G, pos,
            labels=special_labels,
            font_size=config.SSG_CONFIG['font_sizes']['node_labels'] * 1.2  # Larger font for special nodes
        )

        # Set node sizes and colors
        node_sizes = [
            config.SSG_CONFIG['node_sizes']['special'] if node in config.SSG_CONFIG['special_nodes']
            else config.SSG_CONFIG['node_sizes']['normal']
            for node in G.nodes
        ]

        node_colors = []
        for node in G.nodes:
            if node == "USN":
                node_colors.append(config.SSG_CONFIG['node_colors']['unseen_syscall'])
            elif node == "UAN":
                node_colors.append(config.SSG_CONFIG['node_colors']['unseen_arg'])
            else:
                node_colors.append(config.SSG_CONFIG['node_colors']['seen'])

        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos,
            node_size=node_sizes,
            node_color=node_colors,
            alpha=config.SSG_CONFIG['node_alpha']
        )

        # Draw edges with weights
        edge_weights = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edges(
            G, pos,
            alpha=config.SSG_CONFIG['edge_alpha'],
            edge_color='gray'
        )

        # Add node labels
        node_labels = {
            node: node if len(node) < 15 else f"{node[:12]}..."
            for node in G.nodes
        }
        nx.draw_networkx_labels(
            G, pos,
            labels=node_labels,
            font_size=config.SSG_CONFIG['font_sizes']['node_labels']
        )

        # Add edge labels
        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels=edge_weights,
            font_size=config.SSG_CONFIG['font_sizes']['edge_labels']
        )

        # Set title and other plot parameters
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.title(
            f"{title}\n{timestamp}",
            fontsize=config.SSG_CONFIG['font_sizes']['title']
        )
        plt.axis('off')

        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=config.SSG_CONFIG['node_colors']['seen'],
                       label='Seen Syscall', markersize=10),
            plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=config.SSG_CONFIG['node_colors']['unseen_syscall'],
                       label='Unseen Syscall', markersize=10),
            plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=config.SSG_CONFIG['node_colors']['unseen_arg'],
                       label='Unseen Argument', markersize=10)
        ]
        plt.legend(
            handles=legend_elements,
            loc='center left',
            bbox_to_anchor=(1, 0.5),
            fontsize=config.SSG_CONFIG['font_sizes']['legend']
        )

        # Save the plot
        graphs_dir = os.path.join(
            config.GRAPHS_DIR,
            datetime.now().strftime("%Y%m%d")
        )
        os.makedirs(graphs_dir, exist_ok=True)

        plot_path = os.path.join(
            graphs_dir,
            f"ssg_{title.replace(' ', '_')}_{timestamp}.{config.SSG_CONFIG['save_format']}"
        )
        plt.savefig(
            plot_path,
            dpi=config.SSG_CONFIG['dpi'],
            bbox_inches='tight',
            format=config.SSG_CONFIG['save_format']
        )
        plt.close()

        logger.info(f"Successfully saved SSG plot to: {plot_path}")

    except Exception as e:
        logger.error(f"Error plotting SSG: {e}")
        logger.error(traceback.format_exc())
        plt.close('all')

def normalize_data(X_train):
    """Normalize training data using StandardScaler"""
    try:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        return scaler, X_train_scaled
    except Exception as e:
        logger.error(f"Error in data normalization: {e}")
        return None, None

def normalize_features(X, scaler):
    """Normalize features using provided scaler"""
    try:
        if scaler is None:
            logger.error("Scaler not provided for feature normalization")
            return None
        return scaler.transform(X)
    except Exception as e:
        logger.error(f"Error normalizing features: {e}")
        return None
