import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from .config import config
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import logging
from pathlib import Path
import os
from typing import Tuple
from .logging_setup import get_logger
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn




device = torch.device("cpu")
logger = logging.getLogger(__name__)

# ANSI color codes
RED = '\033[91m'
GREEN = '\033[92m'
BLUE = '\033[94m'
YELLOW = '\033[93m'
RESET = '\033[0m'

device = torch.device("cpu")
logger = logging.getLogger(__name__)



# class Autoencoder(nn.Module):
#     def __init__(self, input_dim):
#         super(Autoencoder, self).__init__()
#
#         # Encoder
#         self.encoder = nn.Sequential(
#             nn.Linear(input_dim, 256),
#             nn.LayerNorm(256),
#             nn.LeakyReLU(0.2),
#             nn.Dropout(0.3),
#
#             nn.Linear(256, 128),
#             nn.LayerNorm(128),
#             nn.LeakyReLU(0.2),
#             nn.Dropout(0.3),
#
#             nn.Linear(128, 64),
#             nn.LayerNorm(64)
#         )
#
#         # Decoder
#         self.decoder = nn.Sequential(
#             nn.Linear(64, 128),
#             nn.LayerNorm(128),
#             nn.LeakyReLU(0.2),
#             nn.Dropout(0.3),
#
#             nn.Linear(128, 256),
#             nn.LayerNorm(256),
#             nn.LeakyReLU(0.2),
#             nn.Dropout(0.3),
#
#             nn.Linear(256, input_dim),
#             nn.Tanh()
#         )
#
#     def forward(self, x):
#         encoded = self.encoder(x)
#         decoded = self.decoder(encoded)
#         return decoded


# class Autoencoder(nn.Module):
#     def __init__(self, input_dim):
#         super(Autoencoder, self).__init__()
#
#         cfg = config.MODEL_CONFIG
#
#         # Encoder
#         self.encoder = nn.Sequential(
#             nn.Linear(input_dim, cfg['encoder_dims'][0]),
#             nn.LayerNorm(cfg['encoder_dims'][0]),
#             nn.LeakyReLU(cfg['leaky_relu_slope']),
#             nn.Dropout(cfg['dropout_rate']),
#
#             nn.Linear(cfg['encoder_dims'][0], cfg['encoder_dims'][1]),
#             nn.LayerNorm(cfg['encoder_dims'][1]),
#             nn.LeakyReLU(cfg['leaky_relu_slope']),
#             nn.Dropout(cfg['dropout_rate']),
#
#             nn.Linear(cfg['encoder_dims'][1], cfg['encoder_dims'][2]),
#             nn.LayerNorm(cfg['encoder_dims'][2])
#         )
#
#         # Decoder (similar structure)
#         self.decoder = nn.Sequential(
#             nn.Linear(cfg['encoder_dims'][2], cfg['encoder_dims'][1]),
#             nn.LayerNorm(cfg['encoder_dims'][1]),
#             nn.LeakyReLU(cfg['leaky_relu_slope']),
#             nn.Dropout(cfg['dropout_rate']),
#
#             nn.Linear(cfg['encoder_dims'][1], cfg['encoder_dims'][0]),
#             nn.LayerNorm(cfg['encoder_dims'][0]),
#             nn.LeakyReLU(cfg['leaky_relu_slope']),
#             nn.Dropout(cfg['dropout_rate']),
#
#             nn.Linear(cfg['encoder_dims'][0], input_dim),
#             nn.Tanh()
#         )
#
#     def forward(self, x):
#         """Forward pass through the autoencoder"""
#         encoded = self.encoder(x)
#         decoded = self.decoder(encoded)
#         return decoded


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int):
        super(Autoencoder, self).__init__()

        # Calculate dimensions
        dim1 = max(input_dim * 2, 128)
        dim2 = max(input_dim, 64)
        dim3 = max(input_dim // 2, 32)

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, dim1),
            nn.BatchNorm1d(dim1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),  # Increased dropout

            nn.Linear(dim1, dim2),
            nn.BatchNorm1d(dim2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(dim2, dim3),
            nn.BatchNorm1d(dim3),
            nn.LeakyReLU(0.2)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(dim3, dim2),
            nn.BatchNorm1d(dim2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(dim2, dim1),
            nn.BatchNorm1d(dim1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(dim1, input_dim),
            nn.Tanh()
        )

    def forward(self, x):
        """Forward pass through autoencoder"""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# def train_autoencoder(X_train, X_val=None, epochs=300, batch_size=64):
#     """Enhanced autoencoder training with detailed monitoring"""
#     try:
#         if len(X_train) < batch_size:
#             raise ValueError(f"Training data size ({len(X_train)}) is less than batch size ({batch_size})")
#
#         input_dim = X_train.shape[1]
#         autoencoder = Autoencoder(input_dim).to(device)
#
#         optimizer = optim.AdamW(autoencoder.parameters(), lr=1e-5, weight_decay=1e-5)
#         scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
#
#         train_loader = DataLoader(
#             TensorDataset(torch.tensor(X_train, dtype=torch.float32)),
#             batch_size=batch_size,
#             shuffle=True
#         )
#
#         if X_val is not None:
#             val_loader = DataLoader(
#                 TensorDataset(torch.tensor(X_val, dtype=torch.float32)),
#                 batch_size=batch_size,
#                 shuffle=False
#             )
#
#         best_val_loss = float('inf')
#         trigger_times = 0
#         training_losses = []
#         validation_losses = []
#         best_epoch = 0
#
#         print(f"{BLUE}Starting Training:{RESET}")
#         print(f"Epochs: {epochs}")
#         print(f"Batch Size: {batch_size}")
#         print(f"Input Dimension: {input_dim}")
#         print("=" * 50)
#
#         for epoch in range(epochs):
#             autoencoder.train()
#             train_loss = 0.0
#             batch_count = 0
#
#             for data in train_loader:
#                 inputs = data[0].to(device)
#                 optimizer.zero_grad()
#                 outputs = autoencoder(inputs)
#                 loss = nn.MSELoss()(outputs, inputs)
#                 loss.backward()
#                 torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), max_norm=1.0)
#                 optimizer.step()
#                 train_loss += loss.item()
#                 batch_count += 1
#
#             avg_train_loss = train_loss / batch_count
#             training_losses.append(avg_train_loss)
#
#             if val_loader is not None:
#                 val_loss = validate_model(autoencoder, val_loader)
#                 validation_losses.append(val_loss)
#                 scheduler.step(val_loss)
#
#                 if val_loss < best_val_loss * 0.99:
#                     best_val_loss = val_loss
#                     torch.save(autoencoder.state_dict(), config.MODEL_FILE)
#                     trigger_times = 0
#                     best_epoch = epoch
#                 else:
#                     trigger_times += 1
#
#                 if epoch % 10 == 0:
#                     print(f"{YELLOW}Epoch {epoch + 1}/{epochs}:{RESET}")
#                     print(f"Training Loss: {avg_train_loss:.6f}")
#                     print(f"Validation Loss: {val_loss:.6f}")
#                     print("-" * 30)
#
#                 if trigger_times >= 15:
#                     print(f"{GREEN}Early stopping at epoch {epoch + 1}{RESET}")
#                     break
#
#         plot_training_history(training_losses, validation_losses)
#         return autoencoder, best_val_loss
#
#     except Exception as e:
#         logger.error(f"Error in training: {e}")
#         return None, None

def train_autoencoder(X_train, X_val=None, epochs=None, batch_size=None):
    """Enhanced autoencoder training with detailed monitoring using config parameters"""
    try:
        # Use config parameters with fallback defaults
        batch_size = batch_size or config.BATCH_SIZE
        epochs = epochs or config.NUM_EPOCHS

        if len(X_train) < batch_size:
            raise ValueError(f"Training data size ({len(X_train)}) is less than batch size ({batch_size})")

        input_dim = X_train.shape[1]
        autoencoder = Autoencoder(input_dim).to(device)

        optimizer = optim.AdamW(
            autoencoder.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=config.STOPPING_PATIENCE,
            verbose=True
        )

        train_loader = DataLoader(
            TensorDataset(torch.tensor(X_train, dtype=torch.float32)),
            batch_size=batch_size,
            shuffle=True
        )

        if X_val is not None:
            val_loader = DataLoader(
                TensorDataset(torch.tensor(X_val, dtype=torch.float32)),
                batch_size=batch_size,
                shuffle=False
            )

        best_val_loss = float('inf')
        trigger_times = 0
        training_losses = []
        validation_losses = []
        best_epoch = 0

        print(f"{BLUE}Starting Training:{RESET}")
        print(f"Epochs: {epochs}")
        print(f"Batch Size: {batch_size}")
        print(f"Learning Rate: {config.LEARNING_RATE}")
        print(f"Weight Decay: {config.WEIGHT_DECAY}")
        print(f"Early Stopping Patience: {config.EARLY_STOPPING_PATIENCE}")
        print(f"Input Dimension: {input_dim}")
        print("=" * 50)

        for epoch in range(epochs):
            autoencoder.train()
            train_loss = 0.0
            batch_count = 0

            for data in train_loader:
                inputs = data[0].to(device)
                optimizer.zero_grad()
                outputs = autoencoder(inputs)
                loss = nn.MSELoss()(outputs, inputs)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
                batch_count += 1

            avg_train_loss = train_loss / batch_count
            training_losses.append(avg_train_loss)

            if X_val is not None:
                val_loss = validate_model(autoencoder, val_loader)
                validation_losses.append(val_loss)
                scheduler.step(val_loss)

                if val_loss < best_val_loss * 0.99:  # 1% improvement threshold
                    best_val_loss = val_loss
                    torch.save(autoencoder.state_dict(), config.MODEL_FILE)
                    trigger_times = 0
                    best_epoch = epoch
                else:
                    trigger_times += 1

                if epoch % 10 == 0:
                    print(f"{YELLOW}Epoch {epoch + 1}/{epochs}:{RESET}")
                    print(f"Training Loss: {avg_train_loss:.6f}")
                    print(f"Validation Loss: {val_loss:.6f}")
                    print("-" * 30)

                # Early stopping check
                if trigger_times >= config.EARLY_STOPPING_PATIENCE:
                    print(f"{GREEN}Early stopping at epoch {epoch + 1}{RESET}")
                    break

                # Validation frequency check
                if epoch % config.VALIDATION_FREQUENCY == 0:
                    print(f"Validation Loss at epoch {epoch + 1}: {val_loss:.6f}")

        # Plot and save training history
        plot_training_history(training_losses, validation_losses)

        # Print final training summary
        print(f"\n{BLUE}Training Complete:{RESET}")
        print(f"Best epoch: {best_epoch + 1}")
        print(f"Best validation loss: {best_val_loss:.6f}")
        print(f"Final training loss: {avg_train_loss:.6f}")
        print("=" * 50)

        return autoencoder, best_val_loss

    except Exception as e:
        logger.error(f"Error in training: {e}")
        return None, None


def calculate_fixed_threshold(mse_values, percentile=99):
    """More robust threshold calculation"""
    if len(mse_values) == 0:
        logging.error("Empty MSE values")
        return None

    try:
        base_threshold = np.percentile(mse_values, percentile)
        mean_std_threshold = np.mean(mse_values) + 2 * np.std(mse_values)
        dynamic_threshold = min(base_threshold, mean_std_threshold)
        min_threshold = np.mean(mse_values) * 0.5
        return max(dynamic_threshold, min_threshold)
    except Exception as e:
        logging.error(f"Error calculating threshold: {e}")
        return None

# def detect_anomalies_with_fixed_threshold(autoencoder, X_test, threshold_trained, threshold_unseen, is_trained):
#     """Enhanced anomaly detection with detailed output"""
#     try:
#         autoencoder.eval()
#         with torch.no_grad():
#             X_tensor = torch.tensor(X_test, dtype=torch.float32)
#             reconstructions = autoencoder(X_tensor).cpu().numpy()
#
#             # Calculate reconstruction error
#             mse = np.mean(np.power(X_test - reconstructions, 2), axis=1)
#
#             # Determine threshold
#             threshold = threshold_trained if is_trained else threshold_unseen
#
#             # Detect anomalies
#             anomalies = mse > threshold
#
#             # Print detection summary
#             print(f"\n{GREEN}Detection Summary:{RESET}")
#             print(f"Samples processed: {len(X_test)}")
#             print(f"Anomalies detected: {np.sum(anomalies)}")
#             print(f"Average MSE: {np.mean(mse):.4f}")
#             print(f"Max MSE: {np.max(mse):.4f}")
#             #print(f"Threshold used: {threshold:.4f}")
#             print("=" * 200)
#
#             if np.sum(anomalies) > 0:
#                 print(f"\n{RED}Detailed Anomalies:{RESET}")
#                 for i, (is_anomaly, score) in enumerate(zip(anomalies, mse)):
#                     if is_anomaly:
#                         print(f"Sample {i}:")
#                         print(f"MSE Score: {score:.4f}")
#                         print(f"Threshold: {threshold:.4f}")
#                         print("-" * 200)
#
#             return anomalies, mse
#
#     except Exception as e:
#         logger.error(f"Error in anomaly detection: {e}")
#         return np.array([]), np.array([])

def detect_anomalies_with_fixed_threshold(model, data, threshold_trained, threshold_unseen, is_trained=True):
    """Detect anomalies using fixed thresholds"""
    try:
        with torch.no_grad():
            reconstructions = model(data).cpu().numpy()
            mse = np.mean(np.power(data.cpu().numpy() - reconstructions, 2), axis=1)
            threshold = threshold_trained if is_trained else threshold_unseen
            anomalies = (mse > threshold).astype(int)
            return anomalies, mse
    except Exception as e:
        logging.error(f"Error in anomaly detection: {e}")
        return None, None

def calculate_reconstruction_error(model, data):
    """Calculate reconstruction error for given data"""
    try:
        with torch.no_grad():
            reconstructions = model(data).cpu().numpy()
            mse = np.mean(np.power(data.cpu().numpy() - reconstructions, 2), axis=1)
            return mse
    except Exception as e:
        logging.error(f"Error calculating reconstruction error: {e}")
        return None


def load_training_data():
    """Load preprocessed training data"""
    try:
        logger = get_logger(__name__)
        logger.info("Loading training data...")

        train_data = []
        training_data_dir = config.DATA_STORAGE['training_data']
        feature_dir = os.path.join(training_data_dir, 'features')

        # Load all .npy files from the features directory
        for feature_file in Path(feature_dir).glob('*.npy'):
            try:
                data = np.load(feature_file)
                if len(data) > 0:
                    # Convert to tensor
                    data_tensor = torch.FloatTensor(data)
                    train_data.append(data_tensor)
                    logger.info(f"Loaded {len(data)} samples from {feature_file}")
            except Exception as e:
                logger.error(f"Error loading {feature_file}: {e}")
                continue

        if not train_data:
            raise ValueError("No training data found")

        # Combine all data
        train_data = torch.cat(train_data, dim=0)
        logger.info(f"Total training samples loaded: {len(train_data)}")

        # Create dataset
        dataset = TensorDataset(train_data)

        return dataset

    except Exception as e:
        logger.error(f"Error loading training data: {e}")
        raise



def validate_model(model, val_loader):
    """Validate the model"""
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for data in val_loader:
            inputs = data[0].to(device)
            outputs = model(inputs)
            loss = nn.MSELoss()(outputs, inputs)
            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def plot_training_history(training_losses, validation_losses):
    """Plot and save training history"""
    plt.figure(figsize=(10, 5))
    plt.plot(training_losses, label='Training Loss')
    plt.plot(validation_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_history.png')
    plt.close()



def load_autoencoder(model_file, input_dim):
    """Load trained autoencoder model with proper weights loading"""
    try:
        autoencoder = Autoencoder(input_dim).to(device)
        # Use weights_only=True to avoid the warning
        state_dict = torch.load(model_file, map_location=device, weights_only=True)
        autoencoder.load_state_dict(state_dict)
        autoencoder.eval()
        return autoencoder
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None

def normalize_data(X_train):
    """Normalize training data"""
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
