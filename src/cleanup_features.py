import os
import shutil
from datetime import datetime
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def cleanup_feature_files():
    """Clean up feature files with proper error handling"""
    try:
        # Base directory
        base_dir = "/home/sivaprasad/PycharmProjects/HIDS_Modules/data"

        # Create backup directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = os.path.join(base_dir, f"backup_{timestamp}")
        os.makedirs(backup_dir, exist_ok=True)

        # Directories to clean
        directories = [
            os.path.join(base_dir, "testing", "features"),
            os.path.join(base_dir, "training", "features")
        ]

        for directory in directories:
            if os.path.exists(directory):
                # Create backup
                backup_path = os.path.join(backup_dir, os.path.basename(os.path.dirname(directory)), "features")
                logger.info(f"Creating backup at: {backup_path}")

                try:
                    shutil.copytree(directory, backup_path)
                    logger.info(f"Backup created for: {directory}")
                except Exception as e:
                    logger.error(f"Backup failed for {directory}: {e}")
                    continue

                # Remove files
                try:
                    # First try using Path
                    feature_path = Path(directory)
                    for file in feature_path.glob("*"):
                        try:
                            if file.is_file():
                                os.remove(file)
                                logger.info(f"Removed file: {file}")
                        except Exception as e:
                            logger.error(f"Failed to remove {file}: {e}")

                    # If directory still has files, try rmtree
                    if any(feature_path.iterdir()):
                        shutil.rmtree(directory)
                        os.makedirs(directory, exist_ok=True)
                        logger.info(f"Recreated directory: {directory}")

                except Exception as e:
                    logger.error(f"Failed to clean directory {directory}: {e}")

                # Set proper permissions
                try:
                    os.system(f"sudo chown -R sivaprasad:sivaprasad {directory}")
                    os.system(f"sudo chmod -R 755 {directory}")
                    logger.info(f"Set permissions for: {directory}")
                except Exception as e:
                    logger.error(f"Failed to set permissions for {directory}: {e}")

        logger.info("Cleanup completed")
        return True

    except Exception as e:
        logger.error(f"Critical error during cleanup: {e}")
        return False


if __name__ == "__main__":
    cleanup_feature_files()