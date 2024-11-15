#!/usr/bin/env python3
import os
import sys
import psutil
from pathlib import Path
import traceback
from .logging_setup import get_logger
from .config import config
from .data_gathering import DataCollector


def main():
    logger = get_logger(__name__)

    try:
        collector = DataCollector()

        # Verify all processes are running
        missing_processes = []
        for process in config.TESTING_PROCESSES:
            if not any(process in ' '.join(p.cmdline()) for p in psutil.process_iter(['cmdline'])):
                missing_processes.append(process)

        if missing_processes:
            logger.error("The following processes are not running:")
            for proc in missing_processes:
                logger.error(f"- {proc}")
            logger.error("Please start these processes before collecting data")
            return False

        # Start collection
        logger.info("Starting test data collection...")
        if collector.collect_test_data():
            logger.info("Test data collection completed successfully")

            # Verify collected data
            if collector.verify_collected_data():
                logger.info("Data verification successful")
                return True
            else:
                logger.error("Data verification failed")
                return False
        else:
            logger.error("Test data collection failed")
            return False

    except Exception as e:
        logger.error(f"Error in data collection: {e}")
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    main()
