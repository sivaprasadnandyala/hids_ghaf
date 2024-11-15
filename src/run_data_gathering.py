import os
import argparse
from pathlib import Path
from .data_gathering import DataCollector

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', default=os.getenv('PROJECT_BASE_DIR'))
    parser.add_argument('--tetragon-bpf', default=os.getenv('TETRAGON_BPF_PATH'))
    parser.add_argument('--policy-file', default=os.getenv('POLICY_FILE'))
    parser.add_argument('--duration', type=int, default=600)
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Update config paths
    os.environ['PROJECT_BASE_DIR'] = args.base_dir
    os.environ['TETRAGON_BPF_PATH'] = args.tetragon_bpf
    os.environ['POLICY_FILE'] = args.policy_file

    collector = DataCollector()
    collector.run_collection()

if __name__ == "__main__":
    main()
