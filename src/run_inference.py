import os
import argparse
from pathlib import Path
from .realtime_inference import RealTimeDetector

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', default=os.getenv('PROJECT_BASE_DIR'))
    parser.add_argument('--tetragon-bpf', default=os.getenv('TETRAGON_BPF_PATH'))
    parser.add_argument('--policy-file', default=os.getenv('POLICY_FILE'))
    parser.add_argument('--model-path', default=None)
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Update config paths
    os.environ['PROJECT_BASE_DIR'] = args.base_dir
    os.environ['TETRAGON_BPF_PATH'] = args.tetragon_bpf
    os.environ['POLICY_FILE'] = args.policy_file

    detector = RealTimeDetector()
    if detector.load_model_artifacts():
        detector.start_monitoring()

if __name__ == "__main__":
    main()
