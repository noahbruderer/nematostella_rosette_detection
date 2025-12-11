#!/usr/bin/env python3
"""
Create train/val split for rosette detection model training.

This script discovers all available training samples and creates a reproducible
train/validation split, saving the results to config files for Snakemake.
"""

import argparse
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split



def create_train_val_split(sample_ids, test_size=0.2, random_state=42):
    """Create train/validation split and return sample lists."""
    
    # Sort sample IDs for consistent ordering across runs
    sorted_sample_ids = sorted(sample_ids)
    
    train_ids, val_ids = train_test_split(
        sorted_sample_ids, 
        test_size=test_size, 
        random_state=random_state
    )
    
    print(f"Created train/val split:")
    print(f"  Training samples: {len(train_ids)}")
    print(f"  Validation samples: {len(val_ids)}")
    
    return sorted(train_ids), sorted(val_ids)


def create_split_manifest(config_dir, sample_ids, train_ids, val_ids, test_size, random_state, output_name="split_manifest.json"):
    """Create a manifest file with split metadata for tracking."""
    manifest = {
        "split_created": datetime.now().isoformat(),
        "total_samples": len(sample_ids),
        "train_samples": len(train_ids),
        "val_samples": len(val_ids),
        "test_size": test_size,
        "random_state": random_state,
        "sample_order_hash": hash(tuple(sorted(sample_ids))),
        "train_sample_list": sorted(train_ids),
        "val_sample_list": sorted(val_ids)
    }
    
    manifest_file = config_dir / output_name
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"Created split manifest: {manifest_file}")
    return manifest_file


def main():
    parser = argparse.ArgumentParser(
        description="Create train/validation split for model training"
    )
    parser.add_argument(
        "--samples",
        nargs='+',
        required=True,
        help="List of sample IDs to split"
    )
    parser.add_argument(
        "--config-dir",
        default="config",
        help="Directory to save config files (default: config)"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data to use for validation (default: 0.2)"
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible splits (default: 42)"
    )
    parser.add_argument(
        "--output-name",
        default="split_manifest.json",
        help="Output filename for the split manifest (default: split_manifest.json)"
    )
    parser.add_argument(
        "--config-file",
        help="Path to module config file to load experiment parameters"
    )
    parser.add_argument(
        "--experiment-name", 
        help="Name of the experiment to get parameters for"
    )
    
    args = parser.parse_args()
    
    # Load config if provided to get experiment-specific parameters
    if args.config_file and args.experiment_name:
        import yaml
        with open(args.config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Find the experiment
        experiments = config.get('experiments', [])
        experiment = next((exp for exp in experiments if exp['name'] == args.experiment_name), {})
        
        # Get experiment-specific parameters, fallback to defaults
        defaults = config.get('default_training', {})
        test_size = experiment.get('val_split', defaults.get('val_split', args.test_size))
        random_state = experiment.get('seed', defaults.get('seed', args.random_state))
        
        print(f"Using experiment '{args.experiment_name}' parameters:")
        print(f"  val_split: {test_size}")
        print(f"  seed: {random_state}")
    else:
        test_size = args.test_size
        random_state = args.random_state
    
    # Create config directory if it doesn't exist
    config_dir = Path(args.config_dir)
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # Use provided samples directly, ensure consistent sorting
    sample_ids = sorted(args.samples)
    print(f"Using {len(sample_ids)} provided samples")
    
    if not sample_ids:
        print(f"ERROR: No samples provided")
        return 1
    
    # Create train/val split
    train_ids, val_ids = create_train_val_split(
        sample_ids,
        test_size=test_size,
        random_state=random_state
    )
    
    # Create split manifest for tracking
    create_split_manifest(
        config_dir, 
        sample_ids, 
        train_ids, 
        val_ids, 
        test_size, 
        random_state,
        args.output_name
    )
    
    print("\nSplit summary:")
    print(f"  Total samples: {len(sample_ids)}")
    print(f"  Training: {len(train_ids)} ({len(train_ids)/len(sample_ids)*100:.1f}%)")
    print(f"  Validation: {len(val_ids)} ({len(val_ids)/len(sample_ids)*100:.1f}%)")
    print(f"  Split is reproducible with random_state={args.random_state}")
    
    return 0


if __name__ == "__main__":
    exit(main())