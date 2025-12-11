#!/usr/bin/env python3
"""
Create experiment summary CSV from model training results.
"""

import argparse
import json
import pandas as pd
import os
from pathlib import Path


def get_experiment_param(config, module_name, exp_name, param_name):
    """Get parameter for an experiment, falling back to defaults."""
    defaults = config["modules"][module_name]["config"]["default_training"]
    experiment_params = config["modules"][module_name]["config"]["experiments"]
    
    # Find the specific experiment
    experiment = next((exp for exp in experiment_params if exp['name'] == exp_name), {})
    
    # Handle nested parameters like augmentation
    if '.' in param_name:
        main_param, sub_param = param_name.split('.', 1)
        if main_param in experiment:
            return experiment[main_param].get(sub_param, defaults.get(main_param, {}).get(sub_param))
        return defaults.get(main_param, {}).get(sub_param)
    
    return experiment.get(param_name, defaults.get(param_name))


def create_experiment_summary(config_files, output_file, config_data, module_name):
    """Create experiment summary CSV from config files."""
    
    # Create the experiment lookup dictionary
    experiments_dict = {
        exp['name']: exp 
        for exp in config_data["modules"][module_name]["config"]["experiments"]
    }
    
    all_results = []
    for config_file in config_files:
        # Extract experiment name from path
        exp_name = Path(config_file).parent.name
        
        # Load experiment results
        with open(config_file, 'r') as f:
            experiment_data = json.load(f)
        
        # Get experiment parameters
        exp_params = experiments_dict[exp_name]
        
        # Combine into single row
        row = {
            'experiment_name': exp_name,
            'description': exp_params.get('description', ''),
            'learning_rate': exp_params.get('learning_rate', get_experiment_param(config_data, module_name, exp_name, 'learning_rate')),
            'num_epochs': exp_params.get('num_epochs', get_experiment_param(config_data, module_name, exp_name, 'num_epochs')),
            'batch_size': exp_params.get('batch_size', get_experiment_param(config_data, module_name, exp_name, 'batch_size')),
            'total_parameters': experiment_data.get('total_parameters', 'N/A'),
            'device': experiment_data.get('device', 'N/A'),
        }
        all_results.append(row)
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv(output_file, index=False)
    
    print(f"Experiment summary saved to {output_file}")
    print(f"Completed {len(all_results)} experiments")
    
    return len(all_results)


def main():
    parser = argparse.ArgumentParser(description='Create experiment summary CSV')
    parser.add_argument('--config-files', nargs='+', required=True,
                        help='List of experiment config.json files')
    parser.add_argument('--output', required=True,
                        help='Output CSV file path')
    parser.add_argument('--pipeline-config', required=True,
                        help='Pipeline config YAML file')
    parser.add_argument('--module-name', default='model_training',
                        help='Module name (default: model_training)')
    
    args = parser.parse_args()
    
    # Load pipeline config
    import yaml
    with open(args.pipeline_config, 'r') as f:
        config_data = yaml.safe_load(f)
    
    # Create summary
    create_experiment_summary(
        config_files=args.config_files,
        output_file=args.output,
        config_data=config_data,
        module_name=args.module_name
    )


if __name__ == "__main__":
    main()