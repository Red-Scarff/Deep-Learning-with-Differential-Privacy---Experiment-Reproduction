"""
Main script to run all differential privacy experiments.
Reproduces the experiments from "Deep Learning with Differential Privacy" paper.
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict
import argparse

from data_loader import load_mnist_data
from dp_training import DPTrainer
from config import DP_CONFIGS, RESULTS_CONFIG, TRAINING_CONFIG, DATASET_CONFIG


class ExperimentRunner:
    """Runs and manages differential privacy experiments."""
    
    def __init__(self, results_dir: str = None):
        """
        Initialize experiment runner.
        
        Args:
            results_dir: Directory to save results
        """
        self.results_dir = results_dir or RESULTS_CONFIG['results_dir']
        self.models_dir = RESULTS_CONFIG['model_dir']
        
        # Create directories
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        
        self.all_results = []
        
    def run_single_experiment(self, dp_config: Dict, 
                            x_train: np.ndarray, y_train: np.ndarray,
                            x_val: np.ndarray, y_val: np.ndarray,
                            x_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Run a single experiment with given DP configuration.
        
        Args:
            dp_config: Differential privacy configuration
            x_train, y_train: Training data
            x_val, y_val: Validation data  
            x_test, y_test: Test data
            
        Returns:
            Experiment results dictionary
        """
        print(f"\n{'='*60}")
        print(f"Running experiment: {dp_config['name']}")
        print(f"{'='*60}")
        
        # Create trainer
        trainer = DPTrainer(dp_config, save_dir=self.results_dir)
        
        # Run training
        results = trainer.train_model(
            x_train, y_train,
            x_val, y_val, 
            x_test, y_test
        )
        
        # Save model if configured
        if RESULTS_CONFIG['save_models']:
            model_path = os.path.join(
                self.models_dir, 
                f"{dp_config['name']}_model.h5"
            )
            trainer.model.save(model_path)
            results['model_path'] = model_path
            print(f"Model saved to: {model_path}")
        
        return results
    
    def run_all_experiments(self, configs: List[Dict] = None) -> List[Dict]:
        """
        Run all experiments with different DP configurations.
        
        Args:
            configs: List of DP configurations to run
            
        Returns:
            List of all experiment results
        """
        if configs is None:
            configs = DP_CONFIGS
            
        print("Loading MNIST dataset...")
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_mnist_data()
        
        print(f"\nDataset information:")
        print(f"  Training samples: {len(x_train)}")
        print(f"  Validation samples: {len(x_val) if x_val is not None else 0}")
        print(f"  Test samples: {len(x_test)}")
        print(f"  Batch size: {DATASET_CONFIG['batch_size']}")
        print(f"  Epochs: {TRAINING_CONFIG['epochs']}")
        
        # Run experiments
        self.all_results = []
        for i, dp_config in enumerate(configs):
            print(f"\n[{i+1}/{len(configs)}] Starting experiment: {dp_config['name']}")
            
            try:
                results = self.run_single_experiment(
                    dp_config, x_train, y_train, x_val, y_val, x_test, y_test
                )
                self.all_results.append(results)
                
                # Print summary
                epsilon = results['privacy_spent']['epsilon']
                test_acc = results['test_accuracy']
                print(f"Completed: ε={epsilon:.3f}, Test Accuracy={test_acc:.4f}")
                
            except Exception as e:
                print(f"Error in experiment {dp_config['name']}: {str(e)}")
                continue
        
        # Save summary results
        self._save_summary_results()
        
        return self.all_results
    
    def _save_summary_results(self):
        """Save summary of all experiment results."""
        if not self.all_results:
            return
            
        # Create summary DataFrame
        summary_data = []
        for result in self.all_results:
            summary_data.append({
                'experiment_name': result['config_name'],
                'epsilon': result['privacy_spent']['epsilon'],
                'delta': result['privacy_spent']['delta'],
                'l2_norm_clip': result['dp_config'].get('l2_norm_clip', None),
                'noise_multiplier': result['dp_config']['noise_multiplier'],
                'final_train_accuracy': result['final_train_accuracy'],
                'final_val_accuracy': result['final_val_accuracy'],
                'test_accuracy': result['test_accuracy'],
                'training_time': result['training_time']
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(self.results_dir, f"experiment_summary_{timestamp}.csv")
        summary_df.to_csv(csv_path, index=False)
        
        # Save to JSON
        json_path = os.path.join(self.results_dir, f"all_results_{timestamp}.json")
        with open(json_path, 'w') as f:
            json.dump(self.all_results, f, indent=2)
        
        print(f"\nSummary results saved:")
        print(f"  CSV: {csv_path}")
        print(f"  JSON: {json_path}")
        
        # Print summary table
        print(f"\n{'='*80}")
        print("EXPERIMENT SUMMARY")
        print(f"{'='*80}")
        print(summary_df.to_string(index=False, float_format='%.4f'))
    
    def get_results_summary(self) -> pd.DataFrame:
        """Get summary of results as DataFrame."""
        if not self.all_results:
            return pd.DataFrame()
            
        summary_data = []
        for result in self.all_results:
            summary_data.append({
                'experiment': result['config_name'],
                'epsilon': result['privacy_spent']['epsilon'],
                'delta': result['privacy_spent']['delta'],
                'test_accuracy': result['test_accuracy'],
                'noise_multiplier': result['dp_config']['noise_multiplier']
            })
        
        return pd.DataFrame(summary_data)


def main():
    """Main function to run experiments."""
    parser = argparse.ArgumentParser(description='Run DP experiments')
    parser.add_argument('--config', type=str, help='Specific config to run')
    parser.add_argument('--results-dir', type=str, default='results', 
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Initialize experiment runner
    runner = ExperimentRunner(results_dir=args.results_dir)
    
    # Select configurations to run
    if args.config:
        configs = [cfg for cfg in DP_CONFIGS if cfg['name'] == args.config]
        if not configs:
            print(f"Configuration '{args.config}' not found!")
            return
    else:
        configs = DP_CONFIGS
    
    print("Starting differential privacy experiments...")
    print(f"Configurations to run: {[cfg['name'] for cfg in configs]}")
    
    # Run experiments
    results = runner.run_all_experiments(configs)
    
    print(f"\nAll experiments completed!")
    print(f"Total experiments run: {len(results)}")
    
    # Show final summary
    summary_df = runner.get_results_summary()
    if not summary_df.empty:
        print("\nFinal Results Summary:")
        print(summary_df.to_string(index=False, float_format='%.4f'))


if __name__ == "__main__":
    main()
