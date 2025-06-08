"""
Visualization module for differential privacy experiment results.
Creates publication-quality plots similar to the DP paper.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
import os
from typing import List, Dict, Optional
import glob

from config import RESULTS_CONFIG


class DPResultsVisualizer:
    """Visualizer for differential privacy experiment results."""
    
    def __init__(self, results_dir: str = None, plots_dir: str = None):
        """
        Initialize visualizer.
        
        Args:
            results_dir: Directory containing experiment results
            plots_dir: Directory to save plots
        """
        self.results_dir = results_dir or RESULTS_CONFIG['results_dir']
        self.plots_dir = plots_dir or RESULTS_CONFIG['plots_dir']
        
        # Create plots directory
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def load_results(self, results_file: str = None) -> List[Dict]:
        """
        Load experiment results from JSON file.
        
        Args:
            results_file: Specific results file to load
            
        Returns:
            List of experiment results
        """
        if results_file:
            with open(results_file, 'r') as f:
                return json.load(f)
        else:
            # Find the most recent results file
            pattern = os.path.join(self.results_dir, "all_results_*.json")
            files = glob.glob(pattern)
            if not files:
                raise FileNotFoundError("No results files found!")
            
            latest_file = max(files, key=os.path.getctime)
            print(f"Loading results from: {latest_file}")
            
            with open(latest_file, 'r') as f:
                return json.load(f)
    
    def create_privacy_accuracy_plot(self, results: List[Dict], 
                                   save_path: str = None) -> plt.Figure:
        """
        Create privacy vs accuracy plot (main result from DP paper).
        
        Args:
            results: List of experiment results
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        # Prepare data
        data = []
        for result in results:
            epsilon = result['privacy_spent']['epsilon']
            if epsilon == float('inf'):
                epsilon = 100  # For plotting non-private baseline
            
            data.append({
                'epsilon': epsilon,
                'test_accuracy': result['test_accuracy'],
                'experiment': result['config_name'],
                'noise_multiplier': result['dp_config']['noise_multiplier']
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values('epsilon')
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot points
        colors = sns.color_palette("husl", len(df))
        for i, (_, row) in enumerate(df.iterrows()):
            if row['epsilon'] >= 100:
                # Non-private baseline
                ax.scatter(row['epsilon'], row['test_accuracy'], 
                          s=100, color='red', marker='*', 
                          label='Non-private', zorder=5)
            else:
                ax.scatter(row['epsilon'], row['test_accuracy'], 
                          s=80, color=colors[i], 
                          label=f"σ={row['noise_multiplier']}", zorder=4)
        
        # Connect points with line (excluding non-private)
        private_df = df[df['epsilon'] < 100].sort_values('epsilon')
        if len(private_df) > 1:
            ax.plot(private_df['epsilon'], private_df['test_accuracy'], 
                   'b--', alpha=0.7, linewidth=2, zorder=3)
        
        # Formatting
        ax.set_xlabel('Privacy Budget (ε)', fontsize=12)
        ax.set_ylabel('Test Accuracy', fontsize=12)
        ax.set_title('Privacy-Accuracy Trade-off\n(Deep Learning with Differential Privacy)', 
                    fontsize=14, fontweight='bold')
        
        # Set x-axis to log scale for better visualization
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        
        # Add annotations
        for _, row in df.iterrows():
            if row['epsilon'] < 100:
                ax.annotate(f"ε={row['epsilon']:.1f}", 
                           (row['epsilon'], row['test_accuracy']),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=9, alpha=0.8)
        
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = os.path.join(self.plots_dir, 'privacy_accuracy_tradeoff.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Privacy-accuracy plot saved to: {save_path}")
        
        return fig
    
    def create_training_curves(self, results: List[Dict], 
                             save_path: str = None) -> plt.Figure:
        """
        Create training curves for different privacy settings.
        
        Args:
            results: List of experiment results
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, result in enumerate(results[:4]):  # Show first 4 experiments
            if i >= 4:
                break
                
            ax = axes[i]
            history = result['history']
            epochs = range(1, len(history['accuracy']) + 1)
            
            # Plot training and validation accuracy
            ax.plot(epochs, history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
            if history['val_accuracy']:
                ax.plot(epochs, history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
            
            # Formatting
            epsilon = result['privacy_spent']['epsilon']
            if epsilon == float('inf'):
                title = f"{result['config_name']} (Non-private)"
            else:
                title = f"{result['config_name']} (ε={epsilon:.2f})"
            
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
        
        # Hide unused subplots
        for j in range(len(results), 4):
            axes[j].set_visible(False)
        
        plt.suptitle('Training Curves for Different Privacy Settings', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = os.path.join(self.plots_dir, 'training_curves.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves plot saved to: {save_path}")
        
        return fig
    
    def create_noise_impact_plot(self, results: List[Dict], 
                               save_path: str = None) -> plt.Figure:
        """
        Create plot showing impact of noise multiplier on accuracy.
        
        Args:
            results: List of experiment results
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        # Prepare data
        data = []
        for result in results:
            if result['dp_config']['noise_multiplier'] > 0:  # Only DP experiments
                data.append({
                    'noise_multiplier': result['dp_config']['noise_multiplier'],
                    'test_accuracy': result['test_accuracy'],
                    'epsilon': result['privacy_spent']['epsilon']
                })
        
        if not data:
            print("No DP experiments found for noise impact plot")
            return None
        
        df = pd.DataFrame(data)
        df = df.sort_values('noise_multiplier')
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot with color coding by epsilon
        scatter = ax.scatter(df['noise_multiplier'], df['test_accuracy'], 
                           c=df['epsilon'], s=100, cmap='viridis_r', 
                           alpha=0.8, edgecolors='black', linewidth=1)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Privacy Budget (ε)', fontsize=12)
        
        # Connect points with line
        ax.plot(df['noise_multiplier'], df['test_accuracy'], 
               'r--', alpha=0.7, linewidth=2)
        
        # Formatting
        ax.set_xlabel('Noise Multiplier (σ)', fontsize=12)
        ax.set_ylabel('Test Accuracy', fontsize=12)
        ax.set_title('Impact of Noise Multiplier on Model Accuracy', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add annotations
        for _, row in df.iterrows():
            ax.annotate(f"ε={row['epsilon']:.1f}", 
                       (row['noise_multiplier'], row['test_accuracy']),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=9, alpha=0.8)
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = os.path.join(self.plots_dir, 'noise_impact.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Noise impact plot saved to: {save_path}")
        
        return fig
    
    def create_summary_table(self, results: List[Dict], 
                           save_path: str = None) -> pd.DataFrame:
        """
        Create summary table of all experiments.
        
        Args:
            results: List of experiment results
            save_path: Path to save the table
            
        Returns:
            Summary DataFrame
        """
        # Prepare data
        data = []
        for result in results:
            epsilon = result['privacy_spent']['epsilon']
            if epsilon == float('inf'):
                epsilon_str = "∞"
            else:
                epsilon_str = f"{epsilon:.2f}"
            
            data.append({
                'Experiment': result['config_name'],
                'ε': epsilon_str,
                'δ': f"{result['privacy_spent']['delta']:.0e}",
                'Noise Multiplier': result['dp_config']['noise_multiplier'],
                'L2 Clip': result['dp_config'].get('l2_norm_clip', 'N/A'),
                'Test Accuracy': f"{result['test_accuracy']:.4f}",
                'Training Time (s)': f"{result['training_time']:.1f}"
            })
        
        df = pd.DataFrame(data)
        
        # Save to CSV
        if save_path is None:
            save_path = os.path.join(self.plots_dir, 'results_summary.csv')
        
        df.to_csv(save_path, index=False)
        print(f"Summary table saved to: {save_path}")
        
        return df
    
    def generate_all_plots(self, results_file: str = None):
        """
        Generate all visualization plots.
        
        Args:
            results_file: Specific results file to use
        """
        print("Generating all visualization plots...")
        
        # Load results
        results = self.load_results(results_file)
        
        # Generate plots
        self.create_privacy_accuracy_plot(results)
        self.create_training_curves(results)
        self.create_noise_impact_plot(results)
        self.create_summary_table(results)
        
        print(f"All plots saved to: {self.plots_dir}")


if __name__ == "__main__":
    # Test visualization
    visualizer = DPResultsVisualizer()
    
    try:
        visualizer.generate_all_plots()
    except FileNotFoundError:
        print("No results files found. Run experiments first with run_experiments.py")
