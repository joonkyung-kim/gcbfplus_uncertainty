#!/usr/bin/env python3
"""
Comprehensive noise robustness test script for GCBF+.
This script runs the edge noise test across multiple noise levels and generates plots.
"""

import argparse
import datetime
import os
import subprocess
import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple
import json

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")


def run_single_test(args: argparse.Namespace, noise_std: float) -> Dict:
    """Run a single noise test for a given noise standard deviation."""
    
    cmd = [
        sys.executable, "test_noise.py",
        "--path", args.path,
        "--area-size", str(args.area_size),
        "--edge-noise-std", str(noise_std),
        "--epi", str(args.episodes_per_test),
        "--seed", str(args.seed),
        "--no-video"  # Disable video generation for batch testing
    ]
    
    print(f"Running test with noise_std={noise_std:.1f}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            print(f"Error running test with noise_std={noise_std}: {result.stderr}")
            return None
        
        # Parse the output to extract results
        lines = result.stdout.split('\n')
        results = {
            'noise_std': noise_std,
            'safe_rate': 0.0,
            'finish_rate': 0.0,
            'success_rate': 0.0,
            'reward_mean': 0.0,
            'reward_std': 0.0,
            'cost_mean': 0.0,
            'cost_std': 0.0
        }
        
        # Parse results from output
        for line in lines:
            if "Safe rate:" in line:
                parts = line.split()
                results['safe_rate'] = float(parts[2].replace('%', ''))
            elif "Finish rate:" in line:
                parts = line.split()
                results['finish_rate'] = float(parts[2].replace('%', ''))
            elif "Success rate:" in line:
                parts = line.split()
                results['success_rate'] = float(parts[2].replace('%', ''))
            elif "Reward:" in line:
                parts = line.split()
                results['reward_mean'] = float(parts[1])
                results['reward_std'] = float(parts[3])
            elif "Cost:" in line:
                parts = line.split()
                results['cost_mean'] = float(parts[1])
                results['cost_std'] = float(parts[3])
        
        return results
        
    except subprocess.TimeoutExpired:
        print(f"Test with noise_std={noise_std} timed out")
        return None
    except Exception as e:
        print(f"Error running test with noise_std={noise_std}: {e}")
        return None


def run_robustness_tests(args: argparse.Namespace) -> pd.DataFrame:
    """Run robustness tests across multiple noise levels."""
    
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    all_results = []
    
    print(f"Running robustness tests for noise levels: {noise_levels}")
    print(f"Episodes per test: {args.episodes_per_test}")
    print(f"Using model: {args.path}")
    print("=" * 60)
    
    start_time = time.time()
    
    # Run tests sequentially for better resource management
    for noise_std in noise_levels:
        result = run_single_test(args, noise_std)
        if result is not None:
            all_results.append(result)
            print(f"✓ Completed noise_std={noise_std:.1f}: "
                  f"safe={result['safe_rate']:.1f}%, "
                  f"finish={result['finish_rate']:.1f}%, "
                  f"success={result['success_rate']:.1f}%")
        else:
            print(f"✗ Failed noise_std={noise_std:.1f}")
    
    elapsed_time = time.time() - start_time
    print(f"\nCompleted all tests in {elapsed_time:.1f} seconds")
    
    if not all_results:
        raise ValueError("No tests completed successfully")
    
    return pd.DataFrame(all_results)


def create_plots(df: pd.DataFrame, save_dir: Path):
    """Create comprehensive plots of the robustness test results."""
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('GCBF+ Robustness to Edge Feature Noise', fontsize=16, fontweight='bold')
    
    # Colors for different metrics
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # Plot 1: Safety, Finish, and Success Rates
    ax1 = axes[0, 0]
    ax1.plot(df['noise_std'], df['safe_rate'], 'o-', color=colors[0], label='Safe Rate', linewidth=2, markersize=8)
    ax1.plot(df['noise_std'], df['finish_rate'], 's-', color=colors[1], label='Finish Rate', linewidth=2, markersize=8)
    ax1.plot(df['noise_std'], df['success_rate'], '^-', color=colors[2], label='Success Rate', linewidth=2, markersize=8)
    ax1.set_xlabel('Edge Noise Standard Deviation')
    ax1.set_ylabel('Rate (%)')
    ax1.set_title('Performance Metrics vs Noise Level')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 105)
    
    # Plot 2: Reward Statistics
    ax2 = axes[0, 1]
    ax2.errorbar(df['noise_std'], df['reward_mean'], yerr=df['reward_std'], 
                 fmt='o-', color=colors[3], label='Reward', linewidth=2, markersize=8, capsize=5)
    ax2.set_xlabel('Edge Noise Standard Deviation')
    ax2.set_ylabel('Average Reward')
    ax2.set_title('Reward vs Noise Level')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Cost Statistics
    ax3 = axes[1, 0]
    if df['cost_mean'].sum() > 0:  # Only plot if there are non-zero costs
        ax3.errorbar(df['noise_std'], df['cost_mean'], yerr=df['cost_std'], 
                     fmt='o-', color=colors[3], label='Cost', linewidth=2, markersize=8, capsize=5)
        ax3.set_ylabel('Average Cost')
    else:
        ax3.text(0.5, 0.5, 'No costs incurred\n(Perfect safety)', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        ax3.set_ylabel('Average Cost')
    ax3.set_xlabel('Edge Noise Standard Deviation')
    ax3.set_title('Cost vs Noise Level')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Performance Degradation
    ax4 = axes[1, 1]
    baseline_success = df.loc[df['noise_std'] == 0.0, 'success_rate'].iloc[0]
    degradation = baseline_success - df['success_rate']
    ax4.plot(df['noise_std'], degradation, 'o-', color=colors[3], 
             label='Success Rate Degradation', linewidth=2, markersize=8)
    ax4.set_xlabel('Edge Noise Standard Deviation')
    ax4.set_ylabel('Performance Degradation (%)')
    ax4.set_title('Performance Degradation vs Noise Level')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(bottom=0)
    
    # Adjust layout and save
    plt.tight_layout()
    plot_path = save_dir / 'robustness_analysis.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")
    
    # Create a detailed summary table plot
    fig2, ax = plt.subplots(figsize=(12, 8))
    
    # Create a summary table
    table_data = df[['noise_std', 'safe_rate', 'finish_rate', 'success_rate', 'reward_mean']].round(2)
    table_data.columns = ['Noise Std', 'Safe Rate (%)', 'Finish Rate (%)', 'Success Rate (%)', 'Avg Reward']
    
    # Create the table
    table = ax.table(cellText=table_data.values, colLabels=table_data.columns, 
                    cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)
    
    # Style the table
    for i in range(len(table_data.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax.axis('off')
    ax.set_title('Robustness Test Results Summary', fontsize=16, fontweight='bold', pad=20)
    
    table_path = save_dir / 'robustness_table.png'
    plt.savefig(table_path, dpi=300, bbox_inches='tight')
    print(f"Table saved to: {table_path}")
    
    plt.show()


def save_results(df: pd.DataFrame, save_dir: Path, args: argparse.Namespace):
    """Save results to CSV and JSON files."""
    
    # Save to CSV
    csv_path = save_dir / 'robustness_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"Results saved to: {csv_path}")
    
    # Save metadata
    metadata = {
        'timestamp': datetime.datetime.now().isoformat(),
        'model_path': args.path,
        'episodes_per_test': args.episodes_per_test,
        'area_size': args.area_size,
        'seed': args.seed,
        'noise_levels': df['noise_std'].tolist(),
        'summary': {
            'baseline_success_rate': float(df.loc[df['noise_std'] == 0.0, 'success_rate'].iloc[0]),
            'max_degradation': float((df.loc[df['noise_std'] == 0.0, 'success_rate'].iloc[0] - df['success_rate'].min())),
            'noise_tolerance': float(df.loc[df['success_rate'] >= 80.0, 'noise_std'].max()) if len(df.loc[df['success_rate'] >= 80.0]) > 0 else 0.0
        }
    }
    
    json_path = save_dir / 'robustness_metadata.json'
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {json_path}")


def main():
    parser = argparse.ArgumentParser(description="Run comprehensive robustness tests for GCBF+ with edge noise")
    
    # Required arguments
    parser.add_argument("--path", type=str, required=True, help="Path to trained GCBF+ model")
    parser.add_argument("--area-size", type=float, required=True, help="Area size for the environment")
    
    # Test parameters
    parser.add_argument("--episodes-per-test", type=int, default=100, help="Number of episodes per noise level test")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    
    # Output parameters
    parser.add_argument("--output-dir", type=str, default="robustness_results", help="Output directory for results")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.path):
        raise ValueError(f"Model path does not exist: {args.path}")
    
    if not os.path.exists(os.path.join(args.path, "config.yaml")):
        raise ValueError(f"Config file not found: {os.path.join(args.path, 'config.yaml')}")
    
    # Create output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(args.output_dir) / f"robustness_test_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting robustness test...")
    print(f"Output directory: {save_dir}")
    print(f"Episodes per test: {args.episodes_per_test}")
    print("=" * 60)
    
    try:
        # Run the robustness tests
        df = run_robustness_tests(args)
        
        # Save results
        save_results(df, save_dir, args)
        
        # Generate plots
        if not args.no_plots:
            create_plots(df, save_dir)
        
        # Print summary
        print("\n" + "=" * 60)
        print("ROBUSTNESS TEST SUMMARY")
        print("=" * 60)
        print(f"Baseline (no noise) success rate: {df.loc[df['noise_std'] == 0.0, 'success_rate'].iloc[0]:.1f}%")
        print(f"Success rate at max noise (0.5): {df.loc[df['noise_std'] == 0.5, 'success_rate'].iloc[0]:.1f}%")
        
        max_degradation = df.loc[df['noise_std'] == 0.0, 'success_rate'].iloc[0] - df['success_rate'].min()
        print(f"Maximum performance degradation: {max_degradation:.1f}%")
        
        # Find noise tolerance (noise level where success rate >= 80%)
        tolerant_df = df[df['success_rate'] >= 80.0]
        if len(tolerant_df) > 0:
            noise_tolerance = tolerant_df['noise_std'].max()
            print(f"Noise tolerance (80% success): {noise_tolerance:.1f} std")
        else:
            print("Noise tolerance (80% success): < 0.1 std")
        
        print(f"\nResults saved to: {save_dir}")
        
    except Exception as e:
        print(f"Error during robustness test: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()