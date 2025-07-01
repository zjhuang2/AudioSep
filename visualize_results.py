#!/usr/bin/env python3
"""
Visualization script for noise sensitivity evaluation results
Creates plots and analysis of AudioSep's performance on triggering sounds
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import argparse
from typing import Dict, List


def load_evaluation_results(results_file: str) -> List[Dict]:
    """Load evaluation results from JSON file."""
    with open(results_file, 'r') as f:
        return json.load(f)


def create_performance_heatmap(results: List[Dict], output_dir: Path):
    """Create a heatmap showing performance across sounds and contexts."""
    # Extract metrics into a matrix
    sound_names = []
    contexts = ['quiet', 'single_noise', 'multiple_noises']
    
    # Metrics to visualize
    metrics = {
        'SDR Improvement': [],
        'Energy Reduction': [],
        'Target Residual': []
    }
    
    for sound_result in results:
        sound_name = sound_result['sound']
        sound_names.append(sound_name)
        
        for metric_name in metrics.keys():
            metric_row = []
            
            for context in contexts:
                if context in sound_result['contexts']:
                    # Average across all evaluations for this context
                    values = []
                    for result_group in sound_result['contexts'][context]:
                        for eval in result_group['evaluations']:
                            if eval['success']:
                                if metric_name == 'SDR Improvement':
                                    values.append(eval['separation_metrics']['sdr_improvement'])
                                elif metric_name == 'Energy Reduction':
                                    values.append(eval['elimination_metrics']['energy_reduction_ratio'])
                                elif metric_name == 'Target Residual':
                                    values.append(eval['elimination_metrics']['target_residual_correlation'])
                    
                    metric_row.append(np.mean(values) if values else 0)
                else:
                    metric_row.append(0)
            
            metrics[metric_name].append(metric_row)
    
    # Create heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    
    for idx, (metric_name, metric_data) in enumerate(metrics.items()):
        ax = axes[idx]
        
        # Create DataFrame for heatmap
        df = pd.DataFrame(metric_data, index=sound_names, columns=contexts)
        
        # Choose colormap based on metric
        if metric_name == 'Target Residual':
            cmap = 'RdYlGn_r'  # Lower is better
        else:
            cmap = 'RdYlGn'    # Higher is better
        
        sns.heatmap(df, annot=True, fmt='.2f', cmap=cmap, ax=ax, 
                   cbar_kws={'label': metric_name})
        ax.set_title(f'{metric_name} by Sound and Context')
        ax.set_xlabel('Audio Context')
        ax.set_ylabel('Triggering Sound')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_snr_analysis(results: List[Dict], output_dir: Path):
    """Analyze performance vs SNR for noisy contexts."""
    snr_data = []
    
    for sound_result in results:
        sound_name = sound_result['sound']
        
        for context_name, context_results in sound_result['contexts'].items():
            if context_name != 'quiet':
                for result_group in context_results:
                    for eval in result_group['evaluations']:
                        if eval['success'] and eval['snr_db'] is not None:
                            snr_data.append({
                                'sound': sound_name,
                                'context': context_name,
                                'snr_db': eval['snr_db'],
                                'sdr_improvement': eval['separation_metrics']['sdr_improvement'],
                                'energy_reduction': eval['elimination_metrics']['energy_reduction_ratio']
                            })
    
    df = pd.DataFrame(snr_data)
    
    # Plot SDR improvement vs SNR
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Group by SNR and calculate mean/std
    snr_grouped = df.groupby('snr_db').agg({
        'sdr_improvement': ['mean', 'std'],
        'energy_reduction': ['mean', 'std']
    }).reset_index()
    
    # Plot SDR improvement
    ax1.errorbar(snr_grouped['snr_db'], 
                snr_grouped['sdr_improvement']['mean'],
                yerr=snr_grouped['sdr_improvement']['std'],
                marker='o', capsize=5, capthick=2)
    ax1.set_xlabel('Input SNR (dB)')
    ax1.set_ylabel('SDR Improvement (dB)')
    ax1.set_title('Separation Performance vs Input SNR')
    ax1.grid(True, alpha=0.3)
    
    # Plot energy reduction
    ax2.errorbar(snr_grouped['snr_db'],
                snr_grouped['energy_reduction']['mean'],
                yerr=snr_grouped['energy_reduction']['std'],
                marker='s', capsize=5, capthick=2, color='orange')
    ax2.set_xlabel('Input SNR (dB)')
    ax2.set_ylabel('Energy Reduction Ratio')
    ax2.set_title('Elimination Performance vs Input SNR')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'snr_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_sound_category_analysis(results: List[Dict], output_dir: Path):
    """Analyze performance by sound category."""
    # Define sound categories
    categories = {
        'Eating/Mouth': ['chewing', 'slurping', 'lip_smacking', 'swallowing'],
        'Breathing/Nasal': ['sniffing', 'heavy_breathing', 'snoring', 'throat_clearing'],
        'Repetitive': ['pen_clicking', 'keyboard_typing', 'finger_tapping', 'foot_tapping'],
        'Environmental': ['clock_ticking', 'water_dripping', 'dog_barking', 'baby_crying'],
        'Electronic': ['phone_notification', 'buzzing'],
        'Mechanical': ['vacuum_cleaner', 'lawn_mower']
    }
    
    category_performance = []
    
    for category, sounds in categories.items():
        for sound_result in results:
            if sound_result['sound'] in sounds:
                # Calculate average performance across all contexts
                for context_name, context_results in sound_result['contexts'].items():
                    for result_group in context_results:
                        for eval in result_group['evaluations']:
                            if eval['success']:
                                category_performance.append({
                                    'category': category,
                                    'sound': sound_result['sound'],
                                    'context': context_name,
                                    'sdr_improvement': eval['separation_metrics']['sdr_improvement'],
                                    'energy_reduction': eval['elimination_metrics']['energy_reduction_ratio']
                                })
    
    df = pd.DataFrame(category_performance)
    
    # Create box plots by category
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # SDR Improvement by category
    df.boxplot(column='sdr_improvement', by='category', ax=ax1)
    ax1.set_ylabel('SDR Improvement (dB)')
    ax1.set_title('Separation Performance by Sound Category')
    ax1.set_xlabel('Sound Category')
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Energy Reduction by category
    df.boxplot(column='energy_reduction', by='category', ax=ax2)
    ax2.set_ylabel('Energy Reduction Ratio')
    ax2.set_title('Elimination Performance by Sound Category')
    ax2.set_xlabel('Sound Category')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'category_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_success_rate_analysis(results: List[Dict], output_dir: Path):
    """Analyze success rates for different sounds and queries."""
    success_data = []
    
    for sound_result in results:
        sound_name = sound_result['sound']
        total_evals = 0
        successful_evals = 0
        
        for context_name, context_results in sound_result['contexts'].items():
            for result_group in context_results:
                for eval in result_group['evaluations']:
                    total_evals += 1
                    if eval['success']:
                        successful_evals += 1
                    
                    success_data.append({
                        'sound': sound_name,
                        'query': eval['query'],
                        'context': context_name,
                        'success': eval['success']
                    })
        
        if total_evals > 0:
            success_rate = successful_evals / total_evals
            print(f"{sound_name}: {success_rate:.1%} success rate ({successful_evals}/{total_evals})")
    
    # Visualize success rates
    df = pd.DataFrame(success_data)
    
    # Success rate by sound
    sound_success = df.groupby('sound')['success'].mean().sort_values(ascending=False)
    
    plt.figure(figsize=(10, 8))
    sound_success.plot(kind='barh')
    plt.xlabel('Success Rate')
    plt.title('AudioSep Success Rate by Triggering Sound')
    plt.xlim(0, 1)
    
    # Add percentage labels
    for i, v in enumerate(sound_success):
        plt.text(v + 0.01, i, f'{v:.1%}', va='center')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'success_rates.png', dpi=300, bbox_inches='tight')
    plt.close()


def generate_summary_report(results: List[Dict], output_dir: Path):
    """Generate a comprehensive text summary report."""
    report_lines = []
    report_lines.append("AUDIOSEP NOISE SENSITIVITY FEASIBILITY STUDY")
    report_lines.append("=" * 50)
    report_lines.append("")
    
    # Overall statistics
    total_sounds = len(results)
    total_evaluations = sum(
        len(eval)
        for sound in results
        for context_results in sound['contexts'].values()
        for result_group in context_results
        for eval in result_group['evaluations']
    )
    
    successful_evaluations = sum(
        1 for sound in results
        for context_results in sound['contexts'].values()
        for result_group in context_results
        for eval in result_group['evaluations']
        if eval['success']
    )
    
    report_lines.append(f"Total triggering sounds tested: {total_sounds}")
    report_lines.append(f"Total evaluations performed: {total_evaluations}")
    report_lines.append(f"Overall success rate: {successful_evaluations/total_evaluations:.1%}")
    report_lines.append("")
    
    # Best and worst performing sounds
    sound_performance = {}
    for sound_result in results:
        sound_name = sound_result['sound']
        sdr_values = []
        energy_values = []
        
        for context_results in sound_result['contexts'].values():
            for result_group in context_results:
                for eval in result_group['evaluations']:
                    if eval['success']:
                        sdr_values.append(eval['separation_metrics']['sdr_improvement'])
                        energy_values.append(eval['elimination_metrics']['energy_reduction_ratio'])
        
        if sdr_values:
            sound_performance[sound_name] = {
                'avg_sdr': np.mean(sdr_values),
                'avg_energy_reduction': np.mean(energy_values)
            }
    
    # Sort by SDR improvement
    sorted_sounds = sorted(sound_performance.items(), 
                         key=lambda x: x[1]['avg_sdr'], 
                         reverse=True)
    
    report_lines.append("TOP 5 BEST PERFORMING SOUNDS (by SDR improvement):")
    for sound, metrics in sorted_sounds[:5]:
        report_lines.append(f"  - {sound}: {metrics['avg_sdr']:.2f} dB SDR improvement, "
                          f"{metrics['avg_energy_reduction']:.1%} energy reduction")
    
    report_lines.append("")
    report_lines.append("TOP 5 CHALLENGING SOUNDS (by SDR improvement):")
    for sound, metrics in sorted_sounds[-5:]:
        report_lines.append(f"  - {sound}: {metrics['avg_sdr']:.2f} dB SDR improvement, "
                          f"{metrics['avg_energy_reduction']:.1%} energy reduction")
    
    # Save report
    with open(output_dir / 'summary_report.txt', 'w') as f:
        f.write('\n'.join(report_lines))
    
    print('\n'.join(report_lines))


def main():
    parser = argparse.ArgumentParser(description="Visualize noise sensitivity evaluation results")
    parser.add_argument('--results-file', type=str, required=True,
                      help='Path to evaluation results JSON file')
    parser.add_argument('--output-dir', type=str, default='visualizations',
                      help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    print("Loading evaluation results...")
    results = load_evaluation_results(args.results_file)
    
    # Generate visualizations
    print("Creating performance heatmap...")
    create_performance_heatmap(results, output_dir)
    
    print("Creating SNR analysis...")
    create_snr_analysis(results, output_dir)
    
    print("Creating sound category analysis...")
    create_sound_category_analysis(results, output_dir)
    
    print("Creating success rate analysis...")
    create_success_rate_analysis(results, output_dir)
    
    print("Generating summary report...")
    generate_summary_report(results, output_dir)
    
    print(f"\nVisualization complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()