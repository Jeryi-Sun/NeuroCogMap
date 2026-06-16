#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
提取并绘制不同模型的检测指标对比图
"""

import json
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Set Nature publication-style parameters
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300

# Define color scheme for Nature-style plotting
def get_method_color(method_name):
    """Get color for each method, highlighting important ones"""
    # Define color map for all methods
    color_map = {
        # Main methods - prominent colors
        'llm_detector': '#2E7D32',           # Dark green
        'llm_detector_simple': '#66BB6A',   # Light green
        'our_method': '#1976D2',             # Deep blue
        
        # Baseline methods - muted, low-saturation colors
        'selfcheckgpt': '#8D6E63',           # Muted brown
        'semantic_entropy': '#6C757D',      # Muted blue-gray
        'ppl': '#95A5A6',                   # Light gray-blue
        'entropy': '#A1887F',               # Muted tan
        'ln_entropy': '#A1887F',            # Muted tan
        'eigenscore': '#CFCFCF',            # Very light gray
    }
    
    # Return color from map, or default muted gray
    return color_map.get(method_name, '#BDBDBD')

def load_metrics_from_json(file_path):
    """Load metrics from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Failed to load file {file_path}: {e}")
        return None

def extract_all_metrics(base_dir):
    """Extract metrics from all models"""
    metrics_data = defaultdict(dict)
    
    # Iterate through all models in detection directory
    detection_dir = os.path.join(base_dir, 'results/detection')
    for model_dir in glob.glob(os.path.join(detection_dir, '*')):
        if os.path.isdir(model_dir) and os.path.basename(model_dir) != 'baselines':
            model_name = os.path.basename(model_dir)

            cv_metrics_file = os.path.join(model_dir, 'cv_metrics.json')
            
            if os.path.exists(cv_metrics_file):
                data = load_metrics_from_json(cv_metrics_file)
                if data:
                    metrics_data[model_name]['our_method'] = {
                        'accuracy': data.get('mean_accuracy'),
                        'precision': data.get('mean_precision'),
                        'recall': data.get('mean_recall'),
                        'f1': data.get('mean_f1'),
                        'auroc': data.get('mean_auroc'),
                        'auprc': data.get('mean_auprc')
                    }
    
    # Iterate through all models in baselines directory
    baselines_dir = os.path.join(base_dir, 'results/detection/baselines')
    for model_dir in glob.glob(os.path.join(baselines_dir, '*')):
        if os.path.isdir(model_dir):
            model_name = os.path.basename(model_dir)
            # List all available method files
            metric_files = glob.glob(os.path.join(model_dir, '*_cv_metrics.json'))
            
            for metric_file in metric_files:
                method_name = os.path.basename(metric_file).replace('_cv_metrics.json', '')
                data = load_metrics_from_json(metric_file)
                
                if data:
                    metrics_data[model_name][method_name] = {
                        'accuracy': data.get('mean_accuracy'),
                        'precision': data.get('mean_precision'),
                        'recall': data.get('mean_recall'),
                        'f1': data.get('mean_f1'),
                        'auroc': data.get('mean_auroc'),
                        'auprc': data.get('mean_auprc')
                    }
    
    # Iterate through all models in llm_detection directory
    llm_detection_dir = os.path.join(base_dir, 'results/llm_detection')
    for model_dir in glob.glob(os.path.join(llm_detection_dir, '*')):
        if os.path.isdir(model_dir):
            model_name = os.path.basename(model_dir)
            
            # Read cv_metrics.json (with neural activity)
            cv_metrics_file = os.path.join(model_dir, 'cv_metrics.json')
            if os.path.exists(cv_metrics_file):
                data = load_metrics_from_json(cv_metrics_file)
                if data:
                    if model_name not in metrics_data:
                        metrics_data[model_name] = {}
                    metrics_data[model_name]['llm_detector'] = {
                        'accuracy': data.get('mean_accuracy'),
                        'precision': data.get('mean_precision'),
                        'recall': data.get('mean_recall'),
                        'f1': data.get('mean_f1'),
                        'auroc': data.get('mean_auroc'),
                        'auprc': data.get('mean_auprc')
                    }
            
            # Read cv_metrics_simple.json (without neural activity)
            cv_metrics_simple_file = os.path.join(model_dir, 'cv_metrics_simple.json')
            if os.path.exists(cv_metrics_simple_file):
                data = load_metrics_from_json(cv_metrics_simple_file)
                if data:
                    if model_name not in metrics_data:
                        metrics_data[model_name] = {}
                    metrics_data[model_name]['llm_detector_simple'] = {
                        'accuracy': data.get('mean_accuracy'),
                        'precision': data.get('mean_precision'),
                        'recall': data.get('mean_recall'),
                        'f1': data.get('mean_f1'),
                        'auroc': data.get('mean_auroc'),
                        'auprc': data.get('mean_auprc')
                    }
    
    return metrics_data

def plot_comparison(metrics_data, output_dir, selected_models=None):
    """Plot comparison charts"""
    # Define metrics
    metrics = ['accuracy', 'f1', 'auroc', 'auprc']
    metric_labels = {
        'accuracy': 'Accuracy',
        'f1': 'F1 Score',
        'auroc': 'AUROC',
        'auprc': 'AUPRC'
    }
    
    # Get all model names
    if selected_models:
        # Filter to only selected models
        model_names = [name for name in sorted(metrics_data.keys()) if name in selected_models]
    else:
        model_names = sorted(metrics_data.keys())
    
    # Plot separate figure for each metric
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Prepare data
        data_to_plot = []
        method_names = []
        
        # Collect all method names
        all_methods = set()
        for model_name in model_names:
            all_methods.update(metrics_data[model_name].keys())
        all_methods = sorted(list(all_methods))
        
        # Prepare data for each model and method
        for method in all_methods:
            values = []
            for model_name in model_names:
                if method in metrics_data[model_name]:
                    value = metrics_data[model_name][method].get(metric)
                    if value is not None:
                        values.append(value)
                    else:
                        values.append(0)
                else:
                    values.append(0)
            data_to_plot.append(values)
            method_names.append(method)
        
        # Set bar chart parameters
        x = np.arange(len(model_names))
        width = 0.8 / len(all_methods)
        
        # Plot bar chart with Nature-style colors
        for i, (method, values) in enumerate(zip(method_names, data_to_plot)):
            offset = (i - len(all_methods)/2) * width + width/2
            color = get_method_color(method)
            ax.bar(x + offset, values, width, label=method, color=color, 
                   edgecolor='white', linewidth=0.5, alpha=0.9)
        
        # Set labels and title in Nature style
        ax.set_ylabel(metric_labels[metric], fontsize=10)
        ax.set_title(f'{metric_labels[metric]}', fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=0, ha='center', fontsize=9)
        ax.legend(loc='best', fontsize=8, frameon=False, ncol=2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='x', length=0)
        ax.set_ylim([0, 1])
        # Add subtle grid
        ax.grid(axis='y', alpha=0.2, linestyle='--', linewidth=0.5)
        
        plt.tight_layout()
        
        # Save figure
        output_file = os.path.join(output_dir, f'{metric}_comparison.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved figure: {output_file}")
        plt.close()
    
    # Plot comprehensive comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        # Prepare data
        data_to_plot = []
        method_names = []
        
        # Collect all method names
        all_methods = set()
        for model_name in model_names:
            all_methods.update(metrics_data[model_name].keys())
        all_methods = sorted(list(all_methods))
        
        # Prepare data for each model and method
        for method in all_methods:
            values = []
            for model_name in model_names:
                if method in metrics_data[model_name]:
                    value = metrics_data[model_name][method].get(metric)
                    if value is not None:
                        values.append(value)
                    else:
                        values.append(0)
                else:
                    values.append(0)
            data_to_plot.append(values)
            method_names.append(method)
        
        # Set bar chart parameters
        x = np.arange(len(model_names))
        width = 0.8 / len(all_methods)
        
        # Plot bar chart with Nature-style colors
        for i, (method, values) in enumerate(zip(method_names, data_to_plot)):
            offset = (i - len(all_methods)/2) * width + width/2
            color = get_method_color(method)
            ax.bar(x + offset, values, width, label=method, color=color,
                   edgecolor='white', linewidth=0.3, alpha=0.9)
        
        # Set labels and title in Nature style
        ax.set_ylabel(metric_labels[metric], fontsize=9)
        ax.set_title(metric_labels[metric], fontsize=10, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=0, ha='center', fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='x', length=0)
        ax.set_ylim([0, 1])
        # Add subtle grid
        ax.grid(axis='y', alpha=0.2, linestyle='--', linewidth=0.3)
        if idx == 0:
            ax.legend(loc='best', fontsize=7, frameon=False, ncol=2)
    
    plt.tight_layout()
    
    # Save comprehensive comparison
    output_file = os.path.join(output_dir, 'all_metrics_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved comprehensive comparison: {output_file}")
    plt.close()

def main():
    base_dir = '/path/to/project_root/safety_explanation/hallucination'
    output_dir = '/path/to/project_root/safety_explanation/hallucination/results/detection/graphs'
    
    # Selected datasets to plot
    selected_datasets = ['MedHallu_gemma-2-2b', 'truthfulqa_gemma-2-2b', 'nq_open_gemma-2-2b']
    
    # Extract metrics
    print("Extracting metrics...")
    metrics_data = extract_all_metrics(base_dir)
    
    # Print extracted data (filtered to selected datasets)
    print("\nExtracted metrics data (filtered):")
    for model_name in selected_datasets:
        if model_name in metrics_data:
            print(f"\n{model_name}:")
            methods = metrics_data[model_name]
            for method_name, metrics in methods.items():
                print(f"  {method_name}:")
                for metric_name, value in metrics.items():
                    if value is not None:
                        print(f"    {metric_name}: {value:.4f}")
    
    # Plot comparison
    print("\nPlotting comparison...")
    plot_comparison(metrics_data, output_dir, selected_models=selected_datasets)
    
    print("\nDone!")

if __name__ == '__main__':
    main()
