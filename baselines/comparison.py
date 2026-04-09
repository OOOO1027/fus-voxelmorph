"""
Comparison framework for registration methods.

Compares VoxelMorph with traditional methods using metrics from Zhong et al.:
- NCC (Normalized Cross-Correlation)
- MS-SSIM (Multi-Scale Structural Similarity)
- DSC (Dice Similarity Coefficient)
- Inference time
- Jacobian statistics (mean, folding percentage)
"""

import os
import time
import json
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import metrics
from utils.metrics import (
    compute_ncc, compute_mse, compute_ssim, compute_ms_ssim,
    compute_dsc, jacobian_determinant_2d
)


def evaluate_registration(source, target, warped, flow=None, threshold=0.5):
    """
    Evaluate registration result with comprehensive metrics.

    Parameters
    ----------
    source, target, warped : np.ndarray (H, W)
        Source, target, and warped images
    flow : np.ndarray (2, H, W), optional
        Displacement field
    threshold : float
        Threshold for DSC computation

    Returns
    -------
    metrics : dict
        Dictionary of evaluation metrics
    """
    metrics = {}

    # Image similarity metrics
    metrics['ncc_before'] = compute_ncc(source, target)
    metrics['ncc_after'] = compute_ncc(warped, target)
    metrics['mse_before'] = compute_mse(source, target)
    metrics['mse_after'] = compute_mse(warped, target)
    metrics['ssim_after'] = compute_ssim(warped, target)
    metrics['ms_ssim_after'] = compute_ms_ssim(warped, target)

    # Improvement
    metrics['ncc_improvement'] = metrics['ncc_after'] - metrics['ncc_before']
    metrics['mse_improvement'] = metrics['mse_before'] - metrics['mse_after']

    # DSC (binary mask overlap)
    metrics['dsc_after'] = compute_dsc(warped, target, threshold=threshold)

    # Deformation field analysis
    if flow is not None:
        flow_mag = np.sqrt(flow[0]**2 + flow[1]**2)
        metrics['flow_mean'] = flow_mag.mean()
        metrics['flow_max'] = flow_mag.max()
        metrics['flow_std'] = flow_mag.std()

        # Jacobian analysis
        _, jac_stats = jacobian_determinant_2d(flow)
        metrics['jac_mean'] = jac_stats['mean']
        metrics['jac_std'] = jac_stats['std']
        metrics['jac_min'] = jac_stats['min']
        metrics['jac_pct_neg'] = jac_stats['pct_neg']
    else:
        # No deformation field (e.g., for rigid/affine)
        metrics['flow_mean'] = 0.0
        metrics['flow_max'] = 0.0
        metrics['jac_mean'] = 1.0
        metrics['jac_pct_neg'] = 0.0

    return metrics


def compare_methods(source, target, methods_dict, device='cpu'):
    """
    Compare multiple registration methods on a single image pair.

    Parameters
    ----------
    source, target : np.ndarray (H, W)
        Source and target images
    methods_dict : dict
        Dictionary of {method_name: method_instance}
    device : str
        Device for deep learning methods

    Returns
    -------
    results : dict
        Dictionary of {method_name: {warped, flow, metrics, time}}
    """
    results = {}

    for method_name, method in methods_dict.items():
        print(f"  Running {method_name}...")

        try:
            # Check if it's a PyTorch model (VoxelMorph)
            if hasattr(method, 'forward') and callable(getattr(method, 'forward')):
                import torch
                method.eval()
                with torch.no_grad():
                    src_tensor = torch.from_numpy(source[np.newaxis, np.newaxis]).float().to(device)
                    tgt_tensor = torch.from_numpy(target[np.newaxis, np.newaxis]).float().to(device)

                    start_time = time.time()
                    warped, flow = method(src_tensor, tgt_tensor)
                    execution_time = time.time() - start_time

                    warped = warped[0, 0].cpu().numpy()
                    flow = flow[0].cpu().numpy()
            else:
                # Traditional method
                warped, transform_params = method.register(source, target)
                execution_time = transform_params.get('time', 0)

                # Get displacement field if available
                if hasattr(method, 'get_displacement_field'):
                    try:
                        flow = method.get_displacement_field(source.shape)
                    except:
                        flow = None
                else:
                    flow = None

            # Evaluate
            metrics = evaluate_registration(source, target, warped, flow)
            metrics['time'] = execution_time

            results[method_name] = {
                'warped': warped,
                'flow': flow,
                'metrics': metrics,
            }

        except Exception as e:
            print(f"    Failed: {e}")
            results[method_name] = {'error': str(e)}

    return results


def run_comparison_experiment(test_cases, methods_dict, output_dir='comparison_results',
                              device='cpu'):
    """
    Run comparison experiment on multiple test cases.

    Parameters
    ----------
    test_cases : list of dict
        List of {'name': str, 'source': array, 'target': array}
    methods_dict : dict
        Dictionary of methods
    output_dir : str
        Output directory
    device : str
        Device for deep learning methods

    Returns
    -------
    all_results : dict
        All results organized by test case and method
    summary_df : pd.DataFrame
        Summary statistics table
    """
    os.makedirs(output_dir, exist_ok=True)

    all_results = {}

    for case in test_cases:
        case_name = case['name']
        source = case['source']
        target = case['target']

        print(f"\nTest case: {case_name}")
        print(f"  Image shape: {source.shape}")

        # Run comparison
        results = compare_methods(source, target, methods_dict, device)
        all_results[case_name] = results

    # Save raw results
    with open(os.path.join(output_dir, 'raw_results.json'), 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for case_name, case_results in all_results.items():
            json_results[case_name] = {}
            for method_name, method_results in case_results.items():
                if 'metrics' in method_results:
                    json_results[case_name][method_name] = {
                        'metrics': method_results['metrics']
                    }
        json.dump(json_results, f, indent=2)

    # Create summary table
    summary_df = create_comparison_table(all_results)
    summary_df.to_csv(os.path.join(output_dir, 'comparison_table.csv'))
    summary_df.to_excel(os.path.join(output_dir, 'comparison_table.xlsx'))

    print(f"\nResults saved to {output_dir}/")

    return all_results, summary_df


def create_comparison_table(all_results):
    """
    Create comparison table from results.

    Parameters
    ----------
    all_results : dict
        Results from run_comparison_experiment

    Returns
    -------
    df : pd.DataFrame
        Comparison table
    """
    rows = []

    for case_name, case_results in all_results.items():
        for method_name, method_results in case_results.items():
            if 'metrics' not in method_results:
                continue

            metrics = method_results['metrics']

            row = {
                'Test Case': case_name,
                'Method': method_name,
                'NCC (after)': metrics.get('ncc_after', np.nan),
                'NCC Improvement': metrics.get('ncc_improvement', np.nan),
                'MSE (after)': metrics.get('mse_after', np.nan),
                'MSE Improvement': metrics.get('mse_improvement', np.nan),
                'MS-SSIM': metrics.get('ms_ssim_after', np.nan),
                'DSC': metrics.get('dsc_after', np.nan),
                'Flow Mean (px)': metrics.get('flow_mean', np.nan),
                'Flow Max (px)': metrics.get('flow_max', np.nan),
                'Jac Mean': metrics.get('jac_mean', np.nan),
                'Jac %Neg': metrics.get('jac_pct_neg', np.nan),
                'Time (s)': metrics.get('time', np.nan),
            }
            rows.append(row)

    df = pd.DataFrame(rows)

    # Compute mean across test cases
    if len(df) > 0:
        mean_row = df.groupby('Method').mean(numeric_only=True).reset_index()
        mean_row['Test Case'] = 'Average'
        df = pd.concat([df, mean_row], ignore_index=True)

    return df


def create_comparison_figures(all_results, output_dir='comparison_results'):
    """
    Create comparison figures from results.

    Parameters
    ----------
    all_results : dict
        Results from run_comparison_experiment
    output_dir : str
        Output directory
    """
    os.makedirs(output_dir, exist_ok=True)

    # Extract data for plotting
    methods = set()
    cases = list(all_results.keys())

    for case_results in all_results.values():
        methods.update(case_results.keys())

    methods = sorted(methods)

    # Prepare data
    metrics_to_plot = [
        ('ncc_after', 'NCC (after)', 'higher'),
        ('ncc_improvement', 'NCC Improvement', 'higher'),
        ('ms_ssim_after', 'MS-SSIM', 'higher'),
        ('dsc_after', 'DSC', 'higher'),
        ('jac_pct_neg', 'Jacobian Folding %', 'lower'),
        ('time', 'Inference Time (s)', 'lower'),
    ]

    metric_values = defaultdict(lambda: defaultdict(list))

    for case_name, case_results in all_results.items():
        for method_name, method_results in case_results.items():
            if 'metrics' not in method_results:
                continue

            metrics = method_results['metrics']
            for metric_key, _, _ in metrics_to_plot:
                metric_values[metric_key][method_name].append(
                    metrics.get(metric_key, np.nan)
                )

    # Create bar plots for each metric
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, (metric_key, metric_name, direction) in enumerate(metrics_to_plot):
        ax = axes[idx]

        data = []
        labels = []
        for method in methods:
            values = metric_values[metric_key][method]
            if values:
                data.append(np.nanmean(values))
                labels.append(method)

        if not data:
            continue

        colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))
        bars = ax.bar(labels, data, color=colors)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)

        ax.set_ylabel(metric_name)
        ax.set_title(metric_name)
        ax.tick_params(axis='x', rotation=45)

        # Add grid
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Figures saved to {output_dir}/")


def create_side_by_side_comparison(all_results, case_name, output_path):
    """
    Create side-by-side visualization of all methods for one test case.

    Parameters
    ----------
    all_results : dict
        Results dictionary
    case_name : str
        Name of test case to visualize
    output_path : str
        Output file path
    """
    case_results = all_results[case_name]

    # Get source and target from first successful method
    source, target = None, None
    for method_results in case_results.values():
        if 'warped' in method_results:
            # Load original images if needed
            # For now, we skip this visualization
            break

    n_methods = len([m for m in case_results.values() if 'warped' in m])

    fig, axes = plt.subplots(n_methods + 2, 4, figsize=(16, 4 * (n_methods + 2)))

    # Show source and target in first row
    # Note: This is a placeholder - you'd need to pass source/target to this function

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def print_comparison_table(df):
    """Pretty print comparison table."""
    print("\n" + "=" * 100)
    print("Registration Methods Comparison")
    print("=" * 100)

    # Format for display
    display_df = df.copy()
    numeric_cols = display_df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        if 'Time' in col:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}")
        else:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")

    print(display_df.to_string(index=False))
    print("=" * 100)
