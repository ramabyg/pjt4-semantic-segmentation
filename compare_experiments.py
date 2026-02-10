#!/usr/bin/env python
"""
Utility script for comparing multiple experiments.

Usage:
    python compare_experiments.py --output comparison_report.html
    python compare_experiments.py --experiments baseline large_model fast_train
"""

import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List
import pandas as pd
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_experiment_metrics(experiment_dir: Path) -> Dict:
    """
    Load metrics from an experiment directory

    Args:
        experiment_dir: Path to experiment directory

    Returns:
        Dictionary with metrics
    """
    metrics_dir = experiment_dir / "metrics"

    if not metrics_dir.exists():
        logger.warning(f"Metrics directory not found: {metrics_dir}")
        return {}

    json_files = list(metrics_dir.glob("metrics_*.json"))
    if not json_files:
        logger.warning(f"No metrics files found in: {metrics_dir}")
        return {}

    # Load the first (or most recent) metrics file
    metrics_file = sorted(json_files)[-1]  # Gets last one (most recent)

    with open(metrics_file) as f:
        metrics = json.load(f)

    return metrics


def extract_summary_metrics(metrics: List[Dict]) -> Dict:
    """
    Extract summary statistics from raw metrics

    Args:
        metrics: List of metric dictionaries

    Returns:
        Dictionary with summary statistics
    """
    if not metrics:
        return {}

    # Filter validation metrics
    val_metrics = [m for m in metrics if m.get('stage') == 'val']

    if not val_metrics:
        return {}

    summary = {
        'num_iterations': len(val_metrics),
    }

    # Get best/worst/avg for common metrics
    for metric_name in ['loss', 'iou', 'dice']:
        values = [m.get(metric_name) for m in val_metrics if metric_name in m]

        if values:
            summary[f'{metric_name}_best'] = max(values) if metric_name != 'loss' else min(values)
            summary[f'{metric_name}_worst'] = min(values) if metric_name != 'loss' else max(values)
            summary[f'{metric_name}_avg'] = sum(values) / len(values)
            summary[f'{metric_name}_latest'] = values[-1]

    return summary


def compare_experiments(experiment_names: List[str], output_dir: str = "./outputs") -> pd.DataFrame:
    """
    Compare multiple experiments

    Args:
        experiment_names: List of experiment names
        output_dir: Base output directory

    Returns:
        DataFrame with comparison results
    """
    results = {}

    for exp_name in experiment_names:
        exp_dir = Path(output_dir) / exp_name

        if not exp_dir.exists():
            logger.warning(f"Experiment directory not found: {exp_dir}")
            continue

        logger.info(f"Loading experiment: {exp_name}")
        metrics = load_experiment_metrics(exp_dir)
        summary = extract_summary_metrics(metrics)

        results[exp_name] = summary

    return pd.DataFrame(results).T


def generate_comparison_report(comparison_df: pd.DataFrame, output_file: str = None):
    """
    Generate comparison report

    Args:
        comparison_df: Comparison dataframe
        output_file: Output file path (HTML or CSV)
    """
    if comparison_df.empty:
        logger.warning("No data to compare")
        return

    print("\n" + "="*80)
    print("EXPERIMENT COMPARISON REPORT")
    print("="*80 + "\n")

    print(comparison_df.to_string())
    print()

    # Find best models
    print("\n" + "="*80)
    print("BEST MODELS BY METRIC")
    print("="*80 + "\n")

    for col in comparison_df.columns:
        if '_best' in col or '_avg' in col:
            metric_name = col.replace('_best', '').replace('_avg', '')
            if 'loss' in col:
                best_idx = comparison_df[col].idxmin()
            else:
                best_idx = comparison_df[col].idxmax()

            best_value = comparison_df.loc[best_idx, col]
            print(f"{col:20s}: {best_idx:30s} ({best_value:.6f})")

    # Save to file if specified
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_file.endswith('.csv'):
            comparison_df.to_csv(output_file)
            logger.info(f"Saved comparison to CSV: {output_file}")
        elif output_file.endswith('.html'):
            html = comparison_df.to_html()
            with open(output_file, 'w') as f:
                f.write(html)
            logger.info(f"Saved comparison to HTML: {output_file}")
        else:
            # Save as both
            comparison_df.to_csv(output_file + '.csv')
            comparison_df.to_html(output_file + '.html')
            logger.info(f"Saved comparison to {output_file}.*")


def plot_comparison(comparison_df: pd.DataFrame, metric: str = 'iou_best', output_file: str = None):
    """
    Plot comparison of experiments

    Args:
        comparison_df: Comparison dataframe
        metric: Metric to plot
        output_file: Output file for plot
    """
    if metric not in comparison_df.columns:
        logger.warning(f"Metric not found: {metric}")
        return

    plt.figure(figsize=(10, 6))
    comparison_df[metric].plot(kind='bar')
    plt.title(f"Experiment Comparison: {metric}")
    plt.ylabel(metric)
    plt.xlabel("Experiment")
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=100, bbox_inches='tight')
        logger.info(f"Saved plot to: {output_file}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Compare multiple experiments")
    parser.add_argument(
        "--experiments",
        nargs='+',
        help="List of experiment names to compare",
    )
    parser.add_argument(
        "--output-dir",
        default="./outputs",
        help="Base output directory (default: ./outputs)",
    )
    parser.add_argument(
        "--output",
        help="Output file for report (CSV or HTML)",
    )
    parser.add_argument(
        "--plot-metric",
        default="iou_best",
        help="Metric to plot (default: iou_best)",
    )

    args = parser.parse_args()

    # If no experiments specified, find all in output directory
    if not args.experiments:
        output_path = Path(args.output_dir)
        experiments = [d.name for d in output_path.iterdir() if d.is_dir()]
        logger.info(f"Found experiments: {experiments}")
    else:
        experiments = args.experiments

    if not experiments:
        logger.error(f"No experiments found in: {args.output_dir}")
        return

    # Compare experiments
    comparison_df = compare_experiments(experiments, args.output_dir)

    # Generate report
    generate_comparison_report(comparison_df, args.output)

    # Plot comparison
    plot_comparison(comparison_df, args.plot_metric)


if __name__ == "__main__":
    main()
