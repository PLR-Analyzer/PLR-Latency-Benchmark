"""
Evaluate latency detection methods across different sample rates.

Generates synthetic PLR datasets at various sample rates and evaluates
selected latency detection methods against ground truth latency.
Produces a plot showing estimation error vs sample rate.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path so we can import from latency_methods
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from data_generation.simulation import simulate_sample
from latency_methods import LatencyMethods


def evaluate_method_on_sample(
    t, D_obs, true_latency, method_name, stim_time, led_duration, fps
):
    """
    Evaluate a single latency method on one sample.

    Parameters
    ----------
    t : array
        Time points (seconds).
    D_obs : array
        Observed pupil diameter.
    true_latency : float
        Ground truth latency time (seconds).
    method_name : str
        Name of the method to evaluate.
    stim_time : float
        Time when stimulus starts (seconds).
    led_duration : float
        Duration of LED stimulus (seconds).
    fps : float
        Sampling rate (frames per second).

    Returns
    -------
    float or np.nan
        Estimated latency time (seconds), or np.nan if estimation failed.
    """
    try:
        latency_est, _ = LatencyMethods.compute_by_name(
            method_name, t, D_obs, stim_time, led_duration=led_duration, fps=fps
        )
        return latency_est
    except Exception as e:
        print(f"    Warning: Method '{method_name}' failed: {e}")
        return np.nan


def evaluate_methods_at_fps(
    fps,
    n_samples,
    method_names,
    noise_sd,
    D_min,
    D_max,
    duration=5000,
    stim_time=500,
    led_duration=200,
    verbose=True,
):
    """
    Generate samples at a given FPS and evaluate all methods.

    Parameters
    ----------
    fps : float
        Sampling rate in frames per second.
    n_samples : int
        Number of samples to generate.
    method_names : list of str
        Names of latency methods to evaluate.
    noise_sd : float
        Standard deviation of Gaussian noise.
    D_min : float
        Minimum pupil diameter for stimulus.
    D_max : float
        Maximum pupil diameter (baseline).
    duration : float
        Duration of each recording in milliseconds.
    stim_time : float
        Time when stimulus starts in milliseconds.
    led_duration : float
        Duration of LED stimulus in milliseconds.
    verbose : bool
        Print progress.

    Returns
    -------
    dict
        Results with keys mapping method names to lists of errors (in ms),
        plus "fps", "true_latencies", and "quantization_errors".
    """
    if verbose:
        print(f"\nGenerating {n_samples} samples at {fps} fps...")

    results = {method_name: [] for method_name in method_names}
    results["fps"] = fps
    results["true_latencies"] = []
    results["quantization_errors"] = []

    for i in range(n_samples):
        if verbose and (i + 1) % max(1, n_samples // 10) == 0:
            print(f"  Sample {i + 1}/{n_samples}...")

        # Generate sample
        time, D_obs, D_clean, true_latency, params = simulate_sample(
            duration=duration,
            fps=fps,
            stim_time=stim_time,  # convert to seconds for simulate_sample
            stim_duration=led_duration,
            D_min=D_min,
            D_max=D_max,
            seed=i,
            noise_sd=noise_sd,
            drift_amp=0.2,
        )

        results["true_latencies"].append(true_latency)

        # Calculate actual quantization error: distance to nearest sample point
        # Find the nearest sample point to true_latency (both in ms)
        idx_nearest = np.argmin(np.abs(time - true_latency))
        nearest_time = time[idx_nearest]
        quant_error = abs(nearest_time - true_latency)  # in ms
        results["quantization_errors"].append(quant_error)

        # Evaluate each method
        for method_name in method_names:
            latency_est = evaluate_method_on_sample(
                time,
                D_obs,
                true_latency,
                method_name,
                stim_time,
                led_duration,
                fps,
            )

            if not np.isnan(latency_est):
                # Compute error in milliseconds
                error_ms = abs(latency_est - true_latency)
                results[method_name].append(error_ms)
            # If nan, we skip adding to results for this method

    return results


def compute_error_stats(errors):
    """
    Compute MAE, RMSE, and median error from a list of errors.

    Parameters
    ----------
    errors : list
        List of estimation errors.

    Returns
    -------
    dict
        Dictionary with keys "mae", "rmse", "median", "std", "count".
    """
    if len(errors) == 0:
        return {
            "mae": np.nan,
            "rmse": np.nan,
            "median": np.nan,
            "std": np.nan,
            "count": 0,
        }

    errors = np.array(errors)
    return {
        "mae": float(np.mean(errors)),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "median": float(np.median(errors)),
        "std": float(np.std(errors)),
        "count": len(errors),
    }


def evaluate_across_fps(
    fps_list,
    n_samples,
    method_names,
    noise_sd,
    D_min,
    D_max,
    duration=5000,
    stim_time=500,
    led_duration=200,
    verbose=True,
):
    """
    Evaluate methods across multiple sample rates.

    Parameters
    ----------
    fps_list : list of float
        List of sample rates to evaluate.
    n_samples : int
        Number of samples per sample rate.
    method_names : list of str
        Names of latency methods to evaluate.
    noise_sd : float
        Standard deviation of Gaussian noise.
    D_min : float
        Minimum pupil diameter for stimulus.
    D_max : float
        Maximum pupil diameter (baseline).
    duration : float
        Duration of each recording in milliseconds.
    stim_time : float
        Time when stimulus starts in milliseconds.
    led_duration : float
        Duration of LED stimulus in milliseconds.
    verbose : bool
        Print progress.

    Returns
    -------
    dict
        Nested dictionary with structure:
        {
            fps1: {method1: errors_list, method2: errors_list, ...},
            fps2: {method1: errors_list, method2: errors_list, ...},
            ...
        }
    """
    all_results = {}

    for fps in fps_list:
        results = evaluate_methods_at_fps(
            fps,
            n_samples,
            method_names,
            noise_sd,
            D_min,
            D_max,
            duration,
            stim_time,
            led_duration,
            verbose,
        )
        all_results[fps] = results

    return all_results


def plot_results(all_results, method_names, noise_sd, D_min, D_max, output_path=None):
    """
    Plot error metrics across sample rates for all methods.

    Parameters
    ----------
    all_results : dict
        Nested results from evaluate_across_fps.
    method_names : list of str
        Names of methods being evaluated.
    noise_sd : float
        Noise standard deviation (for title).
    D_min : float
        Minimum diameter (for title).
    D_max : float
        Maximum diameter (for title).
    output_path : str or Path, optional
        Path to save the figure.
    """
    fps_list = sorted(all_results.keys())

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    axes = axes.flatten()  # Flatten to 1D for easier indexing

    # Prepare data for each method
    method_data = {
        method: {"mae": [], "rmse": [], "median": []} for method in method_names
    }

    # Collect quantization errors for the 4th subplot
    quant_mae_per_fps = []
    quant_median_per_fps = []

    for fps in fps_list:
        for method in method_names:
            errors = all_results[fps].get(method, [])
            stats = compute_error_stats(errors)
            method_data[method]["mae"].append(stats["mae"])
            method_data[method]["rmse"].append(stats["rmse"])
            method_data[method]["median"].append(stats["median"])

        # Compute quantization error stats for this fps
        quant_errors = all_results[fps].get("quantization_errors", [])
        quant_stats = compute_error_stats(quant_errors)
        quant_mae_per_fps.append(quant_stats["mae"])
        quant_median_per_fps.append(quant_stats["median"])

    # Plot each metric
    colors = plt.cm.tab10(np.linspace(0, 1, len(method_names)))

    for idx, metric_name in enumerate(["mae", "rmse"]):
        ax = axes[idx]
        for method_idx, method in enumerate(method_names):
            values = method_data[method][metric_name]
            ax.plot(
                fps_list,
                values,
                marker="o",
                label=method,
                color=colors[method_idx],
                linewidth=2,
            )

        ax.plot(
            fps_list,
            quant_median_per_fps,
            marker="^",
            label="Median Quantization Error",
            color="darkred",
            linewidth=2.5,
            markersize=8,
            linestyle="--",
        )

        ax.set_xlabel("Sample Rate (fps)", fontsize=12)
        ax.set_ylabel(f"{metric_name.upper()} Error (ms)", fontsize=12)
        ax.set_yscale("log", base=10)
        ax.set_title(f"{metric_name.upper()} vs Sample Rate", fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

    # Add overall title with parameters
    fig.suptitle(
        f"Latency Estimation Error Across Sample Rates\n"
        f"(noise_sd={noise_sd}, D_min={D_min}, D_max={D_max})",
        fontsize=14,
        y=0.995,
    )
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"\nFigure saved to: {output_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate latency detection methods across sample rates."
    )
    parser.add_argument(
        "-m",
        "--methods",
        nargs="+",
        default=[
            "Min derivative",
            "Min derivative (smoothed)",
            "Velocity 2nd-order deviation",
        ],
        help="Latency methods to evaluate (default: Min derivative, Min derivative (smoothed), Velocity 2nd-order deviation)",
    )
    parser.add_argument(
        "-f",
        "--fps",
        nargs="+",
        type=float,
        default=[25, 30, 90, 200],
        help="Sample rates to evaluate (default: 25 30 90 200)",
    )
    parser.add_argument(
        "-n",
        "--num-samples",
        type=int,
        default=500,
        help="Number of samples per sample rate (default: 500)",
    )
    parser.add_argument(
        "--noise-sd",
        type=float,
        default=0.03,
        help="Standard deviation of Gaussian noise (default: 0.03)",
    )
    parser.add_argument(
        "--D-min",
        type=float,
        default=4.0,
        help="Minimum pupil diameter during stimulus (default: 4.0)",
    )
    parser.add_argument(
        "--D-max",
        type=float,
        default=5.0,
        help="Maximum pupil diameter at baseline (default: 5.0)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output path for the plot (optional; if not provided, plot is only shown)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_false",
        help="Print detailed progress information",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Latency Method Evaluation Across Sample Rates")
    print("=" * 70)
    print(f"Methods: {', '.join(args.methods)}")
    print(f"Sample rates (fps): {args.fps}")
    print(f"Samples per rate: {args.num_samples}")
    print(f"Noise SD: {args.noise_sd}, D_min: {args.D_min}, D_max: {args.D_max}")
    print("=" * 70)

    # Evaluate across all sample rates
    all_results = evaluate_across_fps(
        args.fps,
        args.num_samples,
        args.methods,
        args.noise_sd,
        args.D_min,
        args.D_max,
        verbose=args.verbose or True,
    )

    # Print summary
    print("\n" + "=" * 70)
    print("Summary of Results")
    print("=" * 70)
    for fps in sorted(all_results.keys()):
        print(f"\nFPS: {fps}")
        for method in args.methods:
            errors = all_results[fps].get(method, [])
            stats = compute_error_stats(errors)
            print(
                f"  {method:40s}: MAE={stats['mae']:7.2f}ms, "
                f"RMSE={stats['rmse']:7.2f}ms, Median={stats['median']:7.2f}ms "
                f"({stats['count']} samples)"
            )

    # Generate plot
    output_file = args.output
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"latency_evaluation_{timestamp}.png"

    plot_results(
        all_results, args.methods, args.noise_sd, args.D_min, args.D_max, output_file
    )


if __name__ == "__main__":
    main()
