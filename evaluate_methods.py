"""
Evaluate latency detection methods across different sample rates or noise levels.

Generates synthetic PLR datasets at various sample rates OR noise levels and evaluates
selected latency detection methods against ground truth latency.
Produces a plot showing estimation error vs the varied parameter.
"""

import argparse
import multiprocessing
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pickle
from pathlib import Path

from data_generation.simulation import simulate_sample
from latency_methods import LatencyMethods


def _process_sample(
    i, duration, fps, stim_time, led_duration, D_min, D_max, noise_sd, method_names
):
    # Generate sample
    time, D_obs, D_clean, true_latency, params = simulate_sample(
        duration=duration,
        fps=fps,
        stim_time=stim_time,
        stim_duration=led_duration,
        D_min=D_min,
        D_max=D_max,
        seed=i,
        noise_sd=noise_sd,
        drift_amp=0.2,
    )

    # Calculate actual quantization error: distance to nearest sample point
    idx_nearest = np.argmin(np.abs(time - true_latency))
    nearest_time = time[idx_nearest]
    quant_error = abs(nearest_time - true_latency)

    # Evaluate each method
    errors = {}
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
            error_ms = abs(latency_est - true_latency)
            errors[method_name] = error_ms
        else:
            errors[method_name] = None

    return true_latency, quant_error, errors


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


def evaluate_methods_at_params(
    fps,
    noise_sd,
    n_samples,
    method_names,
    D_min,
    D_max,
    duration=5000,
    stim_time=500,
    led_duration=200,
    verbose=True,
):
    """
    Generate samples at given FPS and noise level and evaluate all methods.

    Parameters
    ----------
    fps : float
        Sampling rate in frames per second.
    noise_sd : float
        Standard deviation of Gaussian noise.
    n_samples : int
        Number of samples to generate.
    method_names : list of str
        Names of latency methods to evaluate.
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
        plus "fps", "noise_sd", "true_latencies", and "quantization_errors".
    """
    if verbose:
        print(
            f"\nGenerating {n_samples} samples at {fps} fps, noise_sd={noise_sd}, diameters={D_min, D_max}..."
        )

    results = {method_name: [] for method_name in method_names}
    results["fps"] = fps
    results["noise_sd"] = noise_sd
    results["true_latencies"] = []
    results["quantization_errors"] = []

    args = (
        duration,
        fps,
        stim_time,
        led_duration,
        D_min,
        D_max,
        noise_sd,
        method_names,
    )
    with multiprocessing.Pool() as pool:
        sample_results = pool.starmap(
            _process_sample, [(i,) + args for i in range(n_samples)]
        )

    for true_lat, quant_err, errs in sample_results:
        results["true_latencies"].append(true_lat)
        results["quantization_errors"].append(quant_err)
        for method_name in method_names:
            if errs[method_name] is not None:
                results[method_name].append(errs[method_name])

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


def save_results(path, all_results, metadata=None):
    """Save evaluation results and metadata to a pickle file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"all_results": all_results, "metadata": metadata or {}}
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"Results saved to: {path}")


def load_results(path):
    """Load evaluation results and metadata from a pickle file."""
    path = Path(path)
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data.get("all_results"), data.get("metadata", {})


def evaluate_across_parameters(
    param_list,
    param_type,
    n_samples,
    method_names,
    fixed_fps,
    fixed_noise,
    D_min,
    D_max,
    duration=5000,
    stim_time=500,
    led_duration=200,
    verbose=True,
):
    """
    Evaluate methods across multiple parameter values (fps or noise).

    Parameters
    ----------
    param_list : list of float
        List of parameter values to evaluate.
    param_type : str
        Either "fps" or "noise" to indicate which parameter varies.
    n_samples : int
        Number of samples per parameter value.
    method_names : list of str
        Names of latency methods to evaluate.
    fixed_fps : float
        Fixed FPS value (used when param_type="noise").
    fixed_noise : float
        Fixed noise SD value (used when param_type="fps").
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
        Nested dictionary with parameter values as keys.
    """
    all_results = {}

    if param_type == "fps":
        eval_values = fixed_noise
    else:
        eval_values = fixed_fps

    for diameters in zip(D_min, D_max):
        # Iterate over the diameter combinations (D_Min, D_Max).
        # Different diameter combinations will be stacked on top of each other
        # in the resulting plot
        diameter_results = {}
        for eval_val in eval_values:
            # Iterate over the fixed noise or fps values.
            # Different fixed values will be stacked next to each other in the
            # resulting plot
            eval_results = {}
            for param_value in param_list:
                # This iterates over the parameter, that should be plotted on
                # the x-axis (fps or noise). This is given as an argument with
                # (MIN, MAX, STEP).
                if param_type == "fps":
                    fps = param_value
                    noise_sd = eval_val
                else:  # param_type == "noise"
                    fps = eval_val
                    noise_sd = param_value

                results = evaluate_methods_at_params(
                    fps,
                    noise_sd,
                    n_samples,
                    method_names,
                    diameters[0],  # D_MIN
                    diameters[1],  # D_MAX
                    duration,
                    stim_time,
                    led_duration,
                    verbose,
                )
                eval_results[param_value] = results
            diameter_results[eval_val] = eval_results
        all_results[diameters] = diameter_results

    return all_results


def plot_results(all_results, method_names, param_type, D_min, D_max, output_path=None):
    """
    Plot error metrics across parameter values for all methods.

    Parameters
    ----------
    all_results : dict
        Nested results from evaluate_across_parameters.
    method_names : list of str
        Names of methods being evaluated.
    param_type : str
        Either "fps" or "noise".
    D_min : float
        Minimum diameter (for title).
    D_max : float
        Maximum diameter (for title).
    output_path : str or Path, optional
        Path to save the figure.
    """

    # The dictonary containing all results (all_results) has the following structure:
    #   {
    #       diameter-tuple: {
    #           fixed-value_n: {
    #               x-axis-value_n:
    #                   latency-method_n: {
    #                       mse-values
    #                   }
    #               }
    #           }
    #           ...
    #       }
    # }
    n_cols = len(next(iter(all_results.values())))
    n_rows = len(all_results.keys())
    fig = plt.figure(
        constrained_layout=True,
        figsize=(8 * n_cols, 5 * n_rows),
    )

    # Set x-axis label based on parameter type
    if param_type == "fps":
        xlabel = "Sample Rate (Hz)"
        param_info = f"noise_sd="
    else:
        xlabel = "Noise SD"
        param_info = f"fps="
    fig.suptitle(
        f"Latency Estimation Error Across {xlabel}\n",
        fontsize=14,
        weight="bold",
    )

    # create nx1 subfigs for rows
    subfigs = fig.subfigures(nrows=n_rows, ncols=1)

    # Plot each metric
    colors = plt.cm.tab10(np.linspace(0, 1, len(method_names)))

    for idx_vert, (diameter_tuple, subfig) in enumerate(
        zip(all_results.keys(), subfigs)
    ):
        subfig.suptitle(
            f"(D_min={diameter_tuple[0]}, D_max={diameter_tuple[1]})",
            fontsize=14,
        )

        axs = subfig.subplots(nrows=1, ncols=n_cols)
        for idx_hori, (eval_value, ax) in enumerate(
            zip(sorted(all_results[diameter_tuple].keys()), axs)
        ):
            # Collect quantization errors
            quant_mae_per_param = []
            quant_median_per_param = []

            # Prepare data for each method
            method_data = {
                method: {"mae": [], "rmse": [], "median": []} for method in method_names
            }

            param_list = sorted(all_results[diameter_tuple][eval_value].keys())

            for param_value in param_list:
                for method in method_names:
                    errors = all_results[diameter_tuple][eval_value][param_value].get(
                        method, []
                    )
                    stats = compute_error_stats(errors)
                    method_data[method]["mae"].append(stats["mae"])
                    method_data[method]["rmse"].append(stats["rmse"])
                    method_data[method]["median"].append(stats["median"])

                # Compute quantization error stats
                quant_errors = all_results[diameter_tuple][eval_value][param_value].get(
                    "quantization_errors", []
                )
                quant_stats = compute_error_stats(quant_errors)
                quant_mae_per_param.append(quant_stats["mae"])
                quant_median_per_param.append(quant_stats["median"])

            for method_idx, method in enumerate(method_names):
                values = method_data[method]["mae"]
                ax.plot(
                    param_list,
                    values,
                    marker="o",
                    label=method,
                    color=colors[method_idx],
                    linewidth=2,
                )

            ax.plot(
                param_list,
                quant_mae_per_param,
                marker="^",
                label="Mean Quantization Error",
                color="darkred",
                linewidth=2.5,
                markersize=8,
                linestyle="--",
            )

            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel(f"MAE Error (ms)", fontsize=12)
            ax.set_yscale("log", base=10)
            ax.set_title(
                f"MAE over {xlabel} ({param_info+str(eval_value)})", fontsize=13
            )
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"\nFigure saved to: {output_path}")

    # plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate latency detection methods across sample rates or noise levels.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # Vary FPS from 25 to 200 with step 25, fixed noise 0.03
            python script.py --fps-range 25 200 25 --noise 0.03

            # Vary noise from 0.01 to 0.1 with step 0.01, fixed FPS 90
            python script.py --noise-range 0.01 0.1 0.01 --fps 90
        """,
    )
    parser.add_argument(
        "-m",
        "--methods",
        nargs="+",
        default=[
            "Threshold crossing",
            "Min derivative",
            "Piecewise-linear fit",
            "Exponential fit",
            "Bergamin & Kardon",
        ],
        help="Latency methods to evaluate",
    )

    # FPS parameters
    fps_group = parser.add_mutually_exclusive_group(required=True)
    fps_group.add_argument(
        "--fps",
        nargs="+",
        type=float,
        help="Fixed FPS value (use with --noise-range)",
    )
    fps_group.add_argument(
        "--fps-range",
        nargs=3,
        type=float,
        metavar=("MIN", "MAX", "STEP"),
        help="FPS range: min max step (use with --noise)",
    )

    # Noise parameters
    noise_group = parser.add_mutually_exclusive_group(required=True)
    noise_group.add_argument(
        "--noise",
        nargs="+",
        type=float,
        help="Fixed noise SD value (use with --fps-range)",
    )
    noise_group.add_argument(
        "--noise-range",
        nargs=3,
        type=float,
        metavar=("MIN", "MAX", "STEP"),
        help="Noise SD range: min max step (use with --fps)",
    )

    parser.add_argument(
        "-n",
        "--num-samples",
        type=int,
        default=500,
        help="Number of samples per parameter value (default: 500)",
    )
    parser.add_argument(
        "--D-min",
        nargs="+",
        type=float,
        default=[3.0, 4.0],
        help="Minimum pupil diameter during stimulus (default: [3.0, 4.0])",
    )
    parser.add_argument(
        "--D-max",
        nargs="+",
        type=float,
        default=[7.0, 5.0],
        help="Maximum pupil diameter at baseline (default: [7.0, 5.0])",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output path for the plot",
    )
    parser.add_argument(
        "--save-results",
        type=str,
        default=None,
        help="Path to save evaluation results (pickle)",
    )
    parser.add_argument(
        "--load-results",
        type=str,
        default=None,
        help="Path to load previously saved results (pickle). If provided, evaluation will be skipped and the saved results will be plotted.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_false",
        help="Print detailed progress information",
    )

    args = parser.parse_args()

    # Validate argument combinations
    if args.fps_range and not args.noise:
        parser.error("--fps-range requires --noise")
    if args.noise_range and not args.fps:
        parser.error("--noise-range requires --fps")

    # Determine which parameter varies
    if args.fps_range:
        param_type = "fps"
        fps_min, fps_max, fps_step = args.fps_range
        param_list = list(np.arange(fps_min, fps_max + fps_step / 2, fps_step))
        fixed_fps = None
        fixed_noise = args.noise
        param_desc = (
            f"FPS: {fps_min} to {fps_max} step {fps_step}, fixed noise={fixed_noise}"
        )
    else:  # noise_range
        param_type = "noise"
        noise_min, noise_max, noise_step = args.noise_range
        param_list = list(np.arange(noise_min, noise_max + noise_step / 2, noise_step))
        fixed_fps = args.fps
        fixed_noise = None
        param_desc = f"Noise: {noise_min} to {noise_max} step {noise_step}, fixed fps={fixed_fps}"

    print("=" * 70)
    print("Latency Method Evaluation")
    print("=" * 70)
    print(f"Methods: {', '.join(args.methods)}")
    print(f"Parameter variation: {param_desc}")
    print(f"Samples per value: {args.num_samples}")
    print(f"D_min: {args.D_min}, D_max: {args.D_max}")
    print("=" * 70)

    # Load previously saved results or evaluate across all parameter values
    if args.load_results:
        all_results, metadata = load_results(args.load_results)
        print(f"\nLoaded results from: {args.load_results}")
        methods_to_plot = metadata.get("methods", args.methods)
        param_type = metadata.get("param_type", param_type)
        D_min_plot = metadata.get("D_min", args.D_min)
        D_max_plot = metadata.get("D_max", args.D_max)
    else:
        all_results = evaluate_across_parameters(
            param_list,
            param_type,
            args.num_samples,
            args.methods,
            fixed_fps,
            fixed_noise,
            args.D_min,
            args.D_max,
            verbose=args.verbose or True,
        )
        methods_to_plot = args.methods
        D_min_plot = args.D_min
        D_max_plot = args.D_max

        if args.save_results:
            metadata = {
                "methods": args.methods,
                "param_type": param_type,
                "D_min": args.D_min,
                "D_max": args.D_max,
                "param_list": param_list,
                "fixed_fps": fixed_fps,
                "fixed_noise": fixed_noise,
                "n_samples": args.num_samples,
            }
            save_results(args.save_results, all_results, metadata)

    # Generate plot
    output_file = args.output
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"latency_evaluation_{param_type}_{timestamp}.png"

    # Generate plot (either from computed results or loaded results)
    output_file = args.output
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"latency_evaluation_{param_type}_{timestamp}.png"

    # When plotting, ensure we use the appropriate methods and D_min/D_max
    plot_results(
        all_results, methods_to_plot, param_type, D_min_plot, D_max_plot, output_file
    )


if __name__ == "__main__":
    main()
