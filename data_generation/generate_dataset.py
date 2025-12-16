"""
Generate synthetic pupil light response (PLR) datasets using the simulation module.

This script creates multiple synthetic PLR recordings and saves them as npz files
in the data directory.
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

from data_generation import stat_values
from data_generation.simulation import simulate_sample


def generate_synthetic_dataset(
    n_samples=100,
    output_dir="data",
    duration=5.0,
    fps=25.0,
    stim_time=0.5,
    led_duration=0.167,
    noise_sd=0.03,
    verbose=True,
):
    """
    Generate multiple synthetic PLR recordings and save them as npz files.

    Parameters
    ----------
    n_samples : int
        Number of synthetic recordings to generate.
    output_dir : str or Path
        Directory where npz files will be saved.
    duration : float
        Duration of each recording in seconds.
    fps : float
        Sampling rate in frames per second.
    stim_time : float
        Time when stimulus (LED) turns on in seconds.
    led_duration : float
        Duration of LED pulse in seconds.
    verbose : bool
        Print progress information.

    Returns
    -------
    list
        Paths to generated npz files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_files = []

    for i in range(n_samples):
        if verbose and (i + 1) % max(1, n_samples // 10) == 0:
            print(f"Generating sample {i + 1}/{n_samples}...")

        D_max = 7.0
        D_min = 3.0

        # Simulate with a unique seed based on sample index
        time, D_obs, D_clean, true_latency, params = simulate_sample(
            duration=duration,
            fps=fps,
            stim_time=stim_time,
            stim_duration=200,
            D_min=D_min,
            D_max=D_max,
            seed=i,  # Deterministic seed based on index
            noise_sd=noise_sd,
            drift_amp=0.2,
        )

        # Create filename with timestamp for uniqueness
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        filename = f"synthetic_sample_{i:04d}_{timestamp}.npz"
        filepath = output_dir / filename

        # Save to npz file
        np.savez(
            filepath,
            diameter_observed=D_obs,
            diameter_clean=D_clean,
            true_latency=true_latency,
            fps=fps,
            stim_time=stim_time,
            led_duration=led_duration,
        )

        generated_files.append(filepath)
        if verbose:
            print(f"  Saved: {filename}")

    if verbose:
        print(f"\nGenerated {n_samples} synthetic datasets in {output_dir}")

    return generated_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic PLR datasets using simulation.py"
    )
    parser.add_argument(
        "-n",
        "--num-samples",
        type=int,
        default=100,
        help="Number of synthetic recordings to generate (default: 100)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="data",
        help="Output directory for npz files (default: data)",
    )
    parser.add_argument(
        "-d",
        "--duration",
        type=float,
        default=stat_values.DURATION,
        help="Duration of each recording in seconds (default: 5.0)",
    )
    parser.add_argument(
        "-f",
        "--fps",
        type=float,
        default=stat_values.FPS,
        help="Sampling rate in fps (default: 25.0)",
    )
    parser.add_argument(
        "-s",
        "--stim-time",
        type=float,
        default=stat_values.LIGHT_STIMULUS_START,
        help="Time when stimulus turns on in seconds (default: 0.5)",
    )
    parser.add_argument(
        "-l",
        "--led-duration",
        type=float,
        default=stat_values.LIGHT_STIMULUS_DURATION,
        help="Duration of LED pulse in seconds (default: 0.167)",
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=0.03,
        help="Duration of LED pulse in seconds (default: 0.03)",
    )

    args = parser.parse_args()

    generate_synthetic_dataset(
        n_samples=args.num_samples,
        output_dir=args.output_dir,
        duration=args.duration,
        fps=args.fps,
        stim_time=args.stim_time,
        led_duration=args.led_duration,
        noise_sd=args.noise,
    )
