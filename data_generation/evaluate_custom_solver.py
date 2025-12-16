import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from data_generation.pamplona_model import (
    blondels_to_footlamberts,
    calc_latency,
    diameter_from_phi,
    phi_from_diameter,
    plr_rhs_with_latency,
)


def blondels_to_lumens(blondels):
    """Convert Blondel units to lumens.

    Based on formula from Pamplona et al..
    """
    return blondels * 10e-6


def blondels_to_phi(blondels):

    D = 4.9 - 3 * np.tanh(0.4 * (np.log10(blondels)) - 0.5)
    return np.pi * (D / 2) ** 2 * blondels * 10e-6


def custom_solver(phi_arr, time, D0, tau_arr):
    """
    Integrate Eq.16 (with Eq.17 speed scaling) using simple Euler method with
    zero-order-hold interpolation of the stimulus to avoid pre-onset ramps.
    """
    dt = float(time[1] - time[0]) / 600
    n = len(time)
    D = np.empty(n, dtype=np.float64)
    D[0] = float(D0)

    # build step interpolator (ZOH / previous)
    phi_interp_step = interp1d(
        time,
        phi_arr,
        kind="previous",  #! This importent! Otherwise, fractional onset time is not handled correctly.
        bounds_error=False,
        fill_value=(phi_arr[0], phi_arr[-1]),
        assume_sorted=True,
    )

    for i in range(1, n):
        t = float(time[i - 1])

        dDdt = plr_rhs_with_latency(D[i - 1], float(phi_interp_step(t - tau_arr[i])))
        D[i] = D[i - 1] + dDdt * dt

    return D


def rk45_solver(phi_arr, time, D0, tau_arr, rtol=1e-6, atol=1e-8):
    """
    Integrate Eq.16 (with Eq.17 speed scaling) using solve_ivp (RK45) with
    zero-order-hold interpolation of the stimulus to avoid pre-onset ramps.
    """

    time = time / 600
    # tau_latency = tau_latency / 600

    # build step interpolator (ZOH / previous)
    phi_interp_step = interp1d(
        time,
        phi_arr,
        kind="previous",  #! This importent! Otherwise, fractional onset time is not handled correctly.
        bounds_error=False,
        fill_value=(phi_arr[0], phi_arr[-1]),
        assume_sorted=True,
    )

    tau_interp_step = interp1d(
        time,
        tau_arr,
        kind="previous",  #! This importent! Otherwise, fractional onset time is not handled correctly.
        bounds_error=False,
        fill_value=(phi_arr[0], phi_arr[-1]),
        assume_sorted=True,
    )

    t0 = float(time[0])
    tf = float(time[-1])

    sol = solve_ivp(
        fun=lambda t, y: plr_rhs_with_latency(
            y,
            phi_interp_step(t - tau_interp_step(t) / 600),
        ),
        t_span=(t0, tf),
        y0=np.array([float(D0)]),
        method="RK45",
        t_eval=time,
        rtol=rtol,
        atol=atol,
        first_step=time[1] - time[0],
        max_step=(time[1] - time[0]),  # keep solver step <= sampling interval
    )

    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")

    return sol.y[0]


if __name__ == "__main__":
    stim_changes = [200, 2500, 4200, 7000]
    stim_values = [10e-2, 10e2, 10 ** (-0.5), 10]

    # Run simulations for two sample rates and plot side-by-side
    sample_rates = [5, 25]
    duration = 9000
    fig, axes = plt.subplots(1, len(sample_rates), figsize=(12, 4), sharey=True)
    if len(sample_rates) == 1:
        axes = [axes]

    for ax, sample_rate in zip(axes, sample_rates):
        n = sample_rate * duration // 1000
        time = np.linspace(0, duration, n)

        phi_arr = np.full(n, blondels_to_lumens(stim_values[1]))
        latency_arr = np.full(
            n, calc_latency(0.4, blondels_to_footlamberts(stim_values[1]))
        )
        stim_mask1 = (time >= stim_changes[0]) & (time < stim_changes[1])
        stim_mask2 = (time >= stim_changes[1]) & (time < stim_changes[2])
        stim_mask3 = (time >= stim_changes[2]) & (time < stim_changes[3])
        stim_mask4 = time >= stim_changes[3]

        phi_arr[stim_mask1] = blondels_to_lumens(stim_values[0])
        phi_arr[stim_mask2] = blondels_to_lumens(stim_values[1])
        phi_arr[stim_mask3] = blondels_to_lumens(stim_values[2])
        phi_arr[stim_mask4] = blondels_to_lumens(stim_values[3])

        latency_arr[stim_mask1] = calc_latency(
            0.4, blondels_to_footlamberts(stim_values[0])
        )
        latency_arr[stim_mask2] = calc_latency(
            0.4, blondels_to_footlamberts(stim_values[1])
        )
        latency_arr[stim_mask3] = calc_latency(
            0.4, blondels_to_footlamberts(stim_values[2])
        )
        latency_arr[stim_mask4] = calc_latency(
            0.4, blondels_to_footlamberts(stim_values[3])
        )

        D0 = diameter_from_phi(blondels_to_lumens(stim_values[1]))
        D1 = custom_solver(phi_arr, time, D0, latency_arr)
        D2 = rk45_solver(phi_arr, time, D0, latency_arr)

        ax.plot(time, D1, label="Simulated Diameter (Euler)")
        ax.plot(time, D2, linestyle="--", label="Simulated Diameter (RK45)")
        ax.set_xlabel("Time (ms)")
        ax.set_title(f"Sample rate: {sample_rate} Hz")
        if ax is axes[0]:
            ax.set_ylabel("Diameter (mm)")

        ax.axvspan(
            stim_changes[0],
            stim_changes[1],
            color="coral",
            alpha=0.1,
            label="$10^{-2}$blondel",
        )
        ax.axvspan(
            stim_changes[1],
            stim_changes[2],
            color="gold",
            alpha=0.1,
            label="$10^{2}$blondel",
        )
        ax.axvspan(
            stim_changes[2],
            stim_changes[3],
            color="powderblue",
            alpha=0.1,
            label="$10^{-0.5}$blondel",
        )
        ax.axvspan(
            stim_changes[3], duration, color="fuchsia", alpha=0.1, label="$10$blondel"
        )
        ax.legend()
        ax.grid()

    plt.tight_layout()
    plt.show()

    # --- Compute MSE between solvers across sample rates ---
    rates = np.arange(5, 101, 5)
    mses = []
    for sample_rate in rates:
        n = sample_rate * duration // 1000
        time = np.linspace(0, duration, n)

        phi_arr = np.full(n, blondels_to_lumens(stim_values[1]))
        latency_arr = np.full(
            n, calc_latency(0.4, blondels_to_footlamberts(stim_values[1]))
        )
        stim_mask1 = (time >= stim_changes[0]) & (time < stim_changes[1])
        stim_mask2 = (time >= stim_changes[1]) & (time < stim_changes[2])
        stim_mask3 = (time >= stim_changes[2]) & (time < stim_changes[3])
        stim_mask4 = time >= stim_changes[3]

        phi_arr[stim_mask1] = blondels_to_lumens(stim_values[0])
        phi_arr[stim_mask2] = blondels_to_lumens(stim_values[1])
        phi_arr[stim_mask3] = blondels_to_lumens(stim_values[2])
        phi_arr[stim_mask4] = blondels_to_lumens(stim_values[3])

        latency_arr[stim_mask1] = calc_latency(
            0.4, blondels_to_footlamberts(stim_values[0])
        )
        latency_arr[stim_mask2] = calc_latency(
            0.4, blondels_to_footlamberts(stim_values[1])
        )
        latency_arr[stim_mask3] = calc_latency(
            0.4, blondels_to_footlamberts(stim_values[2])
        )
        latency_arr[stim_mask4] = calc_latency(
            0.4, blondels_to_footlamberts(stim_values[3])
        )

        D0 = diameter_from_phi(blondels_to_lumens(stim_values[1]))
        D1 = custom_solver(phi_arr, time, D0, latency_arr)
        D2 = rk45_solver(phi_arr, time, D0, latency_arr)

        # Ensure arrays are same length and compute MSE
        if D1.shape != D2.shape:
            # interpolate D2 onto D1 time grid if needed
            D2_interp = np.interp(time, time, D2)
        else:
            D2_interp = D2

        mse = float(np.mean((D1 - D2_interp) ** 2))
        mses.append(mse)

    # Plot MSE vs sample rate
    plt.figure(figsize=(8, 4))
    plt.plot(rates, mses, marker="o")
    plt.xlabel("Sample rate (Hz)")
    plt.ylabel("MSE between solvers ($mm^2$)")
    plt.title("MSE between custom solver and RK45 solver from SciPy")
    plt.grid()
    plt.tight_layout()
    plt.show()
