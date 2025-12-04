from test import simulate_dynamics_rk45

import numpy as np
from matplotlib import pyplot as plt
from pamplona_model import diameter_from_phi, plr_rhs_with_latency
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


def custom_solver(phi_arr, time, D0, tau_latency):
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

        dDdt = plr_rhs_with_latency(t, D[i - 1], phi_interp_step, tau_latency)
        D[i] = D[i - 1] + dDdt * dt

    return D


def rk45_solver(phi_arr, time, D0, tau_latency, rtol=1e-6, atol=1e-8):
    """
    Integrate Eq.16 (with Eq.17 speed scaling) using solve_ivp (RK45) with
    zero-order-hold interpolation of the stimulus to avoid pre-onset ramps.
    """

    time = time / 600
    tau_latency = tau_latency / 600

    # build step interpolator (ZOH / previous)
    phi_interp_step = interp1d(
        time,
        phi_arr,
        kind="previous",  #! This importent! Otherwise, fractional onset time is not handled correctly.
        bounds_error=False,
        fill_value=(phi_arr[0], phi_arr[-1]),
        assume_sorted=True,
    )

    t0 = float(time[0])
    tf = float(time[-1])

    sol = solve_ivp(
        fun=lambda t, y: plr_rhs_with_latency(t, y, phi_interp_step, tau_latency),
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
    stim_start = 500
    stim_duration = 1000
    latency = 200

    # 5 seconds PLR with 60 Hz
    sample_rate = 60
    sim_duration = 5000
    n = sample_rate * sim_duration // 1000
    time = np.linspace(0, sim_duration, n)

    phi_arr = phi = np.full(n, 91e-06)  # 140 lux baseline
    phi_stim = 540e-06  # 1000 lux stimulus
    on_mask = (time >= stim_start) & (time < stim_start + stim_duration)
    phi_arr[on_mask] = phi_stim

    # Diameter from custom solver
    D1 = custom_solver(phi_arr, time, diameter_from_phi(phi[0]), latency)
    D2 = rk45_solver(phi_arr, time, diameter_from_phi(phi[0]), latency)
    # print(len(time), len(D), D[-1], time[-1])

    plt.figure()
    plt.plot(time, D1, label="Simulated Diameter D1")
    plt.plot(time, D2, label="Simulated Diameter D2")
    # plt.scatter(time, D)
    plt.xlabel("Time (s)")
    plt.ylabel("Diameter (mm)")
    plt.title("PLR Simulation Test")
    plt.axvspan(
        stim_start,
        stim_start + stim_duration,
        color="yellow",
        alpha=0.2,
        label="Stimulus",
    )
    plt.axvline(stim_start + latency, color="red", linestyle="--", label="Latency")
    plt.axvline(
        stim_start + stim_duration + latency,
        color="red",
        linestyle="--",
        label="Latency",
    )
    plt.legend()
    plt.grid()
    plt.show()
