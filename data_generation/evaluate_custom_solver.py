import numpy as np
from matplotlib import pyplot as plt
from pamplona_model import diameter_from_phi, phi_from_diameter, plr_rhs_with_latency
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


def blondels_to_lumens(blondels):
    """Convert Blondel units to lumens.

    Based on formula from Pamplona et al..
    """
    return blondels * 10e-6


def blondels_to_phi(blondels):

    D = 4.9 - 3 * np.tanh(0.4 * (np.log10(blondels)) - 0.5)
    return np.pi * (D / 2) ** 2 * blondels * 10e-6


def blondels_to_footlamberts(blondels):
    return blondels * 0.0929


def calc_latency(R: int, L_fL):
    """
    Based on equation 1 from Pamplona et al. 2009.
    """
    return 253 - 14 * np.log(L_fL) + 70 * R - 29 * R * np.log(L_fL)


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

        dDdt = plr_rhs_with_latency(t, D[i - 1], phi_interp_step, latency)
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
    stim_changes = [200, 2500, 4200, 7000]
    stim_values = [10e-2, 10e2, 10 ** (-0.5), 10]

    latency = 200

    # 5 seconds PLR with 60 Hz
    sample_rate = 60
    duration = 9000
    n = sample_rate * duration // 1000
    time = np.linspace(0, duration, n)

    phi_arr = phi = np.full(n, blondels_to_lumens(stim_values[1]))  # 140 lux baseline
    stim_mask1 = (time >= stim_changes[0]) & (time < stim_changes[1])
    stim_mask2 = (time >= stim_changes[1]) & (time < stim_changes[2])
    stim_mask3 = (time >= stim_changes[2]) & (time < stim_changes[3])
    stim_mask4 = time >= stim_changes[3]

    phi_arr[stim_mask1] = blondels_to_lumens(stim_values[0])
    phi_arr[stim_mask2] = blondels_to_lumens(stim_values[1])
    phi_arr[stim_mask3] = blondels_to_lumens(stim_values[2])
    phi_arr[stim_mask4] = blondels_to_lumens(stim_values[3])

    # Diameter from custom solver
    D1 = custom_solver(
        phi_arr, time, diameter_from_phi(blondels_to_lumens(stim_values[1])), latency
    )
    D2 = rk45_solver(
        phi_arr, time, diameter_from_phi(blondels_to_lumens(stim_values[1])), latency
    )

    plt.figure()
    plt.plot(time, D1, label="Simulated Diameter D1")
    plt.plot(time, D2, label="Simulated Diameter D2")
    plt.xlabel("Time (s)")
    plt.ylabel("Diameter (mm)")
    plt.title("PLR Simulation Test")
    plt.axvspan(
        stim_changes[0],
        stim_changes[1],
        color="coral",
        alpha=0.2,
        label="$10^{-2}$blondel",
    )
    plt.axvspan(
        stim_changes[1],
        stim_changes[2],
        color="gold",
        alpha=0.2,
        label="$10^{2}$blondel",
    )
    plt.axvspan(
        stim_changes[2],
        stim_changes[3],
        color="powderblue",
        alpha=0.2,
        label="$10^{-0.5}$blondel",
    )
    plt.axvspan(
        stim_changes[3],
        duration,
        color="fuchsia",
        alpha=0.2,
        label="$10$blondel",
    )
    plt.legend()
    plt.grid()
    plt.show()
