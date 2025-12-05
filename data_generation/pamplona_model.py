import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


def calc_latency(R: int, L_fL):
    """
    Based on equation 1 from Pamplona et al. 2009.
    """
    return 253 - 14 * np.log(L_fL) + 70 * R - 29 * R * np.log(L_fL)


def blondels_to_footlamberts(blondels):
    return blondels * 0.0929


def phi_to_blondels(phi):
    return phi / 10e-6


def diameter_from_phi(phi, phi_ref=4.8118e-10):
    """
    Uses equation 14 and 15 from Pamplona et al. 2009 to compute steady-state
    diameter from retinal light flux (phi).
    The value for phi_ref is taken from the paper.
    """
    MD = (5.2 - 0.45 * np.log(phi / phi_ref)) / 2.3026
    D = np.tanh(MD) * 3 + 4.9

    return D


def phi_from_diameter(D, phi_ref=4.8118e-10):
    """
    Uses equation 14 and 15 from Pamplona et al. 2009 to compute retinal light
    flux from a given Diameter (D).
    The value for phi_ref is taken from the paper.
    """
    MD = np.atanh((D - 4.9) / 3)
    phi = phi_ref * np.exp((2.3026 * MD - 5.2) / -0.45)

    return phi


def dMdD(D):
    """
    Derivative of M(D) = atanh((D - 4.9) / 3) with respect to D.
    Used in the RHS of the ODE."""
    u = (D - 4.9) / 3
    return 1.0 / (3 * (1 - u * u))


def plr_rhs_with_latency(t, D, phi_interp_step, tau_latency, phi_ref=4.8118e-10):
    """
    RHS for dD/dt including:
    - latency t - tau_latency
    - nonlinear iris mechanics (atanh term)
    - asymmetric S scaling (Eq 17)
    phi_interp_step must be a ZOH (previous) interpolator.
    """
    # effective illuminance time
    phi = float(phi_interp_step(t - tau_latency))

    M_D = np.arctanh((D - 4.9) / 3)

    rhs = 5.2 - 0.45 * np.log(phi / phi_ref) - 2.3026 * M_D

    return rhs / dMdD(D)


def simulate_dynamics_euler(phi_arr, time, D0, S):
    """
    Integrate Eq.16 (with Eq.17 speed scaling) using simple Euler method with
    zero-order-hold interpolation of the stimulus to avoid pre-onset ramps.
    tau_latency is computed dynamically based on stimulus intensity and capped
    at the time since the last stimulus change.
    """
    dt = float((time[1] - time[0]) / S)
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

    # Track the last stimulus change time
    last_phi_change_time = float(time[0])
    current_phi = float(phi_arr[0])

    for i in range(1, n):
        t = float(time[i - 1])

        # Check if stimulus has changed at this sample
        new_phi = float(phi_arr[i])
        if new_phi != current_phi:
            last_phi_change_time = t
            current_phi = new_phi

        # Time elapsed since last stimulus change
        time_since_change = t - last_phi_change_time

        # Compute current stimulus intensity in foot-lamberts
        current_stimulus_fL = blondels_to_footlamberts(phi_to_blondels(current_phi))

        # Compute tau_latency dynamically based on stimulus intensity
        tau_latency_dynamic = calc_latency(0.4, current_stimulus_fL)

        # Cap tau_latency at time since last stimulus change
        tau_latency = min(tau_latency_dynamic, time_since_change)

        dDdt = plr_rhs_with_latency(t, D[i - 1], phi_interp_step, tau_latency)

        if dDdt > 0:
            dt_c = dt / 3
        else:
            dt_c = dt
        D[i] = D[i - 1] + dDdt * dt_c

    return D


def simulate_dynamics_rk45(phi_arr, time, D0, tau_latency, S, rtol=1e-6, atol=1e-8):
    """
    Integrate Eq.16 (with Eq.17 speed scaling) using solve_ivp (RK45) with
    zero-order-hold interpolation of the stimulus to avoid pre-onset ramps.
    """
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
        fun=lambda t, y: plr_rhs_with_latency(t, y, phi_interp_step, tau_latency, S),
        t_span=(t0 / S, tf / S),
        y0=np.array([float(D0)]),
        method="RK45",
        t_eval=time / S,
        rtol=rtol,
        atol=atol,
        max_step=(time[1] - time[0]),  # keep solver step <= sampling interval
    )

    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")

    return sol.y[0]
