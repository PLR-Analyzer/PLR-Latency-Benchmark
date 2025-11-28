import matplotlib.pyplot as plt
import numpy as np
import stat_values
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from variability_curves import apply_isocurve


def diameter_from_phi(phi, phi_ref=4.8118e-10):
    MD = (5.2 - 0.45 * np.log(phi / phi_ref)) / 2.3026
    D = np.tanh(MD) * 3 + 4.9

    return D


def phi_from_diameter(D, phi_ref=4.8118e-10):
    MD = np.atanh((D - 4.9) / 3)
    phi = phi_ref * np.exp((2.3026 * MD - 5.2) / -0.45)

    return phi


def find_required_phi(
    D0,
    D_target,
    phi_baseline,
    phi_target_ss,
    time,
    on_mask,
    S,
    tol=1e-3,
    max_factor=1e6,
):
    """
    Find a stimulus flux (phi) that makes the simulated minimum diameter <= D_target.

    Uses exponential expansion then binary search over phi between `phi_baseline` and
    `high` (expanded bound). If even `high` can't reach the target, returns `high`.
    """
    # quick check: steady-state value might already achieve target
    if phi_target_ss <= phi_baseline:
        return phi_target_ss

    # helper to test a candidate phi
    def min_D_for_phi(phi_cand):
        phi_arr = np.full_like(time, phi_baseline)
        phi_arr[on_mask] = phi_cand
        D_sim = simulate_dynamics_euler(phi_arr, time, D0, tau_latency, S)
        return D_sim.min()

    # find an upper bound that achieves target (exponential expansion)
    low = phi_baseline
    high = max(phi_target_ss, phi_baseline * 10.0)
    minD = min_D_for_phi(high)
    expand_iter = 0
    while minD > D_target + tol and high / low < max_factor:
        low = high
        high = high * 10.0
        minD = min_D_for_phi(high)
        expand_iter += 1
        if expand_iter > 20:
            break

    # if even the expanded high doesn't reach the target, return high as best-effort
    if minD > D_target + tol:
        return high

    # binary search between low and high
    for _ in range(30):
        mid = 0.5 * (low + high)
        if min_D_for_phi(mid) <= D_target + tol:
            high = mid
        else:
            low = mid

    return high


def dMdD(D):
    u = (D - 4.9) / 3
    return 1.0 / (3 * (1 - u * u))


def plr_rhs_with_latency(
    t, D, phi_interp_step, tau_latency, S, phi_ref=4.8118e-10, verbose=False
):
    """
    RHS for dD/dt including:
    - latency t - tau_latency
    - nonlinear iris mechanics (atanh term)
    - asymmetric S scaling (Eq 17)
    phi_interp_step must be a ZOH (previous) interpolator.
    """
    # effective illuminance time
    phi = float(phi_interp_step(t - tau_latency))

    if verbose:
        print(phi)
        import time

        time.sleep(0.1)

    M_D = np.arctanh((D - 4.9) / 3)

    rhs = 5.2 - 0.45 * np.log(phi / phi_ref) - 2.3026 * M_D

    return rhs / dMdD(D)


def simulate_dynamics_euler(phi_arr, time, D0, tau_latency, S):
    """
    Integrate Eq.16 (with Eq.17 speed scaling) using simple Euler method with
    zero-order-hold interpolation of the stimulus to avoid pre-onset ramps.
    """
    dt = float(time[1] - time[0])
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

    increasing_counter = 0
    for i in range(1, n):
        t = float(time[i - 1])

        dD_test = plr_rhs_with_latency(t, D[i - 1], phi_interp_step, tau_latency, S)
        increasing = dD_test > 0
        # dDdt = plr_rhs_with_latency(t, D[i - 1], phi_interp_step, tau_latency, S)
        if increasing:
            increasing_counter += dt
            # D increasing → slow down
            dDdt = plr_rhs_with_latency(
                t,
                D[i - 1],
                phi_interp_step,
                min(tau_latency + increasing_counter, 3 * tau_latency),
                S,
                verbose=False,
            )
            if dDdt > 0:
                dt_c = dt / 3
        else:
            increasing_counter = 0
            # D decreasing → normal speed
            dDdt = plr_rhs_with_latency(
                t, D[i - 1], phi_interp_step, tau_latency, S, verbose=False
            )
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


def _compute_constriction_metrics(time, D, stim_time, tau_latency):
    """
    Compute average and maximal constriction velocities from a simulated trace.
    Returns (avg_constr_vel, max_constr_vel) in mm/s (both typically negative).
    """
    dt = float(time[1] - time[0])
    search_start_t = stim_time + tau_latency
    i_start = int(np.searchsorted(time, search_start_t))
    if i_start >= len(D) - 1:
        return None, None
    # find minimum after onset
    i_min_rel = np.argmin(D[i_start:])
    i_min = i_start + int(i_min_rel)
    if i_min <= i_start:
        return None, None
    deriv = np.gradient(D, dt)
    avg_constr_vel = float(np.mean(deriv[i_start:i_min]))
    max_constr_vel = float(np.min(deriv[i_start:i_min]))
    return avg_constr_vel, max_constr_vel


def tune_S(
    D_max,
    D_min,
    time,
    stim_time,
    on_mask,
    tau_latency,
    fps,
    target_avg,
    target_max,
):
    """
    Search for S that makes avg and max constriction velocities match targets.
    Returns best-fit S (float).
    """
    dt = 1.0 / float(fps)

    phi_base = phi_from_diameter(D_max)
    phi_stim_ss = phi_from_diameter(D_min)

    def loss_for_S(S):
        # build phi array with candidate S (S only influences dynamics, not phi)
        phi_arr = np.full_like(time, phi_base)
        # find a phi_stim that reaches D_min at transient if needed; here use phi_stim_ss
        phi_arr[on_mask] = phi_stim_ss
        D_sim = simulate_dynamics_euler(phi_arr, time, D_max, tau_latency, S)
        avg_v, max_v = _compute_constriction_metrics(
            time, D_sim, stim_time, tau_latency
        )
        if avg_v is None or max_v is None:
            return np.inf
        # squared-error loss on both metrics
        L = (avg_v - target_avg) ** 2 + (max_v - target_max) ** 2
        return float(L)

    # coarse grid (log-spaced to cover wide plausible S)
    coarse = np.concatenate(
        [
            np.linspace(0.5, 5, 10),
            np.linspace(5, 50, 10),
            np.linspace(50, 500, 10),
            np.linspace(500, 2000, 10),
        ]
    )
    coarse = np.unique(coarse)
    losses = np.array([loss_for_S(float(S)) for S in coarse])
    idx0 = int(np.nanargmin(losses))
    S0 = float(coarse[idx0])

    # refine around S0
    lower = max(0.1, S0 * 0.5)
    upper = S0 * 2.0 + 1e-9
    fine = np.linspace(lower, upper, 50)
    losses_fine = np.array([loss_for_S(float(S)) for S in fine])
    idx1 = int(np.nanargmin(losses_fine))
    S_best = float(fine[idx1])

    return S_best


if __name__ == "__main__":
    stim_time = 0.5
    tau_latency = np.random.normal(
        stat_values.CONSTRICTION_LATENCY_MEAN, stat_values.CONSTRICTION_LATENCY_STD
    )

    n = int(round(stat_values.DURATION * stat_values.FPS)) + 1
    n *= 100  # increase resolution for better accuracy
    time = np.linspace(0.0, stat_values.DURATION, n)

    D_max = stat_values.MAX_DIAMETER_MEAN
    D_min = stat_values.MINIMUM_DIAMETER_MEAN

    phi_arr = phi = np.full(n, phi_from_diameter(D_max))
    on_mask = (time >= stim_time) & (
        time < stim_time + stat_values.LIGHT_STIMULUS_DURATION
    )

    # S = tune_S(
    #     D_max,
    #     D_min,
    #     time,
    #     stim_time,
    #     on_mask,
    #     tau_latency,
    #     stat_values.FPS * 100,  # increased resolution
    #     stat_values.AVG_CONSTRICTION_VELOCITY_MEAN,
    #     stat_values.MAX_CONSTRICTION_VELOCITY_MEAN,
    # )
    S = 1

    phi_stim = find_required_phi(
        D_max,
        D_min,
        phi_from_diameter(D_max),
        phi_from_diameter(D_min),
        time,
        on_mask,
        S,
        tol=1e-3,
        max_factor=1e6,
    )
    phi_arr[on_mask] = phi_stim

    D = simulate_dynamics_euler(phi_arr, time, D_max, tau_latency, S)

    # r_l = np.random.random()
    # print(f"Using r_I = {r_l:.4f} for individual variability adjustment")
    # D = apply_isocurve(D, r_l)

    # compute constriction metrics to validate fit
    avg_v, max_v = _compute_constriction_metrics(time, D, stim_time, tau_latency)
    print(f"Simulated avg constriction velocity: {avg_v:.3f} mm/s")
    print(f"Simulated max constriction velocity: {max_v:.3f} mm/s")
    print(
        f"Target avg constriction velocity: {stat_values.AVG_CONSTRICTION_VELOCITY_MEAN:.3f} mm/s"
    )
    print(
        f"Target max constriction velocity: {stat_values.MAX_CONSTRICTION_VELOCITY_MEAN:.3f} mm/s"
    )
    if avg_v is not None:
        print(
            f"Avg error: {avg_v - stat_values.AVG_CONSTRICTION_VELOCITY_MEAN:+.3f} mm/s"
        )
    if max_v is not None:
        print(
            f"Max error: {max_v - stat_values.MAX_CONSTRICTION_VELOCITY_MEAN:+.3f} mm/s"
        )

    # plot
    plt.figure()
    plt.plot(time, D, label="Simulated Diameter")
    # plt.scatter(time, D)
    plt.xlabel("Time (s)")
    plt.ylabel("Diameter (mm)")
    plt.title("PLR Simulation Test")
    plt.axvspan(
        stim_time,
        stim_time + stat_values.LIGHT_STIMULUS_DURATION,
        color="yellow",
        alpha=0.2,
        label="Stimulus",
    )
    plt.axvline(stim_time + tau_latency, color="red", linestyle="--", label="Latency")
    plt.legend()
    plt.grid()
    plt.show()
