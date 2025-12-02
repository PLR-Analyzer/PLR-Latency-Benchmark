import matplotlib.pyplot as plt
import numpy as np
import stat_values
from pamplona_model import phi_from_diameter, simulate_dynamics_euler


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


def _compute_constriction_metrics(time, D, stim_time, tau_latency):
    """
    Compute average and maximal constriction velocities from a simulated trace.
    Returns (avg_constr_vel, max_constr_vel) in mm/s (both typically negative).
    """
    dt = float(time[1] - time[0]) / 700
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


if __name__ == "__main__":
    stim_time = 500
    tau_latency = np.random.normal(
        stat_values.CONSTRICTION_LATENCY_MEAN, stat_values.CONSTRICTION_LATENCY_STD
    )

    n = int(round(stat_values.DURATION * stat_values.FPS)) + 1
    n = stat_values.DURATION  # increase resolution for better accuracy
    time = np.linspace(0.0, stat_values.DURATION, n)

    D_max = stat_values.MAX_DIAMETER_MEAN
    D_min = stat_values.MINIMUM_DIAMETER_MEAN

    # S is a constant that affects the constriction/dilation velocity and
    # varies among individuals
    S = 600

    phi_arr = phi = np.full(n, phi_from_diameter(D_max))
    on_mask = (time >= stim_time) & (
        time < stim_time + stat_values.LIGHT_STIMULUS_DURATION
    )

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
    print(
        f"Baseline phi: {phi_from_diameter(D_max) * 1e6:.3e}\tStimulus phi: {phi_stim * 1e6:.3e} lux"
    )
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
