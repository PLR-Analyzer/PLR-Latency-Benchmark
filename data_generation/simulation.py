import matplotlib.pyplot as plt
import numpy as np


def flux_from_diameter(D, phi_ref=1.0):
    arg = np.clip((D - 4.9) / 3.0, -0.9999, 0.9999)
    rhs = 5.2 - 2.3026 * np.arctanh(arg)
    phi_ratio = np.exp(rhs / 0.45)
    return phi_ref * phi_ratio


def plr_rhs(D, phi, S):
    """Compute the right-hand side of the Pamplona-Olivera Model (Eq 16)."""

    ratio = max(1e-9, phi / 1.0)
    rhs = 5.2 - 0.45 * np.log(ratio)

    arg = np.clip((D - 4.9) / 3.0, -0.9999, 0.9999)
    mech = 2.3026 * np.arctanh(arg)

    dDdt = rhs - mech

    # Asymmetric dynamics (Eq. 17)
    if dDdt < 0:
        # constriction → fast
        return dDdt / S
    else:
        # dilation → 3x slower
        return dDdt / (3 * S)


def rk4_step(D, dt, phi, S):
    """Runge-Kutta 4 integration step."""
    k1 = plr_rhs(D, phi, S)
    k2 = plr_rhs(D + 0.5 * dt * k1, phi, S)
    k3 = plr_rhs(D + 0.5 * dt * k2, phi, S)
    k4 = plr_rhs(D + dt * k3, phi, S)

    return D + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate_dynamics_rk4(phi_arr, time, D0, t_onset, S):
    """Simulate dynamics using RK4 with continuous onset."""

    n = len(time)
    D = np.zeros(n)
    D[0] = D0

    for i in range(1, n):

        dt_full = time[i] - time[i - 1]

        # Pre-onset → no RHS contribution
        if time[i] <= t_onset:
            D[i] = D[i - 1]
            continue

        # Partial exposure for the first post-onset frame
        if time[i - 1] < t_onset < time[i]:
            dt_eff = time[i] - t_onset
        else:
            dt_eff = dt_full

        # Effective illuminance (no discrete sample delay needed here)
        phi = phi_arr[i]

        # One RK4 step using only dt_eff (fractional frame time)
        D[i] = rk4_step(D[i - 1], dt_eff, phi, S)

    return D


def _compute_metrics_from_trace(time, D, stim_time, led_duration, tau_latency):
    dt = time[1] - time[0]
    # Start searching after stimulus + latency
    search_start_t = stim_time + tau_latency
    i_start = int(np.searchsorted(time, search_start_t))
    # index of minimal diameter after stimulus
    if i_start >= len(D) - 1:
        return None  # can't compute
    i_min_rel = np.argmin(D[i_start:])  # relative to i_start
    i_min = i_start + i_min_rel

    # compute derivative
    deriv = np.gradient(D, dt)

    # average constriction velocity: mean derivative between onset and minimum
    if i_min <= i_start:
        return None
    avg_constr_vel = np.mean(deriv[i_start:i_min])  # should be negative

    # max constriction velocity
    max_constr_vel = np.min(deriv[i_start:i_min])  # most negative

    # dilation velocity: average derivative in window after minimum (take next 0.5s or until end)
    post_start = i_min + 1
    post_end = min(len(D), post_start + int(0.5 / dt))
    if post_end <= post_start:
        return None
    avg_dil_vel = np.mean(deriv[post_start:post_end])  # should be positive

    # 75% recovery time: time to recover to D75 = min + 0.75*(max-min)
    D_min = D[i_min]
    D_max = np.max(D[: i_start + 1])  # baseline estimate (before constriction)
    D75 = D_min + 0.75 * (D_max - D_min)
    # find first index after i_min where D >= D75
    idx_after = np.where(D[i_min:] >= D75)[0]
    if idx_after.size == 0:
        t75 = np.nan
    else:
        t75 = idx_after[0] * dt  # time relative to i_min
    return {
        "avg_constr_vel": float(avg_constr_vel),
        "max_constr_vel": float(max_constr_vel),
        "avg_dil_vel": float(avg_dil_vel),
        "t75": float(t75),
    }


def tune_S(
    D_max=5.63,
    D_min=3.78,
    stim_time=0.5,
    led_duration=0.167,
    fps=200,
    tau_latency=0.21175,
    target_vel_mean=-4.11,
    target_vel_max=-5.15,
    target_dil_vel=1.02,
    target_75pct=1.77,
):
    """
    Robust two-stage search for S.
    Returns S_best (float).
    """
    dt = 1.0 / fps
    duration = 4.0
    n = int(duration * fps) + 1
    time = np.linspace(0, duration, n)

    phi_base = flux_from_diameter(D_max)
    phi_stim = flux_from_diameter(D_min)
    phi = np.full(n, phi_base)
    on_mask = (time >= stim_time) & (time < stim_time + led_duration)
    phi[on_mask] = phi_stim
    delay_samples = int(max(0, round(tau_latency / dt)))

    # loss weighting
    w = dict(avg=1.0, maxc=1.0, dil=1.0, t75=0.5)

    def loss_for_S(S):
        D = simulate_dynamics_rk4(phi, time, D_max, stim_time, S)
        metrics = _compute_metrics_from_trace(
            time, D, stim_time, led_duration, tau_latency
        )
        if metrics is None:
            return np.inf
        # squared-error style loss (scale each term to typical magnitude)
        L = 0.0
        L += w["avg"] * (metrics["avg_constr_vel"] - target_vel_mean) ** 2
        L += w["maxc"] * (metrics["max_constr_vel"] - target_vel_max) ** 2
        L += w["dil"] * (metrics["avg_dil_vel"] - target_dil_vel) ** 2
        # t75 is in sec relative to min; if nan, penalize heavily
        if np.isnan(metrics["t75"]):
            L += 100.0
        else:
            L += w["t75"] * (metrics["t75"] - target_75pct) ** 2
        return L

    # Coarse grid (log-spaced to capture wide range)
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
    idx0 = np.nanargmin(losses)
    S0 = coarse[idx0]

    # Fine grid around S0
    lower = max(0.1, S0 * 0.5)
    upper = S0 * 2.0 + 1e-9
    fine = np.linspace(lower, upper, 50)
    losses_fine = np.array([loss_for_S(float(S)) for S in fine])
    idx1 = np.nanargmin(losses_fine)
    S_best = float(fine[idx1])

    return S_best


def _find_required_phi(
    D0,
    D_target,
    phi_baseline,
    phi_target_ss,
    time,
    stim_time,
    on_mask,
    delay_samples,
    S,
    dt,
    tol=1e-3,
    max_factor=1e6,
):
    """Find a stimulus flux (phi) that makes the simulated minimum diameter <= D_target.

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
        D_sim = simulate_dynamics_rk4(phi_arr, time, D0, stim_time, S)
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


def simulate_plr_eq16_population(
    duration=5.0,
    fps=200,
    stim_time=0.5,
    led_duration=0.167,
    noise_sd=0.03,
    drift_amp=0.05,
    seed=None,
):
    if seed is not None:
        np.random.seed(seed)

    # sample population params
    D_max = float(np.random.normal(5.63, 0.79))
    D_min = float(np.random.normal(3.78, 0.56))
    tau_latency = float(np.random.normal(0.21175, 0.00951))

    # steady-state flux values corresponding to these diameters
    phi_base = flux_from_diameter(D_max)
    # steady-state stimulus flux corresponding to D_min (may be insufficient for
    # brief pulses). We attempt to find an increased `phi_stim` that makes the
    # transient reach `D_min` during the short LED on-time.
    phi_stim = flux_from_diameter(D_min)

    # tune S for this "subject" (robust)
    S = tune_S(
        D_max=D_max,
        D_min=D_min,
        stim_time=stim_time,
        led_duration=led_duration,
        fps=fps,
        tau_latency=tau_latency,
    )

    dt = 1.0 / fps
    n = int(round(duration * fps)) + 1
    time = np.linspace(0, duration, n)
    phi = np.full(n, phi_base)
    on_mask = (time >= stim_time) & (time < stim_time + led_duration)
    # if the LED is very short relative to dynamics, the steady-state flux
    # `phi_stim_ss` might not drive the transient low enough. Find a stronger
    # stimulus flux if needed.
    delay_samples = int(np.ceil(tau_latency / dt))

    phi_stim = _find_required_phi(
        D_max,
        D_min,
        phi_base,
        phi_stim,
        time,
        stim_time,
        on_mask,
        delay_samples,
        S,
        dt,
    )
    phi[on_mask] = phi_stim

    delay_samples = int(max(0, round(tau_latency / dt)))

    # integrate with Eq17 dynamics
    D_clean = simulate_dynamics_rk4(phi, time, D_max, stim_time, S)

    # add hippus + noise
    hippus_freq = np.random.uniform(0.05, 0.3)
    drift = drift_amp * np.sin(
        2 * np.pi * hippus_freq * time + np.random.uniform(0, 2 * np.pi)
    )
    noise = np.random.normal(0, noise_sd, size=n)
    D_obs = D_clean + drift + noise

    params = dict(
        D_max=D_max,
        D_min=D_min,
        tau_latency=tau_latency,
        S=S,
        phi_base=phi_base,
        phi_stim=phi_stim,
    )
    return time, D_obs, D_clean, stim_time + tau_latency, params


# Quick demo
if __name__ == "__main__":
    t, D_obs, D_clean, true_lat, params = simulate_plr_eq16_population(seed=0)
    print("params:", params)
    plt.plot(t, D_clean, label="clean")
    plt.plot(t, D_obs, alpha=0.6, label="obs")
    plt.axvline(true_lat, color="r", ls="--", label="true latency")
    plt.legend()
    plt.show()
