import matplotlib.pyplot as plt
import numpy as np


def flux_from_diameter(D, phi_ref=1.0):
    """Compute retinal flux φ that gives steady-state diameter D (from Eq. 16)."""
    arg = np.clip((D - 4.9) / 3.0, -0.9999, 0.9999)
    rhs = 5.2 - 2.3026 * np.arctanh(arg)
    phi_ratio = np.exp(rhs / 0.45)
    return phi_ref * phi_ratio


def _simulate_dynamics(phi_arr, dt, D0, delay_samples, dMdD, phi_ref):
    """Simulate diameter time course for a given `phi_arr`.

    Returns array `D` of same length as `phi_arr`.
    """
    n = len(phi_arr)
    D = np.zeros(n)
    D[0] = D0
    for i in range(1, n):
        idx_delay = max(0, i - delay_samples)
        ratio = max(1e-8, phi_arr[idx_delay] / phi_ref)
        rhs = 5.2 - 0.45 * np.log(ratio)
        arg = np.clip((D[i - 1] - 4.9) / 3.0, -0.9999, 0.9999)
        mech = 2.3026 * np.arctanh(arg)
        dDdt = (rhs - mech) / dMdD
        D[i] = D[i - 1] + dDdt * dt
    return D


def _find_required_phi(
    D0,
    D_target,
    phi_baseline,
    phi_target_ss,
    time,
    on_mask,
    delay_samples,
    dMdD,
    phi_ref,
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
        D_sim = _simulate_dynamics(phi_arr, dt, D0, delay_samples, dMdD, phi_ref)
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
    fps=30.0,
    stim_time=0.5,
    led_duration=0.167,
    phi_ref=1.0,
    dMdD=1.0,
    noise_sd=0.03,
    drift_amp=0.05,
    seed=None,
):
    """
    Simulate a population-realistic PLR using Equation (16) with parameters drawn
    from literature (Napieralski & Rynkiewicz, 2019; population data table).

    Returns
    -------
    time, D_obs, D_clean, true_latency, params
    """
    if seed is not None:
        np.random.seed(seed)

    # --- draw subject-specific parameters from reported population stats ---
    D_max = np.random.normal(5.63, 0.79)  # baseline diameter (dark)
    D_min = np.random.normal(3.78, 0.56)  # minimum diameter (bright)
    tau_latency = np.random.normal(0.21175, 0.00951)

    # steady-state flux values corresponding to these diameters
    phi_baseline = flux_from_diameter(D_max, phi_ref)
    # steady-state stimulus flux corresponding to D_min (may be insufficient for
    # brief pulses). We attempt to find an increased `phi_stim` that makes the
    # transient reach `D_min` during the short LED on-time.
    phi_stim_ss = flux_from_diameter(D_min, phi_ref)

    dt = 1.0 / fps
    n = int(np.round(duration * fps)) + 1
    time = np.linspace(0, duration, n)

    # light step function (initially using baseline; we'll fill in the stimulus
    # flux below, possibly boosted to reach D_min during the short pulse)
    phi = np.full(n, phi_baseline)
    on_mask = (time >= stim_time) & (time < stim_time + led_duration)

    # if the LED is very short relative to dynamics, the steady-state flux
    # `phi_stim_ss` might not drive the transient low enough. Find a stronger
    # stimulus flux if needed.
    delay_samples = int(np.ceil(tau_latency / dt))
    phi_stim = _find_required_phi(
        D_max,
        D_min,
        phi_baseline,
        phi_stim_ss,
        time,
        on_mask,
        delay_samples,
        dMdD,
        phi_ref,
        dt,
    )
    phi[on_mask] = phi_stim

    # integration
    D = _simulate_dynamics(phi, dt, D_max, delay_samples, dMdD, phi_ref)

    # add slow drift + measurement noise
    hippus_freq = np.random.uniform(0.05, 0.3)
    drift = drift_amp * np.sin(
        2 * np.pi * hippus_freq * time + np.random.uniform(0, 2 * np.pi)
    )
    noise = np.random.normal(0, noise_sd, size=n)

    D_clean = D.copy()
    D_obs = D_clean + drift + noise

    params = dict(
        D_max=D_max,
        D_min=D_min,
        phi_baseline=phi_baseline,
        phi_stim=phi_stim,
        tau_latency=tau_latency,
        noise_sd=noise_sd,
        drift_amp=drift_amp,
        hippus_freq=hippus_freq,
    )

    return time, D_obs, D_clean, stim_time + tau_latency, params


# --- Example usage ---
if __name__ == "__main__":
    t, D_obs, D_clean, true_lat, params = simulate_plr_eq16_population(seed=None)
    print("Parameters:", params)

    plt.figure(figsize=(10, 5))
    plt.plot(t, D_obs, label="Observed (noisy)", alpha=0.7)
    plt.plot(t, D_clean, marker="o", label="Clean Eq.16 simulation", linewidth=2)
    plt.axvline(true_lat, color="r", ls="--", label=f"True latency = {true_lat:.3f}s")
    plt.axvspan(0.5, 0.5 + 0.167, color="yellow", alpha=0.2, label="LED ON")
    plt.xlabel("Time (s)")
    plt.ylabel("Pupil diameter (mm)")
    plt.legend()
    plt.tight_layout()
    plt.show()
