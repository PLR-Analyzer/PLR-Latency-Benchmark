# SPDX-FileCopyrightText: 2026 Marcel Schepelmann <schepelmann@chi.uni-hannover.de>
#
# SPDX-License-Identifier: GPL-3.0-or-later

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


def plr_rhs_with_latency(D, phi, phi_ref=4.8118e-10):
    # effective illuminance time
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
    D = np.zeros(n, dtype=np.float64)
    D[0] = float(D0)

    # build step interpolator (ZOH / previous)
    phi_interp = interp1d(
        time,
        phi_arr,
        kind="previous",  #! This importent! Otherwise, fractional onset time is not handled correctly.
        bounds_error=False,
        fill_value=(phi_arr[0], phi_arr[-1]),
        assume_sorted=True,
    )

    # initial stimulus state
    current_phi = float(phi_arr[0])
    current_stim_fL = blondels_to_footlamberts(phi_to_blondels(current_phi))

    # Active latency window
    active_latency = calc_latency(0.4, current_stim_fL)
    active_latency_start = float(time[0])
    # sample_phi_at_latency_start must be the phi value before the change that created the latency
    sample_phi_at_latency_start = current_phi

    # pending latency stores (tau_pending, phi_before_pending)
    pending_latency = None

    for i in range(1, n):
        t = float(time[i])
        new_phi = float(phi_arr[i])

        # detect stimulus change at this sample
        if new_phi != current_phi:
            # phi_before is the stimulus value before this change
            phi_before = current_phi
            current_phi = new_phi

            # compute latency for the new stimulus level
            new_fL = blondels_to_footlamberts(phi_to_blondels(new_phi))
            tau_new = calc_latency(0.4, new_fL)

            if tau_new <= active_latency:
                # adopt immediately (shorter latency) and anchor to phi_before
                active_latency = tau_new
                active_latency_start = t
                sample_phi_at_latency_start = phi_before
                pending_latency = None
            else:
                # longer latency: store pending along with its phi_before
                pending_latency = (tau_new, phi_before)

        # if active latency window finished, adopt pending (if any)
        if pending_latency is not None:
            tau_pending, phi_before_pending = pending_latency
            if t >= active_latency_start + active_latency:
                active_latency = tau_pending
                active_latency_start = t
                # anchor to the phi before the pending change
                sample_phi_at_latency_start = phi_before_pending
                pending_latency = None

        # compute delayed-sampling time and choose phi_delayed
        t_delay = t - active_latency
        # If t_delay is earlier than the active_latency_start, we must use the anchored phi
        if t_delay < active_latency_start:
            phi_delayed = sample_phi_at_latency_start
        else:
            phi_delayed = float(phi_interp(t_delay))

        # compute derivative using the delayed phi (your RHS function)
        dDdt = plr_rhs_with_latency(D[i - 1], phi_delayed)

        # asymmetric speed scaling
        dt_c = dt / 3.0 if dDdt > 0 else dt

        D[i] = D[i - 1] + dDdt * dt_c

    return D
