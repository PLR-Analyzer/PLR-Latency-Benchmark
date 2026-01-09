# SPDX-FileCopyrightText: 2026 Marcel Schepelmann <schepelmann@chi.uni-hannover.de>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Latency estimation methods for PLR analysis."""

import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares, minimize
from scipy.signal import savgol_filter


class LatencyMethods:
    """Collection of latency estimation methods for pupillary light reflex analysis."""

    @staticmethod
    def min_derivative(t, signal, stim_time):
        """Compute latency as the time of minimum derivative after stimulus.

        Parameters
        ----------
        t : array
            Time points (seconds).
        signal : array
            Signal values (e.g., pupil diameter in mm).
        stim_time : float
            Time when stimulus starts (seconds).

        Returns
        -------
        float
            Estimated latency time (seconds).
        dict
            Visualization data with type "derivative" and the derivative array.
        """
        dt = t[1] - t[0]
        deriv = np.gradient(signal, dt)
        mask = t >= stim_time
        if not np.any(mask):
            return np.nan, {"type": "derivative", "data": deriv}
        idx_rel = np.argmin(deriv[mask])
        idx = np.where(mask)[0][0] + idx_rel
        return t[idx], {"type": "derivative", "data": deriv}

    @staticmethod
    def min_derivative_smoothed(t, signal, stim_time, window=5):
        """Compute latency as the time of minimum smoothed derivative.

        Parameters
        ----------
        t : array
            Time points (seconds).
        signal : array
            Signal values (e.g., pupil diameter in mm).
        stim_time : float
            Time when stimulus starts (seconds).
        window : int
            Window size for moving average smoothing (default: 5).

        Returns
        -------
        float
            Estimated latency time (seconds).
        dict
            Visualization data with type "derivative" and the smoothed derivative.
        """
        dt = t[1] - t[0]
        deriv = np.gradient(signal, dt)
        kernel = np.ones(window) / window
        deriv_smooth = np.convolve(deriv, kernel, mode="same")
        mask = t >= stim_time
        if not np.any(mask):
            return np.nan, {"type": "derivative", "data": deriv_smooth}
        idx_rel = np.argmin(deriv_smooth[mask])
        idx = np.where(mask)[0][0] + idx_rel
        return t[idx], {"type": "derivative", "data": deriv_smooth}

    @staticmethod
    def threshold_crossing(t, signal, stim_time, threshold=0.5):
        """Compute latency as the first crossing below a threshold derivative.

        Parameters
        ----------
        t : array
            Time points (seconds).
        signal : array
            Signal values (e.g., pupil diameter in mm).
        stim_time : float
            Time when stimulus starts (seconds).
        threshold : float
            Percentile threshold (0-1) for derivative crossing (default: 0.5).

        Returns
        -------
        float
            Estimated latency time (seconds).
        dict
            Visualization data with type "threshold" and the threshold value.
        """
        before_mask = t < stim_time
        after_mask = t >= stim_time

        if not np.any(before_mask) or not np.any(after_mask):
            return np.nan, {"type": "threshold", "threshold": np.nan}

        baseline_mean = np.mean(signal[before_mask])
        baseline_std = np.std(signal[before_mask])

        threshold_val = baseline_mean - 3 * baseline_std
        crossing_idx = np.where(signal[after_mask] < threshold_val)[0]
        if len(crossing_idx) == 0:
            return np.nan, {"type": "threshold", "threshold": threshold_val}
        idx = np.where(after_mask)[0][0] + crossing_idx[0]

        return t[idx], {"type": "threshold", "threshold": threshold_val}

    @staticmethod
    def piecewise_linear(t, signal, stim_time):
        """Fit two linear segments: before onset and after onset, and compute intersection.

        Parameters
        ----------
        t : array
            Time points (seconds).
        signal : array
            Signal values (e.g., pupil diameter in mm).
        stim_time : float
            Time when stimulus starts (seconds).

        Returns
        -------
        float
            Estimated latency time (seconds) at the breakpoint.
        dict
            Visualization data with type "piecewise" and fitted line coefficients.
        """

        latency_guess = stim_time + 0.2  # rough mean from literatur
        # Calcualte relevant fitting windows
        mask = (t >= 0) & (t <= t[np.argmin(signal)])
        tt = t[mask]
        yy = signal[mask]

        # model: y = a1*t + b1  for t < t_change
        #        y = a2*t + b2  for t >= t_change
        # we parameterize with t_change, a1,b1,a2,b2 and enforce continuity at t_change
        def residuals(params):
            t_change, a1, b1, a2 = params  # b2 = (a1 - a2)*t_change + b1 for continuity
            b2 = (a1 - a2) * t_change + b1
            pred = np.where(tt < t_change, a1 * tt + b1, a2 * tt + b2)
            return pred - yy

        # initial guesses
        a_pre = (yy[: max(3, len(yy) // 10)].mean() - yy[:1].mean()) / max(
            0.001, (tt[: max(3, len(tt) // 10)].mean() - tt[0])
        )
        a_post = (yy[-1] - yy[int(len(yy) * 0.7)]) / max(
            0.001, (tt[-1] - tt[int(len(tt) * 0.7)])
        )
        p0 = [latency_guess, a_pre, yy[0], a_post]
        bounds = ([tt[0], -np.inf, -np.inf, -np.inf], [tt[-1], np.inf, np.inf, np.inf])
        res = least_squares(residuals, p0, bounds=bounds)
        t_change, a1, b1, a2 = res.x
        b2 = (a1 - a2) * t_change + b1
        refined = t_change

        fit_lines = {
            "t_change": t_change,
            "a1": a1,
            "b1": b1,
            "a2": a2,
            "b2": b2,
            "residuals": res.fun,
        }

        return refined, {"type": "piecewise", "fit_lines": fit_lines}

    @staticmethod
    def exponential_fit(t, signal, stim_time):
        """
        Bos (1991) ssecond-order mathematical model of the pupil constriction.
        """

        t = np.asarray(t, float)
        y = np.asarray(signal, float)

        # ---- Fitting window: everything before the constriction minimum ----
        end = np.argmin(y)
        mask = np.arange(len(t)) <= end
        tt = t[mask]
        yy = y[mask]

        if len(tt) < 10:
            return np.nan, {"type": "exponential", "fit_params": {}}

        # ---- Simple initial guesses ----
        T0 = stim_time  # Literature suggests a latency of ~200ms
        a1_0 = yy[np.searchsorted(tt, stim_time)]  # value at stim time
        a2_0 = a1_0 - yy[-1]
        a3_0 = 0.3 * a2_0
        b2_0 = 0.01
        b3_0 = 0.002

        p0 = np.array([T0, a1_0, a2_0, b2_0, a3_0, b3_0], float)

        # ---- Bounds ----
        lb = [stim_time, a1_0 - 5, -5 * a2_0, 0.0001, -5 * a3_0, 0.0001]
        ub = [tt[-1] - 1.0, a1_0 + 5, 5 * a2_0, 0.1, 5 * a3_0, 0.02]

        # ---- Bos model with constraints from section 4.2----
        def model(p, ti):
            T, a1, a2, b2, a3, b3 = p

            a0 = a1 + a2 - a3
            b0 = -a2 * b2 + a3 * b3

            dt = ti - T
            before = a0 + b0 * dt
            after = a1 + a2 * np.exp(-b2 * dt) - a3 * np.exp(-b3 * dt)
            return np.where(ti < T, before, after)

        def residuals(p):
            return model(p, tt) - yy

        res = least_squares(residuals, p0, bounds=(lb, ub), max_nfev=4000)
        T, a1, a2, b2, a3, b3 = res.x

        # ---- Derived parameters ----
        a0 = a1 + a2 - a3
        b0 = -a2 * b2 + a3 * b3

        return float(T), {
            "type": "exponential",
            "fit_params": {
                "T": float(T),
                "a0": float(a0),
                "b0": float(b0),
                "a1": float(a1),
                "a2": float(a2),
                "b2": float(b2),
                "a3": float(a3),
                "b3": float(b3),
            },
        }

    @staticmethod
    def max_negative_acceleration(t, signal, stim_time):
        """Compute latency as the time of maximum negative acceleration after stimulus.

        Uses Savitzky-Golay filtering, cubic spline interpolation, and Gaussian
        filtering to smooth data and estimate latency from the second derivative.

        Parameters
        ----------
        t : array
            Time points (seconds).
        signal : array
            Signal values (e.g., pupil diameter in mm).
        stim_time : float
            Time when stimulus starts (seconds).

        Returns
        -------
        float
            Estimated latency time (seconds).
        dict
            Visualization data with type "acceleration" containing first and second derivatives.
        """
        t = np.asarray(t, float)
        signal = np.asarray(signal, float)

        # Step 1: Apply Savitzky-Golay filter (5-point, 2nd order polynomial)
        signal_smoothed = savgol_filter(signal, window_length=5, polyorder=2)

        # Step 2: Cubic spline interpolation for higher resolution (300 Hz)
        f_cubic = interp1d(t, signal_smoothed, kind="cubic", fill_value="extrapolate")
        t_interp = np.linspace(t[0], t[-1], int((t[-1] // 1000) * 300))
        signal_interp = f_cubic(t_interp)
        dt_interp = t_interp[1] - t_interp[0]

        # Step 3: Calculate first derivative
        deriv1 = np.gradient(signal_interp, dt_interp)

        # Step 4: Apply Gaussian filter to first derivative (sigma=25)
        deriv1_filtered = gaussian_filter1d(deriv1, sigma=25)

        # Step 5: Calculate second derivative from filtered first derivative
        deriv2 = np.gradient(deriv1_filtered, dt_interp)

        # Step 6: Find maximum negative acceleration after stimulus
        mask = t_interp >= stim_time
        if not np.any(mask):
            return np.nan, {
                "type": "acceleration",
                "deriv1": deriv1_filtered,
                "deriv2": deriv2,
            }

        # Find the most negative (minimum) second derivative
        idx_rel = np.argmin(deriv2[mask])
        idx = np.where(mask)[0][0] + idx_rel

        return t_interp[idx], {
            "type": "acceleration",
            "deriv1": deriv1_filtered,
            "deriv2": deriv2,
            "t_interp": t_interp,
        }

    @staticmethod
    def get_available_methods():
        """Return list of available method names.

        Returns
        -------
        list
            List of method names available in this class.
        """
        return [
            "Min derivative",
            "Min derivative (smoothed)",
            "Threshold crossing",
            "Piecewise-linear fit",
            "Exponential fit",
            "Bergamin & Kardon",
        ]

    @staticmethod
    def compute_by_name(method_name, t, signal, stim_time, led_duration=None, fps=None):
        """Dispatch to appropriate latency computation method by name.

        Parameters
        ----------
        method_name : str
            Name of the method to use.
        t : array
            Time points (seconds).
        signal : array
            Signal values (e.g., pupil diameter in mm).
        stim_time : float
            Time when stimulus starts (seconds).
        led_duration : float, optional
            Duration of LED stimulus (required for model_fit method).
        fps : float, optional
            Sampling rate (required for model_fit method).

        Returns
        -------
        float
            Estimated latency time (seconds).
        dict
            Visualization data specific to the method.
        """
        if method_name == "Min derivative":
            return LatencyMethods.min_derivative(t, signal, stim_time)
        elif method_name == "Min derivative (smoothed)":
            return LatencyMethods.min_derivative_smoothed(t, signal, stim_time)
        elif method_name == "Threshold crossing":
            return LatencyMethods.threshold_crossing(t, signal, stim_time)
        elif method_name == "Piecewise-linear fit":
            return LatencyMethods.piecewise_linear(t, signal, stim_time)
        elif method_name == "Exponential fit":
            return LatencyMethods.exponential_fit(t, signal, stim_time)
        elif method_name == "Bergamin & Kardon":
            return LatencyMethods.max_negative_acceleration(t, signal, stim_time)
        else:
            dt = t[1] - t[0]
            return np.nan, {
                "type": "derivative",
                "data": np.gradient(signal, dt),
            }
