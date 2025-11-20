"""Latency estimation methods for PLR analysis."""

import numpy as np
from scipy.optimize import least_squares, minimize


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

        print(baseline_mean, baseline_std)

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
    def _flux_from_diameter(D, phi_ref=1.0):
        """Compute retinal flux φ that gives steady-state diameter D."""
        arg = np.clip((D - 4.9) / 3.0, -0.9999, 0.9999)
        rhs = 5.2 - 2.3026 * np.arctanh(arg)
        phi_ratio = np.exp(rhs / 0.45)
        return phi_ref * phi_ratio

    @staticmethod
    def _simulate_dynamics(phi_arr, dt, D0, delay_samples, dMdD, phi_ref):
        """Simulate diameter time course for a given `phi_arr`."""
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

    @staticmethod
    def model_fit(t, signal, stim_time, led_duration, fps):
        """Fit PLR model to observed data and extract latency from parameters.

        Fits the parametric model from the simulation to find:
        - Baseline diameter (D_max)
        - Stimulus diameter (D_min)
        - Latency (tau_latency)
        - Stimulus strength (phi_stim)

        Parameters
        ----------
        t : array
            Time points (seconds).
        signal : array
            Observed pupil diameter (mm).
        stim_time : float
            Time when stimulus starts (seconds).
        led_duration : float
            Duration of stimulus (seconds).
        fps : float
            Sampling rate (frames per second).

        Returns
        -------
        float
            Estimated latency time (seconds).
        dict
            Visualization data with type "model_fit" and fitted model output.
        """
        dt = 1.0 / fps

        # Initial guesses from data
        D_max_guess = signal[: int(stim_time * fps)].mean()
        D_min_guess = signal[int(stim_time * fps) :].min()
        tau_latency_guess = 0.2

        def objective(params):
            D_max, D_min, tau_latency, phi_stim_factor = params

            # Constrain parameters
            # D_max = np.clip(D_max, 3.0, 7.0)
            # D_min = np.clip(D_min, 2.5, D_max - 0.5)
            # tau_latency = np.clip(tau_latency, 0.05, 1.0)
            # phi_stim_factor = np.clip(phi_stim_factor, 0.5, 10.0)

            # Compute baseline flux
            phi_baseline = LatencyMethods._flux_from_diameter(D_max, phi_ref=1.0)
            phi_stim_ss = LatencyMethods._flux_from_diameter(D_min, phi_ref=1.0)

            # Determine stimulus flux
            phi_stim = phi_stim_ss * phi_stim_factor

            # Build stimulus array
            phi_arr = np.full_like(t, phi_baseline)
            on_mask = (t >= stim_time) & (t < stim_time + led_duration)
            phi_arr[on_mask] = phi_stim

            # Simulate
            delay_samples = int(np.ceil(tau_latency / dt))
            try:
                D_sim = LatencyMethods._simulate_dynamics(
                    phi_arr, dt, D_max, delay_samples, dMdD=1.0, phi_ref=1.0
                )
            except:
                return np.full_like(signal, 1e10)

            # Error: focus on response region
            response_start = int((stim_time + tau_latency) * fps)
            response_end = int((stim_time + tau_latency + 2.0) * fps)
            response_end = min(response_end, len(signal))

            if response_start >= len(signal):
                return np.full_like(signal, 1e10)

            error = (
                signal[response_start:response_end] - D_sim[response_start:response_end]
            )
            return error

        # Optimize
        p0 = [D_max_guess, D_min_guess, tau_latency_guess, 2.0]
        try:
            result = least_squares(
                objective,
                p0,
                bounds=(
                    [3.0, 2.5, 0.05, 0.5],
                    [7.0, D_max_guess - 0.1, 1.0, float("inf")],
                ),
                max_nfev=500,
            )
            D_max_fit, D_min_fit, tau_latency_fit, phi_stim_factor = result.x
        except:
            # Fall back to initial guess
            D_max_fit, D_min_fit, tau_latency_fit, phi_stim_factor = p0

        # Generate fitted model
        phi_baseline = LatencyMethods._flux_from_diameter(D_max_fit, phi_ref=1.0)
        phi_stim_ss = LatencyMethods._flux_from_diameter(D_min_fit, phi_ref=1.0)
        phi_stim = phi_stim_ss * phi_stim_factor

        phi_arr = np.full_like(t, phi_baseline)
        on_mask = (t >= stim_time) & (t < stim_time + led_duration)
        phi_arr[on_mask] = phi_stim

        delay_samples = int(np.ceil(tau_latency_fit / dt))
        D_fitted = LatencyMethods._simulate_dynamics(
            phi_arr, dt, D_max_fit, delay_samples, dMdD=1.0, phi_ref=1.0
        )

        latency = stim_time + tau_latency_fit

        fit_data = {
            "D_fitted": D_fitted,
            "D_max": D_max_fit,
            "D_min": D_min_fit,
            "tau_latency": tau_latency_fit,
            "phi_stim_factor": phi_stim_factor,
        }

        return latency, {"type": "model_fit", "fit_data": fit_data}

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
            "Fit to simulation model",
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
        elif method_name == "Fit to simulation model":
            if led_duration is None or fps is None:
                dt = t[1] - t[0]
                return np.nan, {"type": "derivative", "data": np.gradient(signal, dt)}
            return LatencyMethods.model_fit(t, signal, stim_time, led_duration, fps)
        else:
            dt = t[1] - t[0]
            return np.nan, {
                "type": "derivative",
                "data": np.gradient(signal, dt),
            }
