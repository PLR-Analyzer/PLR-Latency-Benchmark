# SPDX-FileCopyrightText: 2026 Marcel Schepelmann <schepelmann@chi.uni-hannover.de>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Plot widget for displaying PLR data and analysis results."""

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PySide6 import QtWidgets


class PlotWidget(QtWidgets.QWidget):
    """Widget containing matplotlib figures for PLR visualization."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.fig = Figure(figsize=(8, 6))
        self.ax1, self.ax2 = self.fig.subplots(2, 1, sharex=True)
        self.fig.tight_layout()

        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        # Secondary axis for method-specific plots (created on demand)
        self.ax2_secondary = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

    def plot_data(
        self,
        t_obs,
        D_obs,
        t_clean,
        D_clean,
        stim_time,
        led_duration,
        predicted_latency,
        true_latency,
        method_data,
    ):
        """Plot the PLR data and method-specific visualization.

        Parameters
        ----------
        t : array
            Time points (seconds).
        D_obs : array
            Observed diameter (mm).
        D_clean : array
            Clean diameter (mm).
        stim_time : float
            Time when stimulus starts (seconds).
        led_duration : float
            Duration of LED stimulus (seconds).
        predicted_latency : float
            Predicted latency from method (seconds).
        true_latency : float
            Ground truth latency (seconds).
        method_data : dict
            Method-specific data for plotting (type, data, etc.).
        """
        # First subplot: observed and clean data
        self.ax1.clear()
        self.ax1.plot(t_obs, D_obs, label="Observed", color="C0")
        self.ax1.plot(t_clean, D_clean, label="Clean", color="C1", linestyle="--")
        self.ax1.scatter(t_obs, D_obs, color="C0", s=20, alpha=0.5, zorder=3)
        self.ax1.set_ylabel("Diameter (mm)")
        self.ax1.axvspan(stim_time, stim_time + led_duration, color="yellow", alpha=0.2)

        # Plot true latency line
        if np.isfinite(true_latency):
            self.ax1.axvline(
                true_latency, color="magenta", linestyle="--", label="True"
            )
        self.ax1.legend()
        # self.ax1.set_ylim([0, 9])

        # Second subplot: method-specific visualization
        # Ensure any previous secondary axis is removed before clearing
        if getattr(self, "ax2_secondary", None) is not None:
            try:
                self.ax2_secondary.remove()
            except Exception:
                pass
            self.ax2_secondary = None
        self.ax2.clear()
        self._plot_method_specific(
            t_obs,
            D_obs,
            stim_time,
            led_duration,
            predicted_latency,
            true_latency,
            method_data,
        )

        self.canvas.draw_idle()

    def _plot_method_specific(
        self,
        t,
        D_obs,
        stim_time,
        led_duration,
        predicted_latency,
        true_latency,
        method_data,
    ):
        """Plot method-specific visualization in ax2."""
        method_type = method_data.get("type", "derivative")

        lines2 = labels2 = None

        if method_type == "derivative":
            deriv = method_data.get("data", np.zeros_like(t))
            self.ax2.plot(t, deriv, color="C2", label="dD/dt")
        elif method_type == "threshold":
            self.ax2.plot(t, D_obs, label="Observed", color="C0")
            self.ax2.scatter(t, D_obs, color="C0", s=20, alpha=0.5, zorder=3)
            threshold_val = method_data.get("threshold", np.nan)
            if np.isfinite(threshold_val):
                self.ax2.axhline(
                    threshold_val, color="orange", linestyle=":", label="Threshold"
                )
            self.ax2.set_ylabel("Diameter (mm)")
            # self.ax2.set_ylim([0, 9])
        elif method_type == "piecewise":
            self.ax2.plot(t, D_obs, label="Observed", color="C0")
            self.ax2.scatter(t, D_obs, color="C0", s=20, alpha=0.5, zorder=3)
            self.ax2.autoscale(False)  # disable autoscaling to ignore the lines
            fit_lines = method_data.get("fit_lines")
            if fit_lines is not None:
                self.ax2.plot(
                    t,
                    fit_lines["a1"] * t + fit_lines["b1"],
                    color="green",
                    linewidth=2,
                    label="Fit 1",
                )
                self.ax2.plot(
                    t,
                    fit_lines["a2"] * t + fit_lines["b2"],
                    color="purple",
                    linewidth=2,
                    label="Fit 2",
                )
            self.ax2.set_ylabel("Diameter (mm)")
            # self.ax2.set_ylim([0, 9])
        elif method_type == "exponential":
            self.ax2.plot(t, D_obs, label="Observed", color="C0", linewidth=1.5)
            self.ax2.scatter(t, D_obs, color="C0", s=20, alpha=0.5, zorder=3)
            self.ax2.autoscale(False)

            fit_params = method_data.get("fit_params")
            if fit_params is not None:
                T = fit_params["T"]
                a1 = fit_params["a1"]
                a2 = fit_params["a2"]
                b2 = fit_params["b2"]
                a3 = fit_params["a3"]
                b3 = fit_params["b3"]

                # Unified Bos (1991) model
                dt = t - T
                absdt = np.abs(dt)

                u = 0.5 * (dt - absdt)  # pre-onset (negative branch)
                v = 0.5 * (dt + absdt)  # post-onset (nonnegative branch)

                f_pred = (
                    a1
                    + 0.5 * (a3 * b3 - a2 * b2) * u
                    + a2 * np.exp(-np.clip(b2 * v, -100, 100))
                    - a3 * np.exp(-np.clip(b3 * v, -100, 100))
                )

                # Plot unified fitted curve
                self.ax2.plot(
                    t,
                    f_pred,
                    color="green",
                    linewidth=2,
                    label="Exponential model fit",
                )

            self.ax2.set_ylabel("Diameter (mm)")
            # self.ax2.set_ylim([0, 9])
        elif method_type == "acceleration":
            deriv1 = method_data.get("deriv1")
            deriv2 = method_data.get("deriv2")
            t_interp = method_data.get("t_interp", t)

            # Plot first derivative on the left axis
            if deriv1 is not None:
                self.ax2.plot(
                    t_interp,
                    deriv1,
                    label="1st derivative (dD/dt)",
                    color="C2",
                    linewidth=1.5,
                )

            # Create or reuse secondary y-axis for second derivative
            if getattr(self, "ax2_secondary", None) is None:
                ax2_secondary = self.ax2.twinx()
                self.ax2_secondary = ax2_secondary
            else:
                ax2_secondary = self.ax2_secondary
                ax2_secondary.cla()

            if deriv2 is not None:
                ax2_secondary.plot(
                    t_interp,
                    deriv2,
                    label="2nd derivative (d²D/dt²)",
                    color="C3",
                    linewidth=1.5,
                )

            self.ax2.set_ylabel("dD/dt", color="C2")
            ax2_secondary.set_ylabel("d²D/dt²", color="C3")
            self.ax2.tick_params(axis="y", labelcolor="C2")
            ax2_secondary.tick_params(axis="y", labelcolor="C3")

            # Get legend handles and labels from secondary axis
            lines2, labels2 = ax2_secondary.get_legend_handles_labels()

        # Common elements for second subplot
        self.ax2.set_xlabel("Time (s)")
        self.ax2.axvspan(stim_time, stim_time + led_duration, color="yellow", alpha=0.2)

        # Predicted and true latency lines (shown on the method plot)
        if np.isfinite(predicted_latency):
            self.ax2.axvline(
                predicted_latency, color="red", linestyle="-", label="Predicted"
            )
        if np.isfinite(true_latency):
            self.ax2.axvline(
                true_latency, color="magenta", linestyle="--", label="True"
            )
        if lines2 and labels2:
            # Combine legends from both axes
            lines1, labels1 = self.ax2.get_legend_handles_labels()
            self.ax2.legend(lines2 + lines1, labels2 + labels1)
        else:
            self.ax2.legend()
