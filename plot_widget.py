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

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

    def plot_data(
        self,
        t,
        D_obs,
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
        self.ax1.plot(t, D_obs, label="Observed", color="C0")
        self.ax1.plot(t, D_clean, label="Clean", color="C1", linestyle="--")
        self.ax1.scatter(t, D_obs, color="C0", s=20, alpha=0.5, zorder=3)
        self.ax1.set_ylabel("Diameter (mm)")
        self.ax1.axvspan(stim_time, stim_time + led_duration, color="yellow", alpha=0.2)

        # Add true latency line only (predicted line removed to reduce clutter)
        if np.isfinite(true_latency):
            self.ax1.axvline(
                true_latency, color="magenta", linestyle="--", label="True"
            )
        self.ax1.legend()

        # Second subplot: method-specific visualization
        self.ax2.clear()
        self._plot_method_specific(
            t,
            D_obs,
            D_clean,
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
        D_clean,
        stim_time,
        led_duration,
        predicted_latency,
        true_latency,
        method_data,
    ):
        """Plot method-specific visualization in ax2."""
        method_type = method_data.get("type", "derivative")

        if method_type == "derivative":
            deriv = method_data.get("data", np.zeros_like(t))
            self.ax2.plot(t, deriv, color="C2")
            self.ax2.set_ylabel("dD/dt")
        elif method_type == "threshold":
            self.ax2.plot(t, D_obs, label="Observed", color="C0")
            self.ax2.scatter(t, D_obs, color="C0", s=20, alpha=0.5, zorder=3)
            threshold_val = method_data.get("threshold", np.nan)
            if np.isfinite(threshold_val):
                self.ax2.axhline(
                    threshold_val, color="orange", linestyle=":", label="Threshold"
                )
            self.ax2.set_ylabel("Diameter (mm)")
            self.ax2.legend()
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
            self.ax2.legend()

        # Common elements for second subplot
        self.ax2.set_xlabel("Time (s)")
        self.ax2.axvspan(stim_time, stim_time + led_duration, color="yellow", alpha=0.2)
        if np.isfinite(predicted_latency):
            self.ax2.axvline(
                predicted_latency, color="red", linestyle="-", label="Predicted"
            )
        if np.isfinite(true_latency):
            self.ax2.axvline(
                true_latency, color="magenta", linestyle="--", label="True"
            )
