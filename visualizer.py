import csv
import os
from pathlib import Path

import numpy as np
from PySide6 import QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFileDialog, QMessageBox

from latency_methods import LatencyMethods
from plot_widget import PlotWidget


class LatencyVisualizer(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("PLR Latency Visualizer")

        self.folder = None
        self.files = []
        self.current_index = -1
        self.data = None

        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        # Top controls
        ctrl_layout = QtWidgets.QHBoxLayout()
        self.load_btn = QtWidgets.QPushButton("Load Folder")
        self.load_btn.clicked.connect(self.on_load_folder)
        ctrl_layout.addWidget(self.load_btn)

        self.prev_btn = QtWidgets.QPushButton("Previous")
        self.prev_btn.clicked.connect(self.on_prev)
        ctrl_layout.addWidget(self.prev_btn)

        self.next_btn = QtWidgets.QPushButton("Next")
        self.next_btn.clicked.connect(self.on_next)
        ctrl_layout.addWidget(self.next_btn)

        ctrl_layout.addStretch()

        ctrl_layout.addWidget(QtWidgets.QLabel("Method:"))
        self.method_combo = QtWidgets.QComboBox()
        self.method_combo.addItems(LatencyMethods.get_available_methods())
        self.method_combo.currentIndexChanged.connect(self.on_method_changed)
        ctrl_layout.addWidget(self.method_combo)

        self.batch_btn = QtWidgets.QPushButton("Batch Evaluate & Export CSV")
        self.batch_btn.clicked.connect(self.on_batch_evaluate)
        ctrl_layout.addWidget(self.batch_btn)

        layout.addLayout(ctrl_layout)

        # File list, canvas, and metrics sidebar
        mid_layout = QtWidgets.QHBoxLayout()

        self.list_widget = QtWidgets.QListWidget()
        self.list_widget.currentRowChanged.connect(self.on_select_file)
        self.list_widget.setMaximumWidth(300)
        mid_layout.addWidget(self.list_widget)

        # Create plot widget
        self.plot_widget = PlotWidget(self)
        mid_layout.addWidget(self.plot_widget, 3)

        # Create metrics sidebar (resizable)
        self.metrics_panel = self._create_metrics_panel()
        mid_layout.addWidget(self.metrics_panel, 1)

        layout.addLayout(mid_layout)

        # Bottom: info label
        self.info_label = QtWidgets.QLabel("")
        layout.addWidget(self.info_label)

    def _create_metrics_panel(self):
        """Create a scrollable panel for dataset metrics."""
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)

        title = QtWidgets.QLabel("Dataset Metrics")
        title.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.addWidget(title)

        # Scrollable area for metrics
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QtWidgets.QWidget()
        self.metrics_layout = QtWidgets.QVBoxLayout(scroll_widget)

        # Dictionary to store metric labels for updates
        self.metric_labels = {}
        metric_keys = [
            "max_diameter",
            "min_diameter",
            "amplitude",
            "t75",
            "avg_constr_vel",
            "max_constr_vel",
            "dil_vel",
        ]
        for key in metric_keys:
            label = QtWidgets.QLabel(f"{key}: N/A")
            label.setStyleSheet("font-size: 10px;")
            label.setWordWrap(True)
            self.metrics_layout.addWidget(label)
            self.metric_labels[key] = label

        self.metrics_layout.addStretch()
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)

        return panel

    def _compute_metrics_from_file(self, filepath):
        """Load a file and compute PLR metrics.

        Returns a dict with metrics or None if loading fails.
        """
        try:
            npz = np.load(str(filepath), allow_pickle=True)
            D_obs = np.asarray(npz["diameter_observed"])
            fps = float(npz.get("fps", 30.0))
            stim_time = float(npz.get("stim_time", 0.5))
            led_duration = float(npz.get("led_duration", 0.167))
        except Exception as e:
            print(f"Failed to load metrics from {filepath}: {e}")
            return None

        dt = 1.0 / fps
        n = len(D_obs)
        t = np.linspace(0, dt * (n - 1), n)

        # Compute derivative and constriction region
        deriv = np.gradient(D_obs, dt)

        # Baseline (before stimulus)
        baseline_mask = t < stim_time
        D_baseline_mean = np.mean(D_obs[baseline_mask])
        D_max = D_baseline_mean  # max diameter (baseline)

        # Response region (after stimulus + latency estimate)
        response_start_idx = int((stim_time + 0.2) * fps)
        response_start_idx = min(response_start_idx, len(D_obs) - 1)
        response_region = D_obs[response_start_idx:]
        D_min = np.min(response_region) if len(response_region) > 0 else np.nan

        # Amplitude (percent constriction)
        amplitude = ((D_max - D_min) / D_max * 100) if D_max > 0 else np.nan

        # 75% recovery time
        if np.isfinite(D_min) and D_max > D_min:
            D75 = D_min + 0.75 * (D_max - D_min)
            min_idx = response_start_idx + np.argmin(response_region)
            idx_recovery = np.where(D_obs[min_idx:] >= D75)[0]
            t75 = (idx_recovery[0] * dt) if len(idx_recovery) > 0 else np.nan
        else:
            t75 = np.nan

        # Constriction and dilation velocities
        constr_mask = (t >= stim_time) & (t <= stim_time + led_duration + 1.0)
        constr_deriv = deriv[constr_mask]
        constr_neg = constr_deriv[constr_deriv < 0]
        avg_constr_vel = np.mean(constr_neg) if len(constr_neg) > 0 else np.nan
        max_constr_vel = np.min(constr_neg) if len(constr_neg) > 0 else np.nan

        # Dilation velocity (after minimum)
        if np.isfinite(D_min) and response_start_idx + 1 < len(D_obs):
            min_idx = response_start_idx + np.argmin(response_region)
            dil_window_end = min(min_idx + int(0.5 * fps), len(deriv))
            dil_deriv = deriv[min_idx:dil_window_end]
            dil_deriv_pos = dil_deriv[dil_deriv > 0]
            dil_vel = np.mean(dil_deriv_pos) if len(dil_deriv_pos) > 0 else np.nan
        else:
            dil_vel = np.nan

        return {
            "D_max": D_max,
            "D_min": D_min,
            "amplitude": amplitude,
            "t75": t75,
            "avg_constr_vel": avg_constr_vel,
            "max_constr_vel": max_constr_vel,
            "dil_vel": dil_vel,
        }

    def _update_metrics_display(self):
        """Recompute and update metrics display for loaded folder."""
        if not self.files:
            # Clear metrics
            for label in self.metric_labels.values():
                label.setText("N/A")
            return

        # Collect metrics from all files
        all_metrics = {
            "D_max": [],
            "D_min": [],
            "amplitude": [],
            "t75": [],
            "avg_constr_vel": [],
            "max_constr_vel": [],
            "dil_vel": [],
        }

        for filepath in self.files:
            metrics = self._compute_metrics_from_file(filepath)
            if metrics is not None:
                for key in all_metrics:
                    if np.isfinite(metrics[key]):
                        all_metrics[key].append(metrics[key])

        # Update labels with mean (std) format
        metric_display = {
            "max_diameter": (all_metrics["D_max"], "Max diameter (mm)"),
            "min_diameter": (all_metrics["D_min"], "Min diameter (mm)"),
            "amplitude": (all_metrics["amplitude"], "Amplitude (%)"),
            "t75": (all_metrics["t75"], "75% Recovery (s)"),
            "avg_constr_vel": (
                all_metrics["avg_constr_vel"],
                "Avg Constr Vel (mm/s)",
            ),
            "max_constr_vel": (
                all_metrics["max_constr_vel"],
                "Max Constr Vel (mm/s)",
            ),
            "dil_vel": (all_metrics["dil_vel"], "Dilation Vel (mm/s)"),
        }

        for key, (data_list, label_text) in metric_display.items():
            if len(data_list) > 0:
                mean_val = np.mean(data_list)
                std_val = np.std(data_list)
                self.metric_labels[key].setText(
                    f"{label_text}: {mean_val:.4f} ({std_val:.4f})"
                )
            else:
                self.metric_labels[key].setText(f"{label_text}: N/A")

    def on_load_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select folder with .npz files", os.getcwd()
        )
        if not folder:
            return
        self.folder = Path(folder)
        self.files = sorted([p for p in self.folder.iterdir() if p.suffix == ".npz"])
        self.list_widget.clear()
        for p in self.files:
            self.list_widget.addItem(p.name)
        if self.files:
            self.list_widget.setCurrentRow(0)
        # Update metrics display
        self._update_metrics_display()

    def on_select_file(self, row):
        if row < 0 or row >= len(self.files):
            return
        self.current_index = row
        filepath = self.files[row]
        try:
            npz = np.load(str(filepath), allow_pickle=True)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load {filepath.name}: {e}")
            return

        # Expect keys: diameter_observed, diameter_clean, true_latency, fps, stim_time, led_duration
        try:
            D_obs = npz["diameter_observed"]
            D_clean = npz["diameter_clean"]
            true_latency = float(npz.get("true_latency", np.nan))
            fps = float(npz.get("fps", 30.0))
            stim_time = float(npz.get("stim_time", 0.5))
            led_duration = float(npz.get("led_duration", 0.167))
        except Exception as e:
            QMessageBox.critical(self, "Error", f"NPZ missing expected keys: {e}")
            return

        self.data = dict(
            D_obs=np.asarray(D_obs),
            D_clean=np.asarray(D_clean),
            true_latency=true_latency,
            fps=fps,
            stim_time=stim_time,
            led_duration=led_duration,
            filename=filepath.name,
        )

        self.update_plots()

    def on_prev(self):
        if not self.files:
            return
        idx = max(0, self.current_index - 1)
        self.list_widget.setCurrentRow(idx)

    def on_next(self):
        if not self.files:
            return
        idx = min(len(self.files) - 1, self.current_index + 1)
        self.list_widget.setCurrentRow(idx)

    def on_method_changed(self, idx):
        self.update_plots()

    def _compute_quantization_error(self, t, true_latency):
        """Compute minimum error introduced by sampling at frame rate.

        Finds the closest frame time to the true latency and returns the error.
        """
        if not np.isfinite(true_latency):
            return np.nan
        # Find closest sample time to true_latency
        closest_idx = np.argmin(np.abs(t - true_latency))
        return t[closest_idx] - true_latency

    def _process_file_for_batch(self, filepath, method_name):
        """Load a file and compute latency using the specified method.

        Returns a dict with filename, predicted_latency, true_latency, error, and quantization_error.
        """
        try:
            npz = np.load(str(filepath), allow_pickle=True)
            D_obs = np.asarray(npz["diameter_observed"])
            true_latency = float(npz.get("true_latency", np.nan))
            fps = float(npz.get("fps", 30.0))
            led_duration = float(npz.get("led_duration", 1.0))
            stim_time = float(npz.get("stim_time", 0.5))
        except Exception as e:
            return {
                "filename": filepath.name,
                "error": f"Failed to load: {e}",
            }

        n = len(D_obs)
        dt = 1.0 / fps
        t = np.linspace(0, dt * (n - 1), n)

        predicted_latency, _ = LatencyMethods.compute_by_name(
            method_name, t, D_obs, stim_time, led_duration, fps
        )

        prediction_error = np.nan
        if np.isfinite(predicted_latency) and np.isfinite(true_latency):
            prediction_error = predicted_latency - true_latency

        quant_error = self._compute_quantization_error(t, true_latency)

        return {
            "filename": filepath.name,
            "predicted_latency": predicted_latency,
            "true_latency": true_latency,
            "prediction_error": prediction_error,
            "quantization_error": quant_error,
        }

    def on_batch_evaluate(self):
        """Batch evaluate all files and export results to CSV."""
        if not self.files:
            QMessageBox.warning(self, "Warning", "No folder loaded.")
            return

        method_name = self.method_combo.currentText()
        results = []

        for filepath in self.files:
            result = self._process_file_for_batch(filepath, method_name)
            results.append(result)

        # Ask user where to save CSV
        csv_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save batch results as CSV",
            str(self.folder / "results.csv"),
            "CSV files (*.csv)",
        )
        if not csv_path:
            return

        # Write CSV
        try:
            with open(csv_path, "w", newline="") as f:
                fieldnames = [
                    "filename",
                    "predicted_latency",
                    "true_latency",
                    "prediction_error",
                    "quantization_error",
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)

            # Calculate mean absolute error
            errors = [
                r["prediction_error"]
                for r in results
                if np.isfinite(r.get("prediction_error", np.nan))
            ]
            if errors:
                mae = np.mean(np.abs(errors))
                message = (
                    f"Batch evaluation complete.\n\n"
                    f"Results saved to:\n{csv_path}\n\n"
                    f"Mean Absolute Error: {mae:.4f} s\n"
                    f"Files processed: {len(results)}"
                )
            else:
                message = (
                    f"Batch evaluation complete.\n\n"
                    f"Results saved to:\n{csv_path}\n\n"
                    f"No valid errors to compute MAE."
                )

            QMessageBox.information(self, "Success", message)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to write CSV: {e}")

    def update_plots(self):
        if self.data is None:
            return

        D_obs = self.data["D_obs"]
        D_clean = self.data["D_clean"]
        fps = self.data["fps"]
        stim_time = self.data["stim_time"]
        led_duration = self.data["led_duration"]
        true_latency = self.data["true_latency"]

        n = len(D_obs)
        dt = 1.0 / fps
        t = np.linspace(0, dt * (n - 1), n)

        # Compute latency according to selected method
        method = self.method_combo.currentText()
        predicted_latency, method_data = LatencyMethods.compute_by_name(
            method, t, D_obs, stim_time, led_duration, fps
        )

        # Update plots via the plot widget
        self.plot_widget.plot_data(
            t,
            D_obs,
            D_clean,
            stim_time,
            led_duration,
            predicted_latency,
            true_latency,
            method_data,
        )

        # error display
        err_text = "N/A"
        if np.isfinite(predicted_latency) and np.isfinite(true_latency):
            err = predicted_latency - true_latency
            err_text = f"Error = {err:.3f} s (predicted - true)"

        quant_err = self._compute_quantization_error(t, true_latency)
        quant_text = ""
        if np.isfinite(quant_err):
            quant_text = f"  |  Min quantization error = {quant_err:.3f} s"

        self.info_label.setText(
            f"File: {self.data['filename']}    FPS: {fps}    {err_text}{quant_text}"
        )


def run_app():
    import sys

    app = QtWidgets.QApplication(sys.argv)
    w = LatencyVisualizer()
    w.resize(1400, 700)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run_app()
