import csv
import os
from pathlib import Path

import numpy as np
from PySide6 import QtWidgets
from PySide6.QtWidgets import QFileDialog, QMessageBox

from latency_methods import LatencyMethods
from metrics import compute_metrics_from_file, summarize_metrics
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
            "constr_latency",
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
        """Wrapper: delegate to metrics.compute_metrics_from_file."""
        return compute_metrics_from_file(filepath)

    def _update_metrics_display(self):
        """Recompute and update metrics display for loaded folder by using
        the summarization helper in `metrics.py`.
        """
        if not self.files:
            for label in self.metric_labels.values():
                label.setText("N/A")
            return

        summary = summarize_metrics(self.files)

        label_texts = {
            "max_diameter": "Max diameter (mm)",
            "min_diameter": "Min diameter (mm)",
            "amplitude": "Amplitude (%)",
            "constr_latency": "Constriction Latency (ms)",
            "t75": "75% Recovery (ms)",
            "avg_constr_vel": "Avg Constr Vel (mm/s)",
            "max_constr_vel": "Max Constr Vel (mm/s)",
            "dil_vel": "Dilation Vel (mm/s)",
        }

        for key, label in self.metric_labels.items():
            mean_val, std_val, count = summary.get(key, (np.nan, np.nan, 0))
            if count > 0 and np.isfinite(mean_val):
                # Values from metrics.py are already in correct units (ms for time)
                label.setText(
                    f"{label_texts.get(key, key)}: {mean_val:.1f} ± {std_val:.1f}"
                )
            else:
                label.setText(f"{label_texts.get(key, key)}: N/A")

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
            stim_time = float(npz.get("stim_time", 500.0))  # in ms
            led_duration = float(npz.get("led_duration", 167.0))  # in ms
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
            led_duration = float(npz.get("led_duration", 167.0))  # in ms
            stim_time = float(npz.get("stim_time", 500.0))  # in ms
        except Exception as e:
            return {
                "filename": filepath.name,
                "error": f"Failed to load: {e}",
            }

        n = len(D_obs)
        dt = 1000.0 / fps  # dt in ms
        t = np.linspace(0, dt * (n - 1), n)  # t in ms

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
        stim_time = self.data["stim_time"]  # in ms
        led_duration = self.data["led_duration"]  # in ms
        true_latency = self.data["true_latency"]  # in ms

        n = len(D_obs)
        dt = 1000.0 / fps  # dt in ms
        t = np.linspace(0, dt * (n - 1), n)  # t in ms
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
            err_text = f"Error = {err:.1f} ms (predicted - true)"

        quant_err = self._compute_quantization_error(t, true_latency)
        quant_text = ""
        if np.isfinite(quant_err):
            quant_text = f"  |  Min quantization error = {quant_err:.1f} ms"

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
