PLR Latency Visualizer
=====================

Desktop application for visualizing synthetic pupillary light reflex (PLR) datasets
and testing different latency estimation methods.

Features
--------

### Data Visualization
- Load folders with `.npz` files containing synthetic PLR recordings
- Plot observed and clean pupil diameter (mm) vs time (s)
- Display frame sample points as scatter dots to see frame capture timing
- Shade LED stimulus duration as a translucent yellow box
- Calculate and display quantization error (sampling rate limitations)

### Latency Estimation Methods
Select from multiple estimation algorithms:
- **Min derivative** — Finds the time of minimum derivative (steepest descent)
- **Min derivative (smoothed)** — Applies moving average smoothing before finding minimum
- **Threshold crossing** — Detects first crossing of derivative below a percentile threshold
- **Piecewise-linear fit** — Fits two linear segments and finds the breakpoint (slope change)

Each method shows:
- Prediction error (predicted latency - true latency)
- Quantization error (best-case error due to sampling rate)
- Method-specific visualization in lower plot

### Batch Evaluation
- **Batch Evaluate & Export CSV** button processes all files in the folder
- Exports results to CSV with columns:
  - filename
  - predicted_latency
  - true_latency
  - prediction_error
  - quantization_error
- Displays **Mean Absolute Error (MAE)** across all files for quick performance assessment

### File Navigation
- Load Folder, Previous, Next buttons for easy file browsing
- File list sidebar shows all available `.npz` files
- Automatic plot updates when switching files

Installation
------------

Install dependencies (recommended in a virtualenv):

```bash
python3 -m pip install -r requirements.txt
```
Generate synthetic data:

```bash
python3 data_generation/generate_dataset.py
```
This will generate synthetic data with the default parameters. Default Parameters are choosen from the work of Capó-Aponte et al.

Possible options the data generation:
```bash
options:
  -h, --help            show this help message and exit
  -n NUM_SAMPLES, --num-samples NUM_SAMPLES
                        Number of synthetic recordings to generate (default: 100)
  -o OUTPUT_DIR, --output-dir OUTPUT_DIR
                        Output directory for npz files (default: data)
  -d DURATION, --duration DURATION
                        Duration of each recording in seconds (default: 5.0)
  -f FPS, --fps FPS     Sampling rate in fps (default: 25.0)
  -s STIM_TIME, --stim-time STIM_TIME
                        Time when stimulus turns on in seconds (default: 0.5)
  -l LED_DURATION, --led-duration LED_DURATION
                        Duration of LED pulse in seconds (default: 0.167)
```
For further information about how the simulation model works and data is generated, please refer to:
[data_generation/Generator.md](data_generation/Generator.md)

Run the app:

```bash
python3 main.py
```

Usage
-----

1. **Load Data**
   - Click "Load Folder" and select a directory containing `.npz` files
   - Files must contain:
     - `diameter_observed` — noisy pupil diameter
     - `diameter_clean` — clean model output
     - `true_latency` — ground truth latency (seconds)
     - `fps` — sampling rate (frames per second)
     - `stim_time` — stimulus onset time (seconds)
     - `led_duration` — stimulus duration (seconds)

2. **Explore Individual Recordings**
   - Select files from the left list or use Previous/Next buttons
   - Upper plot shows observed vs clean diameter with frame dots
   - Lower plot shows method-specific analysis
   - Status bar displays file info, prediction error, and quantization error

3. **Compare Methods**
   - Use the Method dropdown to switch between latency estimation algorithms
   - Plots update instantly to show method-specific visualization
   - Compare errors and visual fit quality

4. **Batch Evaluation**
   - Select desired method from dropdown
   - Click "Batch Evaluate & Export CSV"
   - Choose output location for results CSV
   - View Mean Absolute Error across all files in summary dialog

Architecture
------------

The application is organized into modular components:

- **`main.py`** — Entry point, launches the application
- **`visualizer.py`** — Main GUI class, file loading, controls, and batch evaluation
- **`latency_methods.py`** — `LatencyMethods` class with all estimation algorithms
- **`plot_widget.py`** — `PlotWidget` class for matplotlib visualization
- **`simulation.py`** — PLR simulation for generating synthetic data (from `generate_dataset.py`)
- **`generate_dataset.py`** — Script to create synthetic PLR datasets

Adding New Methods
------------------

To add a new latency estimation method:

1. Add a static method to `LatencyMethods` class in `latency_methods.py`:
   ```python
   @staticmethod
   def my_new_method(t, signal, stim_time):
       # Your implementation here
       return latency_time, {"type": "derivative", "data": your_data}
   ```

2. Add method name to `get_available_methods()` list in `latency_methods.py`

3. Add dispatcher case in `compute_by_name()` method

4. If using a different visualization type (not derivative), update `_plot_method_specific()` 
   in `plot_widget.py` to handle the new visualization

Dependencies
------------

- **PySide6** — Qt6 bindings for GUI
- **matplotlib** — Plotting library
- **numpy** — Numerical computations

Notes
-----

- The app is designed for exploration and testing of latency estimation methods
- Synthetic data allows ground truth comparison and controlled parameter variation
- Frame scatter dots help visualize sampling limitations and quantization error
- CSV export enables statistical analysis and method comparison across datasets
- Modular design makes it easy to add new estimation methods or visualization features


License
--------

This Code is published under the GNU GENERAL PUBLIC LICENSE.

Contributing
--------

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

Authors
--------

Marcel Schepelmann


[1] J. E. Capó-Aponte, T. A. Beltran, D. V. Walsh, W. R. Cole, und J. Y. Dumayas, „Validation of Visual Objective Biomarkers for Acute Concussion“, Military Medicine, Bd. 183, Nr. suppl_1, S. 9–17, März 2018, doi: 10.1093/milmed/usx166.
