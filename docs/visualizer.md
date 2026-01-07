# 📉 PLR Latenzy Visualizer

Desktop application for visualizing synthetic pupillary light reflex (PLR) datasets
and testing different latency estimation methods.

![](../images/visualizer.png)

## Features

### Data Visualization
- Load folders with `.npz` files containing synthetic PLR recordings
- Plot observed and clean pupil diameter (mm) vs time (ms)
- Display frame sample points as scatter dots to see frame capture timing
- Shade LED stimulus duration as a translucent yellow box
- Calculate and display quantization error (sampling rate limitations)

> [!NOTE]
> The clean pupil curve is not a ground truth. Since it is already calculated using the specified sampling rate, it is possible that the ground truth latency does not correspond exactly to the start of the slope.

### Latency Estimation Methods
Select from multiple estimation algorithms:
- **Min derivative** — Finds the time of minimum derivative (steepest descent)
- **Min derivative (smoothed)** — Applies moving average smoothing before finding minimum
- **Threshold crossing** — Detects first crossing of derivative below a percentile threshold
- **Piecewise-linear fit** — Fits two linear segments and finds the breakpoint (slope change)
- **Exponential fit** — Fits a linear segement to the baseline and a exponential segment to the slope
- **Bergamin & Kardon** — Method by Bergmain & Kardon wich uses smoothing and 2nd order derivation

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

### Dataset Metrics
- Statistical pupillometry parameters are calculated and displayed for the data set.
   - Maximal Diameter
   - Minimal Diameter
   - Amplitude
   - Constriction Latency
   - PRT 75 (Time until 75% of the baseline diameter is reached again after the end of the light pulse)
   - Average Constriction Velocity
   - Maximal  Constriction Velocity
   - Dilation Velocity


Usage
-----

0. **Start the application**
```bash
python3 visualizer.py
```

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
   - Use the **Method** dropdown to switch between latency estimation algorithms
   - Plots update instantly to show method-specific visualization
   - Compare errors and visual fit quality

4. **Batch Evaluation**
   - Select desired method from dropdown
   - Click "Batch Evaluate & Export CSV"
   - Choose output location for results CSV
   - View Mean Absolute Error across all files in summary dialog

Notes
-----

- The app is designed for exploration and testing of latency estimation methods
- Synthetic data allows ground truth comparison and controlled parameter variation
- CSV export enables statistical analysis and method comparison across datasets
- Modular design makes it easy to add new estimation methods or visualization features
