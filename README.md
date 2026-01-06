PLR Latency Benchmark
=====================

Desktop application for visualizing synthetic pupillary light reflex (PLR) datasets
and testing different latency estimation methods.

Features
--------

- Data Generation
- Data Visualization
- Estimation Algorithm Evaluation

Installation
------------

Install dependencies (recommended in a virtualenv):

```bash
python3 -m pip install -r requirements.txt
```

Usage
-----

### Data generation
Generate synthetic data:

```bash
python3 -m data_generation.generate_dataset
```
This will generate a synthetic dataset with the default parameters. Default parameters can be changed in [stat_values.py](data_generation/stat_values.py)

Possible options the data generation:
```bash
options:
  -h, --help            show this help message and exit
  -n NUM_SAMPLES, --num-samples NUM_SAMPLES
                        Number of synthetic recordings to generate (default: 100)
  -o OUTPUT_DIR, --output-dir OUTPUT_DIR
                        Output directory for npz files (default: data)
  -d DURATION, --duration DURATION
                        Duration of each recording in milliseconds (default: 5000)
  -f FPS, --fps FPS     Sampling rate in fps (default: 30.0)
  -s STIM_TIME, --stim-time STIM_TIME
                        Time when stimulus turns on in milliseconds (default: 500)
  -l LED_DURATION, --led-duration LED_DURATION
                        Duration of LED pulse in milliseconds (default: 200)
  --noise NOISE         Standard deviation of gaussian noise (default: 0.03)
```
For further information about how the simulation model works and data is generated, please refer to:
[data_generation/Generator.md](data_generation/Generator.md)

### Data Visualization
A short desktop visualizer lets you explore individual recordings and
compare latency estimation methods interactively: load a folder of
.npz files, view observed vs clean pupil traces with frame sample
markers, inspect stimulus timing, and run the available estimation
algorithms. For detailed usage, features and batch-evaluation options,
see the visualizer documentation: [visualizer documentation](docs/visualizer.md).


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
