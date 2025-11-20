# 📘 Synthetic Pupil Light Reflex (PLR) Generator  
## How the PLR curves are created

This repository contains a physiologically realistic simulator for generating synthetic Pupil Light Reflex (PLR) signals.  
The simulation is based on the **nonlinear differential model** of Pamplona & Oliveira [1], extended with:

- **Population-level physiological variability**  
- **Stimulus flux optimization** to achieve target constriction during short light pulses
- **Automatic tuning of asymmetric constriction vs. dilation speeds** to match empirical population PLR metrics
- **Hippus-like drift** and **measurement noise**  
- Delay-based **neural latency** modeling

The generated curves closely replicate real pupillometry recordings and include *clean* (noise-free) and *observed* (noise + drift) signals as well as the *true latency*, making them ideal for evaluating pupil feature extraction algorithms.

---

## 1. Mathematical Model

The pupil diameter \(D(t)\) is governed by the nonlinear dynamic equation:

$$
\frac{dM}{dD}\frac{dD}{dt} + 2.3026\,\mathrm{atanh}\!\left(\frac{D - 4.9}{3}\right) = 5.2 - 0.45\ln\!\left(\frac{\phi(t - \tau)}{\phi_{\mathrm{ref}}}\right)\tag{1}
$$

Where:

| Symbol | Meaning |
|--------|---------|
| $D(t)$ | pupil diameter (mm) |
| $\phi(t)$ | retinal illuminance (“light flux”) |
| $\tau$ | neural latency |
| $dM/dD$ | iris mechanical stiffness (set to 1.0 for stability) |
| $\phi_{\mathrm{ref}}$ | reference illuminance (default = 1.0) |

The **atanh term** models nonlinear iris muscle tension; the **log term** implements the Weber–Fechner law for brightness perception.

## Numerical Integration
The equation is discretized using explicit Euler integration:

```python
dDdt = (rhs - mech) / dMdD
if dDdt < 0:
    dt_eff = dt / S            # fast constriction
else:
    dt_eff = dt / (3 * S)      # slow dilation
D[i] = D[i - 1] + dDdt * dt
```
The retinal illuminance is delayed by the neural latency using a simple index offset:
```python
idx_delay = max(0, i - delay_samples)
phi_effective = phi_arr[idx_delay]
```

## 2. Population-Level Parameter Sampling
To generate realistic human-like PLR variability, the simulator samples subject parameters from Capó-Aponte et al. [2]:
| Parameter                | Mean    | SD      |
| ------------------------ | ------- | ------- |
| Maximum diameter (mm)    | 5.63    | 0.79    |
| Minimum diameter (mm)    | 3.78    | 0.56    |
| Constriction latency (s) | 0.21175 | 0.00951 |

## 3. Mapping Pupil Diameter → Illuminance (Inverse Steady-State)
The steady-state solution of Eq. (1) with $dD/dt=0$ yields:
$$
2.3026\,\mathrm{atanh}\!\left(\frac{D_\infty - 4.9}{3}\right) = 5.2 - 0.45\ln\!\left(\frac{\phi}{\phi_{\mathrm{ref}}}\right) \tag{2}
$$
Solving for 
$$
\phi = \phi_{ref}\cdot\exp\left[\frac{5.2 - 2.3026\,\mathrm{atanh}\!\left(\frac{D_\infty - 4.9}{3}\right)}{0.45}\right]\tag{3}
$$
This function (implemented as flux_from_diameter) allows the simulator to compute:
- Baseline illuminance that produces the drawn baseline diameter $Dmax$⁡
- Stimulus illuminance that produces the drawn minimum diameter $Dmin$⁡

## 4. Light Stimulus and Flux Optimization
### 4.1 Standard steady-state stimulus
The illuminance required for $Dmin$⁡ at steady-state is:
```python
phi_stim_ss = flux_from_diameter(D_min)
```
### 4.2 Why optimization is needed
In real recordings (and in your simulator), the LED is brief (e.g., 0.167 s).
The full effect of `phi_stim_ss` cannot be reached during such short pulses because the iris dynamics are slower.

### 4.3 New algorithm: boosting stimulus to reach the target constriction
The simulator now includes `_find_required_phi`, which:

1. Simulates the PLR for a given candidate flux
2. Checks whether the transient minimum diameter reaches D_min
3. Uses:
    - Exponential search to find an upper bound
    - Binary search to converge to the minimal flux that achieves the desired constriction

This produces a stimulus strong enough to reach the physiologically realistic minimum diameter during a short LED flash.

## 5. Automatic Tuning of the S Parameter
This simulator automates tuning of the parameter S, introduced by Pamplona et al. [1], which is a constant that affects the constriction/dilation velocity and varies among individuals. The simulator is tuning S so that the simulated PLR matches known physiological metrics [2]:

| Metric                        | Target (mean ± SD) |
| ----------------------------- | ------------------ |
| Average constriction velocity | −4.11 ± 0.44 mm/s  |
| Maximum constriction velocity | −5.15 ± 0.99 mm/s  |
| Dilation velocity             | +1.02 ± 0.17 mm/s  |
| 75% recovery time             | 1.77 ± 0.38 s      |

**How S is estimated**
The simulator:
1. Simulates the PLR for many candidate S values
2. Measures realistic features from the simulated trace
3. Computes a weighted squared-error loss
4. Performs a two-stage search:
    - coarse logarithmic grid
    - fine grid around the best value

This yields a physiologically personalized value of S for each synthetic subject.

## 6. Full Simulation Pipeline
Each simulated PLR curve is generated as follows:

### Step 1 — Draw subject parameters
```python
D_max ~ N(5.63, 0.79)
D_min ~ N(3.78, 0.56)
tau_latency ~ N(0.21175, 0.00951)
```

### Step 2 — Compute steady-state retinal fluxes
```python
phi_baseline = flux_from_diameter(D_max)
phi_stim_ss = flux_from_diameter(D_min)
```

### Step 3 — Optimize stimulus flux
```python
phi_stim = _find_required_phi(...)
```

### Step 4 — Create illuminance signal with latency
```python
phi_arr = baseline everywhere
phi_arr[on_mask] = phi_stim
phi_effective = phi_arr[i - delay_samples]
```

### Step 5 — Integrate nonlinear Eq. (16)
This yields the clean signal D_clean.

### Step 6 — Add realistic physiological noise
- Hippus drift (0.05–0.3 Hz sinusoid)
- Additive Gaussian noise

Final observed diameter:

$$
D_{obs}=D_{clean}+\mathrm{drift}+\mathrm{noise}
$$

## 7. Output
The simulator returns:

```python
time        # time vector (s)
D_obs       # noisy PLR diameter (mm)
D_clean     # noise-free diameter (mm)
true_latency  # stim_time + tau_latency
params      # dict containing all sample-specific parameters
```

Example:
```python
time, D_obs, D_clean, true_lat, params = simulate_plr_eq16_population(seed=123)
```

[1] V. F. Pamplona, M. M. Oliveira, und G. V. G. Baranoski, „Photorealistic models for pupil light reflex and iridal pattern deformation“, ACM Trans. Graph., Bd. 28, Nr. 4, S. 106:1-106:12, Sep. 2009, doi: 10.1145/1559755.1559763.
<br>
[2] J. E. Capó-Aponte, T. A. Beltran, D. V. Walsh, W. R. Cole, und J. Y. Dumayas, „Validation of Visual Objective Biomarkers for Acute Concussion“, Military Medicine, Bd. 183, Nr. suppl_1, S. 9–17, März 2018, doi: 10.1093/milmed/usx166.
