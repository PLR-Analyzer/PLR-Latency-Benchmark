# 👁️ Synthetic Pupil Light Reflex (PLR) Generator  
## How the PLR curves are created

This repository contains a physiologically realistic simulator for generating synthetic Pupil Light Reflex (PLR) signals.  
The simulation is based on the **nonlinear differential model** of Pamplona & Oliveira [1], including:

- **Population-level physiological variability**  
- **Stimulus flux optimization** to achieve target constriction during short light pulses
- **Hippus-like drift** and **measurement noise**  
- Delay-based **neural latency** modeling

The generated curves closely replicate real pupillometry recordings and include *clean* (noise-free) and *observed* (noise + drift) signals as well as the *true latency*, making them ideal for evaluating pupil feature extraction algorithms.

---

## 1. Mathematical Model

The pupil diameter \(D(t)\) is governed by the nonlinear dynamic equation:

$$
\frac{dM}{dD}\frac{dD}{dt} + 2.3026\,\mathrm{atanh}\!\left(\frac{D - 4.9}{3}\right) = 5.2 - 0.45\ln\!\left(\frac{\phi(t - \tau)}{\phi_{\mathrm{ref}}}\right)\tag{1}
$$
with
$$
M(D) = \operatorname{atanh} \left( \frac{D-4.9}{3} \right),
$$

Where:

| Symbol | Meaning |
|--------|---------|
| $D(t)$ | pupil diameter (mm) |
| $\phi(t)$ | retinal illuminance (“light flux”) |
| $\tau$ | neural latency |
| $\phi_{\mathrm{ref}}$ | reference illuminance (default = 4.8118e-10) |

## Numerical Integration
The equation is discretized using explicit Euler integration.
The retinal illuminance is delayed by the neural latency. The calculation of the delayed stimulus $\phi(t-\tau)$ is as follows:

Let $t_k$ denote the time of a luminance change, $\tau_k$ the latency associated with the new stimulus, and $\phi(t_{k-1})$ the stimulus immediately preceding the change. Shorter latencies are adopted immediately, while longer latencies are applied only after the currently active latency window has elapsed. Specifically, when a shorter latency becomes active, the tuple $(\tau_\mathrm{active}, t_\mathrm{active}, \phi_\mathrm{hold}) = (\tau_k, t_k, \phi(t_{k-1}))$ is stored. Longer latencies are adopted only for $t \geq t_\mathrm{active} + \tau_\mathrm{active}$. The delayed stimulus is thus defined as
$$
    \phi_\mathrm{delayed}(t) =
    \begin{cases}
        \phi_\mathrm{hold}, & t < t_\mathrm{active} + \tau_\mathrm{active}, \\
        \phi(t - \tau_\mathrm{active}), & t \geq t_\mathrm{active} + \tau_\mathrm{active}.
    \end{cases}
$$
This ensures that brief light pulses with durations shorter than the nominal latency elicit a single, uninterrupted pupil response.

## 3. Mapping Pupil Diameter → Illuminance (Inverse Steady-State)
The steady-state solution of Eq. (1) with $dD/dt=0$ yields:
$$
2.3026\,\mathrm{atanh}\!\left(\frac{D_\infty - 4.9}{3}\right) = 5.2 - 0.45\ln\!\left(\frac{\phi}{\phi_{\mathrm{ref}}}\right) \tag{2}
$$
Solving for 
$$
\phi = \phi_{ref}\cdot\exp\left[\frac{5.2 - 2.3026\,\mathrm{atanh}\!\left(\frac{D_\infty - 4.9}{3}\right)}{0.45}\right]\tag{3}
$$
This function (implemented as `flux_from_diameter`) allows the simulator to compute:
- Baseline illuminance that produces the selected baseline diameter $Dmax$⁡
- Stimulus illuminance that produces the selected minimum diameter $Dmin$⁡

## 4. Light Stimulus and Flux Optimization
The illuminance required for $Dmin$⁡ at steady-state is calculated in `find_required_phi` in `simulation.py`. The find_required_phi function determines what light stimulus intensity (phi) is needed to make the pupil constrict to a target diameter in the time until the pupil starts to redilate.

With short light stimuli, the full effect of the stimulus cannot be achieved because redilation begins before the minimum pupil diameter has been reached. In order for the pupil to achieve the specified constriction, the intensity must therefore be stronger with a short light pulse than with a long light pulse.

The function works as follows:
1. Simulates the PLR for a given candidate flux
2. Checks whether the transient minimum diameter reaches D_min
3. Uses:
    - Exponential search to find an upper bound
    - Binary search to converge to the minimal flux that achieves the desired constriction

This produces a stimulus strong enough to reach the physiologically realistic minimum diameter during a short LED flash.

## 5. Full Simulation Pipeline
Each simulated PLR curve is generated as follows:

### Step 1 — Initialize time vector
Create a discrete time vector based on fps and duration.

### Step 2 — Set up baseline illuminance
$$\phi_{\mathrm{baseline}} = \phi(D_{\max})$$
Initialize the illuminance signal to baseline everywhere.

### Step 3 — Find stimulus flux
Use `find_required_phi()` to compute the stimulus intensity required to achieve constriction to $D_{\min}$ during the light stimulus window (accounting for the brief stimulus duration). This ensures realistic stimulus intensities.

### Step 4 — Create stimulus array
$$\phi_{\mathrm{arr}}[t] = \begin{cases}
\phi_{\mathrm{stim}} & \text{if } t_{\mathrm{stim}} \leq t < t_{\mathrm{stim}} + \Delta t_{\mathrm{stim}} \\
\phi_{\mathrm{baseline}} & \text{otherwise}
\end{cases}$$

### Step 5 — Calculate neural latency
Compute the latency $\tau$ from the stimulus flux using the latency model.

### Step 6 — Integrate the nonlinear ODE
Simulate the pupil diameter response using explicit Euler integration with the illuminance signal (delayed by $\tau$).

### Step 7 — Apply individual variability
Apply an isocurve adjustment ($r_I$ parameter) to introduce population-level physiological variability in pupil response curves.

### Step 8 — Add hippus and measurement noise
- **Hippus drift**: Low-frequency sinusoid (0.05–0.3 Hz) to simulate natural pupil oscillations
- **Measurement noise**: Additive Gaussian noise ($\sigma = 0.03$ mm by default)

$$D_{\mathrm{obs}} = D_{\mathrm{clean}} + \mathrm{drift} + \mathrm{noise}$$

### Step 9 — Return results
Return time vector, observed diameter, clean diameter, stimulus onset + latency, and all simulation parameters.

## Example Usage:
```python
time, D_obs, D_clean, true_lat, params = simulate_sample()
```

[1] V. F. Pamplona, M. M. Oliveira, und G. V. G. Baranoski, „Photorealistic models for pupil light reflex and iridal pattern deformation“, ACM Trans. Graph., Bd. 28, Nr. 4, S. 106:1-106:12, Sep. 2009, doi: 10.1145/1559755.1559763.
