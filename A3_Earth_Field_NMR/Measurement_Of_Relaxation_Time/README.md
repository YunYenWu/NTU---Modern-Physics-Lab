# Measurement of Relaxation Time in Earth-Field NMR

This project provides a comprehensive Python-based toolkit for the automated measurement and analysis of **Longitudinal Relaxation Time ($T_1$)** and **Transverse Relaxation Time ($T_2^*$)** in Earth-field Nuclear Magnetic Resonance (NMR) experiments.

The toolkit is designed to process low-field NMR signals, performing high-precision non-linear regression to extract key relaxation constants essential for studying molecular dynamics and spin-lattice interactions.

---

## 🔬 Scientific Background

The relaxation times are critical parameters in NMR spectroscopy:
- **T1 (Spin-Lattice Relaxation):** Characterizes the recovery of longitudinal magnetization $M_z$ toward thermal equilibrium.
- **T2* (Effective Transverse Relaxation):** Characterizes the decay of phase coherence among nuclear spins, observed as Free Induction Decay (FID).

### Mathematical Models
The toolkit utilizes the following physical models for curve fitting:
- **T1 Recovery:** $M(t) = M_0 \left( 1 - \exp\left( -\frac{t}{T_1} \right) \right)$
- **T2* Decay:** $M(t) = M_{xy} \exp\left( -\frac{t - t_0}{T_2^*} \right)$

---

## 🚀 Key Features

- **Automated Signal Processing**: Includes auto-triggering for signal burst detection, removal of DC offsets, and Hanning windowing to reduce spectral leakage.
- **Spectral Analysis**: Fast Fourier Transform (FFT) implementation to identify precession frequency peaks (e.g., near 2026 Hz) with high resolution.
- **Robust Curve Fitting**: Employs Non-linear Least Squares (via `scipy.optimize.curve_fit`) for precise estimation of $T_1$ and $T_2^*$ with associated error margins.
- **Data Visualization**: Generates professional-grade plots comparing experimental data with theoretical fitting curves.

---

## 🛠️ Requirements & Setup

To use this toolkit, you need a Python environment with the following dependencies:

```bash
pip install -r requirements.txt
