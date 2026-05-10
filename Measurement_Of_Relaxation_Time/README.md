# Earth-Field NMR Signal Analysis Toolkit

A Python-based analysis suite designed for processing and modeling NMR (Nuclear Magnetic Resonance) signals captured in the Earth's magnetic field. This toolkit provides functions for raw data visualization, spectral analysis, and precise estimation of relaxation times ($T_1$ and $T_2^*$).

---

## 🔬 Physics Context
In Earth-field NMR experiments, the magnetic field is significantly weaker than in traditional superconducting magnets. This requires robust signal processing to extract meaningful data from environmental noise. This project implements non-linear least squares fitting to model the longitudinal recovery ($T_1$) and transverse decay ($T_2^*$) of nuclear spins.

## 🚀 Key Features
- **Raw Data Visualization**: Dual-channel time-domain signal plotting.
- **Spectral Analysis**: FFT implementation with Hanning windowing and targeted peak searching (e.g., 2026 Hz).
- **T1 Recovery Fitting**: Batch processing of multiple polarization time files to fit the recovery curve: $M(t) = M_0(1 - e^{-t/T_1})$.
- **T2 Decay Fitting**: Automated trigger detection and exponential decay fitting for Free Induction Decay (FID) signals.

