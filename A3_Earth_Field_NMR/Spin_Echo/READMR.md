# NMR Spin-Echo Signal Processor

## Overview
This repository contains a Python pipeline designed to process high-frequency Nuclear Magnetic Resonance (NMR) magnitude signals specifically to extract and analyze **Spin-Echo** events. 

When dealing with high sampling rates, NMR signals often contain high-frequency noise and beat interference. This script mitigates these issues through efficient **downsampling** and **moving average smoothing**, allowing for the precise extraction of the Spin-Echo signal envelope. It automatically identifies the peak amplitude of the Spin-Echo using predefined temporal windows, providing clean and reliable data points essential for accurate $T_2$ relaxation curve fitting.

## Features
* **Data Downsampling:** Reduces computational load for massive high-frequency datasets without losing the physical signal envelope.
* **Envelope Extraction:** Utilizes a configurable moving average filter to eliminate noise and reveal the true shape of the Spin-Echo.
* **Automated Peak Detection:** Implements a temporal windowing mechanism to reliably isolate the Spin-Echo peak, preventing false positives caused by background noise.
* **Data Visualization:** Automatically generates high-quality, annotated plots highlighting the detected Spin-Echo peak.

## Prerequisites
Ensure you have Python 3.8 or later installed. Install the required dependencies using the provided `requirements.txt` file.

```bash
# Install dependencies
pip install -r requirements.txt
