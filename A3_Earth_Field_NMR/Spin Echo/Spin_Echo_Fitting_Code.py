"""
NMR Signal Processing and Spin-Echo Peak Detection
--------------------------------------------------
This script processes high-frequency Nuclear Magnetic Resonance (NMR) magnitude signals.
It is divided into two main analytical parts:
1. Raw Signal Visualization: Displays the original high-frequency data to observe 
   the overall signal envelope and hardware noise.
2. Signal Processing & Peak Detection: Mitigates high-frequency interference via 
   downsampling and moving average smoothing. It then automatically isolates the 
   Free Induction Decay (FID) and Spin-Echo peaks using predefined temporal windows.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# Configuration & Hyperparameters
# ==========================================
FILE_PATH = 'data.csv'

# Hardware parameters
TIME_INCREMENT = 5e-7       # Sampling time increment (e.g., 500 ns for 2MHz)

# Signal processing parameters
DOWNSAMPLE_STEP = 100       # Reduces computational load for massive high-freq datasets
SMOOTH_WINDOW_SIZE = 800    # Moving average window to extract the signal envelope

# Temporal windows (in seconds) to locate specific NMR events
# Window 1: Initial Excitation / Free Induction Decay (FID)
# Window 2: Spin Echo recovery
SEARCH_WINDOWS = [
    (1.8, 2.2),  
    (2.4, 3.8)   
]

# ==========================================
# Data Loading
# ==========================================
print("Loading NMR data...")
# Read only the first column (Voltage) to optimize memory usage
df = pd.read_csv(FILE_PATH, usecols=[0])
y_raw = df.iloc[:, 0].values

# ==========================================
# PART 1: Raw Signal Visualization
# ==========================================
print("Generating raw signal visualization...")
time_raw = np.arange(len(y_raw)) * TIME_INCREMENT

fig, ax1 = plt.subplots(figsize=(15, 6))
ax1.plot(time_raw, y_raw, color='gray', linewidth=0.1, alpha=0.6)
ax1.set_title('Raw NMR Magnitude Signal', fontsize=14)
ax1.set_xlabel('Time (s)', fontsize=12)
ax1.set_ylabel('Voltage (V)', fontsize=12)
ax1.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
plt.savefig('raw_signal_plot.png', dpi=300)
plt.close() # Close figure to free memory

# ==========================================
# PART 2: Signal Processing & Peak Detection
# ==========================================
print("Processing signal (Downsampling & Smoothing)...")

# 2.1 Downsampling
# Reduces data size significantly, enabling fast convolution without losing the envelope
y_downsampled = y_raw[::DOWNSAMPLE_STEP]
time_downsampled = np.arange(len(y_downsampled)) * TIME_INCREMENT * DOWNSAMPLE_STEP

# 2.2 Smoothing (Moving Average)
# Acts as a low-pass filter to eliminate beat frequency interference and noise
def moving_average(x, w):
    return np.convolve(x, np.ones(w)/w, mode='same')

y_smooth = moving_average(y_downsampled, SMOOTH_WINDOW_SIZE)

# 2.3 Automated Peak Detection via Temporal Windows
# Avoids false positives from noise by restricting the search to expected physical timeframes
final_results = []

for start_t, end_t in SEARCH_WINDOWS:
    # Create a boolean mask for the current time window
    mask = (time_downsampled >= start_t) & (time_downsampled <= end_t)
    
    if np.any(mask):
        # Extract indices within the window
        window_indices = np.where(mask)[0]
        # Find the index of the maximum amplitude within this specific window
        max_idx_in_window = np.argmax(y_smooth[window_indices])
        real_idx = window_indices[max_idx_in_window]
        
        final_results.append({
            'time': time_downsampled[real_idx],
            'amplitude': y_smooth[real_idx]
        })

# ==========================================
# Visualization of Processed Results
# ==========================================
print("Generating processed signal visualization...")
plt.figure(figsize=(15, 6))

# Plot the smoothed envelope
plt.plot(time_downsampled, y_smooth, color='#1f77b4', linewidth=2.5, label='Smoothed Signal Envelope')

# Annotate detected peaks
for res in final_results:
    plt.scatter(res['time'], res['amplitude'], color='red', s=100, zorder=5, edgecolor='white')
    
    label_text = f"Time: {res['time']:.4f} s\nAmp: {res['amplitude']:.4f} V"
    
    # Position the annotation box centrally above the detected peak
    plt.annotate(label_text, 
                 (res['time'], res['amplitude']), 
                 textcoords="offset points", 
                 xytext=(0, 20), 
                 ha='center', 
                 fontsize=11, 
                 color='darkred', 
                 fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.4', fc='yellow', alpha=0.3, ec='none'))

plt.title("NMR Spin Echo Envelope & Peak Detection", fontsize=16)
plt.xlabel("Time (s)", fontsize=13)
plt.ylabel("Voltage (V)", fontsize=13)
# Focus x-axis on the relevant events
plt.xlim(1.0, 4.0) 
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='upper right')
plt.tight_layout()

# Save and show the final plot
plt.savefig('processed_signal_plot.png', dpi=300)
plt.show()

# Print quantitative results to console
print("\n[Detection Results]")
print("-" * 30)
for i, res in enumerate(final_results):
    event_name = "FID" if i == 0 else "Spin Echo"
    print(f"{event_name} Peak: Time = {res['time']:.4f} s | Amplitude = {res['amplitude']:.4f} V")
print("-" * 30)
print("Pipeline execution completed.")