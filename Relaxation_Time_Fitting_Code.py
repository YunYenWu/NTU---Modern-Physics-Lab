import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

# =================================================================
# 1. RAW DATA VISUALIZATION
# =================================================================

def plot_time_domain_signals(file_path, t_inc=1e-6):
    """
    Reads NMR raw data from a CSV file and plots time-domain signals for CH1 and CH2.
    
    Args:
        file_path (str): Path to the .csv data file.
        t_inc (float): Sampling interval in seconds. Default is 1e-6 (1 microsecond).
    """
    try:
        print(f"--- Processing: {os.path.basename(file_path)} ---")
        
        # Load dataset: Assuming Column 0 is CH1 and Column 1 is CH2
        # We use usecols to optimize memory for large CSV files
        df = pd.read_csv(file_path, usecols=[0, 1])
        
        # Convert to numpy arrays for faster computation
        y1 = df.iloc[:, 0].values  # Channel 1 signal (Voltage)
        y2 = df.iloc[:, 1].values  # Channel 2 signal (Voltage)
        
        # Generate time axis based on sampling increment
        time = np.arange(len(y1)) * t_inc

        # Basic statistical diagnostics
        print(f"CH1 Peak-to-Peak: {np.ptp(y1):.4f} V")
        print(f"CH2 Peak-to-Peak: {np.ptp(y2):.4f} V")

        # Plotting Setup: Two-row vertical stack for signal comparison
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

        # Upper Plot: Channel 1 Raw Signal
        ax1.plot(time, y1, color='royalblue', linewidth=0.1, alpha=0.7)
        ax1.set_title('Time-Domain Signal (CH1)', fontsize=14)
        ax1.set_ylabel('Amplitude (V)', fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.5)

        # Lower Plot: Channel 2 Raw Signal
        ax2.plot(time, y2, color='crimson', linewidth=0.1, alpha=0.7)
        ax2.set_title('Time-Domain Signal (CH2)', fontsize=14)
        ax2.set_xlabel('Time (s)', fontsize=12)
        ax2.set_ylabel('Amplitude (V)', fontsize=12)
        ax2.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        
        # Optional: Save the plot automatically
        # output_name = os.path.basename(file_path).replace('.csv', '_raw.png')
        # plt.savefig(output_name, dpi=300)
        
        plt.show()

    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# =================================================================
# 2. FFT FREQUENCY SPECTRUM ANALYSIS
# =================================================================

def analyze_frequency_spectrum(file_path, t_inc=1e-6, target_freq=2026, threshold=1.5):
    """
    Performs FFT analysis to identify the precession frequency peak.
    Includes auto-triggering, signal windowing, and frequency masking.

    Args:
        file_path (str): Path to the CSV file.
        t_inc (float): Sampling interval.
        target_freq (float): The expected frequency in Hz (e.g., 2026 Hz).
        threshold (float): Voltage threshold to detect the start of the NMR signal.
    """
    try:
        print(f"--- Performing FFT Analysis: {os.path.basename(file_path)} ---")
        
        # Load CH1 data only to save memory
        df = pd.read_csv(file_path, usecols=[0])
        y_raw = df.iloc[:, 0].values

        # --- 1. Auto-Trigger Mechanism ---
        # Find the first index where the absolute voltage exceeds the threshold
        trigger_indices = np.where(np.abs(y_raw) > threshold)[0]
        
        if len(trigger_indices) == 0:
            print(f"Warning: No signal detected above {threshold}V threshold.")
            return None

        # Define the processing window (2.5 million points from trigger)
        start_idx = trigger_indices[0]
        end_idx = min(start_idx + 2500000, len(y_raw))
        y_sliced = y_raw[start_idx:end_idx]

        # --- 2. Signal Pre-processing ---
        # Detrending: Remove DC offset to prevent a large spike at 0 Hz
        y_detrend = y_sliced - np.mean(y_sliced)
        
        # Windowing: Apply Hanning window to minimize spectral leakage
        window = np.hanning(len(y_detrend))
        y_windowed = y_detrend * window

        # --- 3. Fast Fourier Transform (FFT) ---
        N = len(y_windowed)
        fft_values = np.fft.fft(y_windowed)
        fft_mag = np.abs(fft_values) / N * 2  # Normalize amplitude
        freqs = np.fft.fftfreq(N, t_inc)

        # --- 4. Targeted Peak Search ---
        # We focus on the 1800-2300 Hz range for visualization
        plot_mask = (freqs > 1800) & (freqs < 2300)
        f_plot = freqs[plot_mask]
        m_plot = fft_mag[plot_mask]
        
        # Precise search: Look for the maximum within +/- 15 Hz of the target frequency
        search_window = 15
        search_mask = (f_plot > target_freq - search_window) & (f_plot < target_freq + search_window)
        
        m_search = m_plot.copy()
        m_search[~search_mask] = 0  # Zero out magnitudes outside the target window
        
        peak_idx = np.argmax(m_search)
        actual_f = f_plot[peak_idx]
        actual_m = m_plot[peak_idx]

        if actual_m == 0:
            print(f"Caution: No valid peak found near {target_freq} Hz.")

        # --- 5. Visualization ---
        plt.figure(figsize=(8, 5))
        plt.plot(f_plot, m_plot, color='steelblue', label='FFT Spectrum', linewidth=1)
        plt.plot(actual_f, actual_m, 'ro', label=f'Peak: {actual_f:.2f} Hz')
        
        # Annotation for the peak frequency
        plt.annotate(f"{actual_f:.2f} Hz", xy=(actual_f, actual_m), 
                     textcoords="offset points", xytext=(0, -15), ha='center', fontweight='bold')

        plt.title(f"NMR Frequency Spectrum ({os.path.basename(file_path)})")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude (V)")
        plt.xlim(1800, 2300)
        plt.ylim(0, max(m_plot) * 1.3)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

        return actual_f, actual_m

    except Exception as e:
        print(f"FFT Analysis failed: {e}")
        return None

# =================================================================
# PHYSICS MODELS
# =================================================================

def t1_recovery_model(t, M0, T1):
    """
    Standard T1 recovery equation (Longitudinal relaxation).
    M(t) = M0 * (1 - exp(-t / T1))
    """
    return M0 * (1 - np.exp(-t / T1))

def t2_star_decay_model(t, Mxy, T2star):
    """
    Exponential decay model for T2* relaxation (Transverse relaxation).
    M(t) = Mxy * exp(-(t - t0) / T2*)
    Note: (t - t0) ensures the decay starts from the first fitting point.
    """
    return Mxy * np.exp(-(t - t[0]) / T2star)

# =================================================================
# 3. T1 RECOVERY BATCH FITTING
# =================================================================

def perform_t1_batch_fitting(file_list, p_times, t_inc=1e-6, threshold=1.5):
    """
    Processes multiple NMR data files, extracts frequency peaks, 
    and performs a non-linear fit to determine T1 relaxation time.

    Args:
        file_list (list): List of paths to CSV files.
        p_times (list): List of polarization times (seconds) corresponding to the files.
        t_inc (float): Sampling interval.
        threshold (float): Trigger threshold for signal detection.
    """
    magnitudes = []
    valid_ptimes = []
    slice_len = 2500000

    print("\n--- Starting T1 Batch Processing ---")

    for p_time, file in zip(p_times, file_list):
        try:
            # Check if file exists to avoid crash
            if not os.path.exists(file):
                print(f"Skipping: {file} (File not found)")
                continue

            # Load data (CH1)
            df = pd.read_csv(file, usecols=[0])
            y_raw = df.iloc[:, 0].values
            
            # 1. Signal Localization (Auto-Trigger)
            trigger_indices = np.where(np.abs(y_raw) > threshold)[0]
            if len(trigger_indices) == 0:
                print(f"  [Warning] {p_time}s: No trigger found. Skipping.")
                continue
                
            start_idx = trigger_indices[0]
            end_idx = min(start_idx + slice_len, len(y_raw))
            y_sliced = y_raw[start_idx:end_idx]
            
            # 2. Signal Conditioning
            y_detrend = y_sliced - np.mean(y_sliced)  # Remove DC
            window = np.hanning(len(y_detrend))       # Apply Hanning window
            y_windowed = y_detrend * window
            
            # 3. Frequency Analysis (FFT)
            N = len(y_windowed)
            fft_mag = np.abs(np.fft.fft(y_windowed)) / N * 2
            freqs = np.fft.fftfreq(N, t_inc)
            
            # Search for peak in the expected precession range (2000-2500 Hz)
            mask = (freqs > 2000) & (freqs < 2500)
            peak_val = np.max(fft_mag[mask])
            
            magnitudes.append(peak_val)
            valid_ptimes.append(p_time)
            print(f"  Processed {p_time:>4}s: Peak Magnitude = {peak_val:.4f} V")
            
        except Exception as e:
            print(f"  [Error] Failed to process {p_time}s: {e}")

    # --- 4. T1 Curve Fitting ---
    if len(valid_ptimes) < 2:
        print("Error: Not enough data points for T1 fitting.")
        return None

    print("\nExecuting Non-linear Least Squares Fit...")
    try:
        # Initial guess p0: [Max observed amplitude, initial T1 estimate of 3s]
        p0_guess = [max(magnitudes), 3.0]
        popt, pcov = curve_fit(t1_recovery_model, valid_ptimes, magnitudes, p0=p0_guess)
        
        M0_fit, T1_fit = popt
        T1_err = np.sqrt(np.diag(pcov))[1]  # Standard deviation of T1

        print("-" * 40)
        print(f"RESULT: T1 = {T1_fit:.3f} ± {T1_err:.3f} seconds")
        print(f"M0 (Saturation): {M0_fit:.3f} V")
        print("-" * 40)

        # --- 5. Result Visualization ---
        t_span = np.linspace(0, max(valid_ptimes) * 1.1, 200)
        y_fit_curve = t1_recovery_model(t_span, *popt)

        plt.figure(figsize=(10, 6))
        plt.scatter(valid_ptimes, magnitudes, color='#E24A33', s=60, 
                    label='Experimental Data (Peak Amplitudes)', zorder=5)
        plt.plot(t_span, y_fit_curve, color='steelblue', linewidth=2, 
                 label=f'Best Fit ($T_1 \approx {T1_fit:.2f}$ s)')

        plt.title("NMR T1 Relaxation Analysis: Signal Recovery", fontsize=14)
        plt.xlabel("Polarization Time $t_p$ (s)", fontsize=12)
        plt.ylabel("Precession Amplitude (V)", fontsize=12)
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.show()

        return popt

    except Exception as fit_error:
        print(f"Fitting failed: {fit_error}")
        return None
    
# =================================================================
# 4. T2* RELAXATION ANALYSIS
# =================================================================

def fit_t2_star_relaxation(file_path, t_inc=2e-7, threshold=1.5, 
                           fit_offset=50000, fit_length=10000000):
    """
    Detects the NMR signal trigger and fits the exponential decay to find T2*.

    Args:
        file_path (str): Path to the CSV file.
        t_inc (float): Sampling interval (default 2e-7 for high-res decay data).
        threshold (float): Signal trigger threshold in Volts.
        fit_offset (int): Number of points to skip after trigger to avoid pulse artifacts.
        fit_length (int): Total number of data points used for the fitting.
    """
    try:
        print(f"\n--- Analyzing T2* Decay: {os.path.basename(file_path)} ---")
        
        # Load CH2 data (Magnitude/Envelope signal)
        df = pd.read_csv(file_path, usecols=[1])
        y_raw = df.iloc[:, 0].values
        time_raw = np.arange(len(y_raw)) * t_inc

        # 1. Trigger Detection
        trigger_indices = np.where(np.abs(y_raw) > threshold)[0]
        if len(trigger_indices) == 0:
            print("Error: Signal trigger not found. Adjust the threshold.")
            return None

        actual_trigger = trigger_indices[0]
        
        # 2. Slicing the Decay Envelope
        # We start fitting after an offset to ensure we're looking at the decay, not the pulse.
        t_start_idx = actual_trigger + fit_offset
        t_end_idx = min(t_start_idx + fit_length, len(y_raw))
        
        fit_time = time_raw[t_start_idx:t_end_idx]
        fit_y = y_raw[t_start_idx:t_end_idx]

        print(f"Trigger found at index {actual_trigger}. Fitting {len(fit_y)} points.")

        # 3. Non-linear Least Squares Fit
        # p0: Initial guess [Max amplitude, T2* estimate of 0.5s]
        popt, pcov = curve_fit(t2_star_decay_model, fit_time, fit_y, p0=[np.max(fit_y), 0.5])
        Mxy_fit, T2star_fit = popt
        T2star_err = np.sqrt(np.diag(pcov))[1]

        print("-" * 40)
        print(f"SUCCESS: T2* = {T2star_fit:.6f} ± {T2star_err:.6f} s")
        print("-" * 40)

        # 4. Result Visualization
        y_fit_curve = t2_star_decay_model(fit_time, *popt)

        plt.figure(figsize=(12, 6))
        plt.plot(fit_time, fit_y, color='blue', alpha=0.4, label='Experimental Data (FID)')
        plt.plot(fit_time, y_fit_curve, color='red', linewidth=2, 
                 label=f'Best Fit ($T_2^* = {T2star_fit:.5f}$ s)')

        plt.title(f"NMR $T_2^*$ Relaxation Fit", fontsize=14)
        plt.xlabel("Time (s)", fontsize=12)
        plt.ylabel("Voltage (V)", fontsize=12)
        plt.xlim(fit_time[0], fit_time[-1])
        plt.legend(loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

        return T2star_fit

    except Exception as e:
        print(f"T2* fitting failed: {e}")
        return None
    
# =================================================================
# MAIN EXECUTION BLOCK (Example Usage)
# =================================================================

if __name__ == "__main__":
    # --- Instructions for Users ---
    # The raw CSV data is NOT included in this repository due to size limits.
    # To run these examples, please create a 'data' folder in the same directory 
    # as this script and place your NMR .csv files inside.
    
    DATA_DIR = "data"
    
    print("=========================================")
    print("  NMR Analysis Toolkit Initialized.")
    print("  Please uncomment the code blocks below ")
    print("  and update the file names to run.")
    print("=========================================\n")

    # ---------------------------------------------------------
    # Example A: Raw Data Visualization
    # ---------------------------------------------------------
    # test_file = os.path.join(DATA_DIR, "your_raw_data.csv")
    # if os.path.exists(test_file):
    #     plot_time_domain_signals(test_file)
    # else:
    #     print(f"File not found: {test_file}. Please add data to run Example A.")

    # ---------------------------------------------------------
    # Example B: T1 Recovery Batch Fitting
    # ---------------------------------------------------------
    # t1_files = [
    #     os.path.join(DATA_DIR, "05sec.csv"),
    #     os.path.join(DATA_DIR, "1sec.csv"),
    #     os.path.join(DATA_DIR, "5sec.csv"),
    #     os.path.join(DATA_DIR, "13sec.csv")
    # ]
    # t1_p_times = [0.5, 1.0, 5.0, 13.0]
    # # perform_t1_batch_fitting(t1_files, t1_p_times)

    # ---------------------------------------------------------
    # Example C: T2* Decay Fitting
    # ---------------------------------------------------------
    # t2_file = os.path.join(DATA_DIR, "t2_decay_data.csv")
    # if os.path.exists(t2_file):
    #     fit_t2_star_relaxation(t2_file, t_inc=2e-7)