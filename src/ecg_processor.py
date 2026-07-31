import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

# --- 1. Signal Generation Functions ---

def gaussian_wave(t, center, amplitude, width):
    """
    Generates a Gaussian-like wave component for ECG.
    """
    return amplitude * np.exp(-((t - center) / width)**2)

def generate_ecg(fs=256, duration_sec=10):
    """
    Generates a synthetic ECG signal with P, Q, R, S, T waves.
    """
    t = np.arange(0, duration_sec, 1/fs)
    ecg = np.zeros_like(t)

    # Simulate heart rate of 60 bpm (1 beat per second)
    beat_times = np.arange(1.0, duration_sec, 1.0) 

    # Add PQRST components for each beat
    for beat in beat_times:
        ecg += gaussian_wave(t, beat - 0.2, 0.12, 0.025)  # P wave
        ecg += gaussian_wave(t, beat - 0.04, -0.15, 0.008) # Q wave
        ecg += gaussian_wave(t, beat, 1.2, 0.01)           # R wave
        ecg += gaussian_wave(t, beat + 0.04, -0.25, 0.01)  # S wave
        ecg += gaussian_wave(t, beat + 0.28, 0.35, 0.06)   # T wave

    # Add low-frequency baseline wander
    baseline = 0.03 * np.sin(2 * np.pi * 0.3 * t)
    ecg += baseline
    
    return t, ecg, fs

# --- 2. Noise & Filtering Functions ---

def add_noise(ecg, noise_level=0.1):
    """
    Adds Gaussian (white) noise to the ECG signal.
    """
    noise = noise_level * np.random.randn(len(ecg))
    ecg_noisy = ecg + noise
    return ecg_noisy

def butter_lowpass_filter(data, cutoff, fs, order=5):
    """
    Applies a Butterworth low-pass filter to the data.
    """
    nyquist = 0.5 * fs
    normalized_cutoff = cutoff / nyquist
    b, a = butter(order, normalized_cutoff, btype='low', analog=False)
    filtered_data = lfilter(b, a, data)
    return filtered_data

# --- 3. Main Execution and Plotting ---

if __name__ == '__main__':
    # 1. Generate Clean ECG
    t, clean_ecg, fs = generate_ecg()

    # 2. Add Noise
    noisy_ecg = add_noise(clean_ecg, noise_level=0.25)
    
    # 3. Filter the Noisy Signal
    cutoff_freq = 30.0  # Common cutoff for ECG (in Hz)
    filtered_ecg = butter_lowpass_filter(noisy_ecg, cutoff_freq, fs)

    # 4. Plotting (Comparison View)
    
    # Create 3 subplots for comparison
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    # Plot 1: Clean ECG
    axs[0].plot(t, clean_ecg, label='1. Clean ECG Signal', color='green')
    axs[0].set_title('Clean Synthetic ECG')
    axs[0].legend(loc="upper right")
    axs[0].grid(True)
    
    # Plot 2: Noisy ECG
    axs[1].plot(t, noisy_ecg, label=f'2. Noisy ECG (Noise Level: 0.25)', color='red')
    axs[1].set_title('ECG with Additive White Noise')
    axs[1].legend(loc="upper right")
    axs[1].grid(True)

    # Plot 3: Filtered ECG
    axs[2].plot(t, filtered_ecg, label=f'3. Filtered ECG (Low-pass Cutoff: {cutoff_freq} Hz)', color='blue')
    axs[2].set_title('Filtered ECG Signal (Butterworth Filter)')
    axs[2].set_xlabel('Time (s)')
    axs[2].legend(loc="upper right")
    axs[2].grid(True)

    # General Plot Settings
    plt.tight_layout()
    
    # Save the figure to the Images folder
    # Note: If the folder doesn't exist, this will cause an error.
    # We assume Images/ exists from previous steps.
    try:
        plt.savefig('Images/Figure_1.png')
        print("New comparison figure saved to Images/Figure_1.png")
    except FileNotFoundError:
        print("Error: The 'Images/' directory was not found. Please create it.")
    
    # Show the plot window
    plt.show()
