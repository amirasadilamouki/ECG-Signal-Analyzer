import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, find_peaks


# --- 1. Signal Generation Functions ---

def gaussian_wave(t, center, amplitude, width):
    """
    Generates a Gaussian-like wave component for a synthetic ECG signal.
    """
    return amplitude * np.exp(-((t - center) / width) ** 2)


def generate_ecg(fs=256, duration_sec=10):
    """
    Generates a synthetic ECG signal with simple P-Q-R-S-T morphology.
    """
    t = np.arange(0, duration_sec, 1 / fs)
    ecg = np.zeros_like(t)

    # Simulate a heart rate of approximately 60 BPM.
    beat_times = np.arange(1.0, duration_sec, 1.0)

    for beat in beat_times:
        ecg += gaussian_wave(t, beat - 0.20, 0.12, 0.025)  # P wave
        ecg += gaussian_wave(t, beat - 0.04, -0.15, 0.008) # Q wave
        ecg += gaussian_wave(t, beat, 1.20, 0.010)         # R wave
        ecg += gaussian_wave(t, beat + 0.04, -0.25, 0.010) # S wave
        ecg += gaussian_wave(t, beat + 0.28, 0.35, 0.060)  # T wave

    # Add low-frequency baseline wander.
    baseline = 0.03 * np.sin(2 * np.pi * 0.3 * t)
    ecg += baseline

    return t, ecg, fs


# --- 2. Noise and Filtering Functions ---

def add_noise(ecg, noise_level=0.1):
    """
    Adds Gaussian white noise to the ECG signal.
    """
    noise = noise_level * np.random.randn(len(ecg))
    return ecg + noise


def butter_lowpass_filter(data, cutoff, fs, order=5):
    """
    Applies a Butterworth low-pass filter to the input signal.
    """
    nyquist = 0.5 * fs
    normalized_cutoff = cutoff / nyquist

    b, a = butter(order, normalized_cutoff, btype="low", analog=False)
    filtered_data = lfilter(b, a, data)

    return filtered_data


# --- 3. R-Peak Detection and Heart Rate Estimation ---

def detect_r_peaks(filtered_ecg, fs):
    """
    Detects R-peaks in the filtered ECG signal.
    """
    peaks, _ = find_peaks(
        filtered_ecg,
        distance=int(0.6 * fs),
        prominence=0.5
    )

    return peaks


def calculate_heart_rate(peaks, fs):
    """
    Calculates estimated heart rate in BPM from R-R intervals.
    """
    if len(peaks) < 2:
        return 0.0

    rr_intervals = np.diff(peaks) / fs
    average_rr_interval = np.mean(rr_intervals)

    return 60 / average_rr_interval


# --- 4. Main Execution and Plotting ---

if __name__ == "__main__":
    # Use a fixed seed so the noise and output figure are reproducible.
    np.random.seed(42)

    # Generate clean synthetic ECG.
    t, clean_ecg, fs = generate_ecg()

    # Add noise.
    noise_level = 0.25
    noisy_ecg = add_noise(clean_ecg, noise_level=noise_level)

    # Filter the noisy ECG.
    cutoff_freq = 30.0
    filtered_ecg = butter_lowpass_filter(noisy_ecg, cutoff_freq, fs)

    # Detect R-peaks and estimate heart rate.
    peaks = detect_r_peaks(filtered_ecg, fs)
    heart_rate = calculate_heart_rate(peaks, fs)

    print(f"Detected R-peaks: {len(peaks)}")
    print(f"Estimated heart rate: {heart_rate:.1f} BPM")

    # --- Figure 1: Signal comparison ---

    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axs[0].plot(t, clean_ecg, label="1. Clean ECG Signal", color="green")
    axs[0].set_title("Clean Synthetic ECG")
    axs[0].legend(loc="upper right")
    axs[0].grid(True)

    axs[1].plot(
        t,
        noisy_ecg,
        label=f"2. Noisy ECG (Noise Level: {noise_level})",
        color="red"
    )
    axs[1].set_title("ECG with Additive White Noise")
    axs[1].legend(loc="upper right")
    axs[1].grid(True)

    axs[2].plot(
        t,
        filtered_ecg,
        label=f"3. Filtered ECG (Low-pass Cutoff: {cutoff_freq} Hz)",
        color="blue"
    )
    axs[2].set_title("Filtered ECG Signal (Butterworth Filter)")
    axs[2].set_xlabel("Time (s)")
    axs[2].legend(loc="upper right")
    axs[2].grid(True)

    plt.tight_layout()
    plt.savefig("Images/Figure_1.png")
    print("Comparison figure saved to Images/Figure_1.png")

    # --- Figure 2: R-peak detection ---

    plt.figure(figsize=(10, 5))

    plt.plot(t, filtered_ecg, label="Filtered ECG", color="blue")
    plt.plot(
        t[peaks],
        filtered_ecg[peaks],
        "ro",
        label="Detected R-peaks"
    )

    plt.title(f"R-peak Detection - Estimated Heart Rate: {heart_rate:.1f} BPM")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig("Images/Figure_2.png")
    print("R-peak detection figure saved to Images/Figure_2.png")

    plt.show()
