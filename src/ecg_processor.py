import csv

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
        ecg += gaussian_wave(t, beat - 0.04, -0.15, 0.008)  # Q wave
        ecg += gaussian_wave(t, beat, 1.20, 0.010)          # R wave
        ecg += gaussian_wave(t, beat + 0.04, -0.25, 0.010)  # S wave
        ecg += gaussian_wave(t, beat + 0.28, 0.35, 0.060)   # T wave

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


# --- 4. Feature Extraction Functions ---

def extract_ecg_features(filtered_ecg, peaks, fs):
    """
    Extracts basic time-domain features from the filtered ECG signal.

    Extracted features:
    - R-peak times
    - R-peak amplitudes
    - RR intervals
    - Summary statistics of RR intervals
    """
    peak_times = peaks / fs
    peak_amplitudes = filtered_ecg[peaks]

    if len(peaks) >= 2:
        rr_intervals = np.diff(peaks) / fs
    else:
        rr_intervals = np.array([])

    if len(rr_intervals) > 0:
        rr_mean = np.mean(rr_intervals)
        rr_std = np.std(rr_intervals)
        rr_min = np.min(rr_intervals)
        rr_max = np.max(rr_intervals)
    else:
        rr_mean = 0.0
        rr_std = 0.0
        rr_min = 0.0
        rr_max = 0.0

    features = {
        "peak_times": peak_times,
        "peak_amplitudes": peak_amplitudes,
        "rr_intervals": rr_intervals,
        "rr_mean": rr_mean,
        "rr_std": rr_std,
        "rr_min": rr_min,
        "rr_max": rr_max,
    }

    return features


def save_features_to_csv(features, file_path):
    """
    Saves extracted R-peak and RR-interval features to a CSV file.
    """
    peak_times = features["peak_times"]
    peak_amplitudes = features["peak_amplitudes"]
    rr_intervals = features["rr_intervals"]

    row_count = max(len(peak_times), len(rr_intervals))

    with open(file_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)

        writer.writerow([
            "Peak Number",
            "R-Peak Time (s)",
            "R-Peak Amplitude",
            "RR Interval (s)"
        ])

        for index in range(row_count):
            peak_number = index + 1

            peak_time = (
                f"{peak_times[index]:.4f}"
                if index < len(peak_times)
                else ""
            )

            peak_amplitude = (
                f"{peak_amplitudes[index]:.4f}"
                if index < len(peak_amplitudes)
                else ""
            )

            rr_interval = (
                f"{rr_intervals[index]:.4f}"
                if index < len(rr_intervals)
                else ""
            )

            writer.writerow([
                peak_number,
                peak_time,
                peak_amplitude,
                rr_interval
            ])

        writer.writerow([])
        writer.writerow(["RR Mean (s)", f"{features['rr_mean']:.4f}"])
        writer.writerow(["RR Std (s)", f"{features['rr_std']:.4f}"])
        writer.writerow(["RR Minimum (s)", f"{features['rr_min']:.4f}"])
        writer.writerow(["RR Maximum (s)", f"{features['rr_max']:.4f}"])

    print(f"Extracted features saved to {file_path}")


# --- 5. Main Execution and Plotting ---

if __name__ == "__main__":
    # Use a fixed seed so the noise and output figures are reproducible.
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

    # Extract basic ECG features.
    features = extract_ecg_features(filtered_ecg, peaks, fs)

    print(f"Detected R-peaks: {len(peaks)}")
    print(f"Estimated heart rate: {heart_rate:.1f} BPM")
    print(f"Mean RR interval: {features['rr_mean']:.3f} seconds")
    print(f"RR interval standard deviation: {features['rr_std']:.3f} seconds")
    print(f"Minimum RR interval: {features['rr_min']:.3f} seconds")
    print(f"Maximum RR interval: {features['rr_max']:.3f} seconds")

    # Save extracted features.
    save_features_to_csv(
        features,
        "data/ecg_features.csv"
    )

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

    plt.title(
        f"R-peak Detection - Estimated Heart Rate: "
        f"{heart_rate:.1f} BPM"
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig("Images/Figure_2.png")
    print("R-peak detection figure saved to Images/Figure_2.png")

    # --- Figure 3: Extracted RR intervals and R-peak amplitudes ---

    fig, axs = plt.subplots(2, 1, figsize=(10, 7))

    if len(features["rr_intervals"]) > 0:
        axs[0].plot(
            range(1, len(features["rr_intervals"]) + 1),
            features["rr_intervals"],
            marker="o",
            color="purple",
            label="RR Intervals"
        )
        axs[0].axhline(
            features["rr_mean"],
            color="black",
            linestyle="--",
            label=f"Mean RR: {features['rr_mean']:.3f} s"
        )
    else:
        axs[0].text(
            0.5,
            0.5,
            "Not enough R-peaks for RR interval analysis",
            ha="center",
            va="center",
            transform=axs[0].transAxes
        )

    axs[0].set_title("RR Interval Analysis")
    axs[0].set_xlabel("Interval Number")
    axs[0].set_ylabel("RR Interval (s)")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(
        features["peak_times"],
        features["peak_amplitudes"],
        marker="o",
        color="darkorange",
        label="R-Peak Amplitudes"
    )
    axs[1].set_title("Detected R-Peak Amplitudes")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Amplitude")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig("Images/Figure_3.png")
    print("Feature extraction figure saved to Images/Figure_3.png")

    plt.show()
