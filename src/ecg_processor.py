import csv
import os

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

    beat_times = np.arange(1.0, duration_sec, 1.0)

    for beat in beat_times:
        ecg += gaussian_wave(t, beat - 0.20, 0.12, 0.025)
        ecg += gaussian_wave(t, beat - 0.04, -0.15, 0.008)
        ecg += gaussian_wave(t, beat, 1.20, 0.010)
        ecg += gaussian_wave(t, beat + 0.04, -0.25, 0.010)
        ecg += gaussian_wave(t, beat + 0.28, 0.35, 0.060)

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

    b, a = butter(
        order,
        normalized_cutoff,
        btype="low",
        analog=False
    )

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

    return {
        "peak_times": peak_times,
        "peak_amplitudes": peak_amplitudes,
        "rr_intervals": rr_intervals,
        "rr_mean": rr_mean,
        "rr_std": rr_std,
        "rr_min": rr_min,
        "rr_max": rr_max,
    }


def save_features_to_csv(features, file_path):
    """
    Saves extracted ECG features to a CSV file.
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
            writer.writerow([
                index + 1,
                f"{peak_times[index]:.4f}"
                if index < len(peak_times) else "",
                f"{peak_amplitudes[index]:.4f}"
                if index < len(peak_amplitudes) else "",
                f"{rr_intervals[index]:.4f}"
                if index < len(rr_intervals) else ""
            ])

        writer.writerow([])
        writer.writerow(["RR Mean (s)", f"{features['rr_mean']:.4f}"])
        writer.writerow(["RR Std (s)", f"{features['rr_std']:.4f}"])
        writer.writerow(["RR Minimum (s)", f"{features['rr_min']:.4f}"])
        writer.writerow(["RR Maximum (s)", f"{features['rr_max']:.4f}"])

    print(f"Extracted features saved to {file_path}")


# --- 5. HRV Analysis Functions ---

def calculate_hrv(rr_intervals):
    """
    Calculates basic time-domain HRV metrics.

    SDNN is the standard deviation of normal-to-normal
    RR intervals.
    """
    if len(rr_intervals) == 0:
        return {
            "mean_rr": 0.0,
            "sdnn": 0.0,
            "min_rr": 0.0,
            "max_rr": 0.0
        }

    return {
        "mean_rr": np.mean(rr_intervals),
        "sdnn": np.std(rr_intervals, ddof=1)
        if len(rr_intervals) > 1 else 0.0,
        "min_rr": np.min(rr_intervals),
        "max_rr": np.max(rr_intervals)
    }


def save_hrv_to_csv(hrv_results, file_path):
    """
    Saves HRV metrics to a CSV file.
    """
    with open(file_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)

        writer.writerow(["HRV Metric", "Value"])
        writer.writerow(["Mean RR (s)", f"{hrv_results['mean_rr']:.4f}"])
        writer.writerow(["SDNN (s)", f"{hrv_results['sdnn']:.4f}"])
        writer.writerow(["Minimum RR (s)", f"{hrv_results['min_rr']:.4f}"])
        writer.writerow(["Maximum RR (s)", f"{hrv_results['max_rr']:.4f}"])

    print(f"HRV results saved to {file_path}")


# --- 6. Main Execution and Plotting ---

if __name__ == "__main__":
    np.random.seed(42)

    os.makedirs("data", exist_ok=True)
    os.makedirs("Images", exist_ok=True)

    t, clean_ecg, fs = generate_ecg()

    noise_level = 0.25
    noisy_ecg = add_noise(clean_ecg, noise_level)

    cutoff_freq = 30.0
    filtered_ecg = butter_lowpass_filter(
        noisy_ecg,
        cutoff_freq,
        fs
    )

    peaks = detect_r_peaks(filtered_ecg, fs)
    heart_rate = calculate_heart_rate(peaks, fs)

    features = extract_ecg_features(
        filtered_ecg,
        peaks,
        fs
    )

    hrv_results = calculate_hrv(
        features["rr_intervals"]
    )

    print(f"Detected R-peaks: {len(peaks)}")
    print(f"Estimated heart rate: {heart_rate:.1f} BPM")
    print(f"Mean RR interval: {hrv_results['mean_rr']:.3f} seconds")
    print(f"SDNN: {hrv_results['sdnn']:.3f} seconds")
    print(f"Minimum RR interval: {hrv_results['min_rr']:.3f} seconds")
    print(f"Maximum RR interval: {hrv_results['max_rr']:.3f} seconds")

    save_features_to_csv(
        features,
        "data/ecg_features.csv"
    )

    save_hrv_to_csv(
        hrv_results,
        "data/hrv_results.csv"
    )

    # --- Figure 1: Signal Comparison ---

    fig, axs = plt.subplots(
        3,
        1,
        figsize=(10, 8),
        sharex=True
    )

    axs[0].plot(
        t,
        clean_ecg,
        label="Clean ECG Signal",
        color="green"
    )
    axs[0].set_title("Clean Synthetic ECG")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(
        t,
        noisy_ecg,
        label=f"Noisy ECG - Noise Level: {noise_level}",
        color="red"
    )
    axs[1].set_title("ECG with Additive White Noise")
    axs[1].legend()
    axs[1].grid(True)

    axs[2].plot(
        t,
        filtered_ecg,
        label=f"Filtered ECG - Cutoff: {cutoff_freq} Hz",
        color="blue"
    )
    axs[2].set_title("Filtered ECG Signal")
    axs[2].set_xlabel("Time (s)")
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    plt.savefig("Images/Figure_1.png")
    plt.close(fig)

    # --- Figure 2: R-Peak Detection ---

    plt.figure(figsize=(10, 5))

    plt.plot(
        t,
        filtered_ecg,
        label="Filtered ECG",
        color="blue"
    )

    plt.plot(
        t[peaks],
        filtered_ecg[peaks],
        "ro",
        label="Detected R-peaks"
    )

    plt.title(
        f"R-Peak Detection - Estimated Heart Rate: "
        f"{heart_rate:.1f} BPM"
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Images/Figure_2.png")
    plt.close()

    # --- Figure 3: Feature Extraction ---

    fig, axs = plt.subplots(
        2,
        1,
        figsize=(10, 7)
    )

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
    plt.close(fig)

    # --- Figure 4: Basic HRV Analysis ---

    plt.figure(figsize=(10, 5))

    rr_intervals = features["rr_intervals"]

    if len(rr_intervals) > 0:
        interval_numbers = range(1, len(rr_intervals) + 1)

        plt.plot(
            interval_numbers,
            rr_intervals,
            marker="o",
            color="teal",
            label="RR Intervals"
        )

        plt.axhline(
            hrv_results["mean_rr"],
            color="black",
            linestyle="--",
            label=f"Mean RR: {hrv_results['mean_rr']:.3f} s"
        )

        plt.text(
            0.02,
            0.95,
            f"SDNN: {hrv_results['sdnn']:.4f} s",
            transform=plt.gca().transAxes,
            verticalalignment="top",
            bbox=dict(
                boxstyle="round",
                facecolor="white",
                alpha=0.8
            )
        )
    else:
        plt.text(
            0.5,
            0.5,
            "Not enough RR intervals for HRV analysis",
            ha="center",
            va="center",
            transform=plt.gca().transAxes
        )

    plt.title("Basic Time-Domain HRV Analysis")
    plt.xlabel("RR Interval Number")
    plt.ylabel("RR Interval (s)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Images/Figure_4.png")
    plt.close()

    print("HRV figure saved to Images/Figure_4.png")
