import numpy as np
import matplotlib.pyplot as plt


def gaussian_wave(t, center, amplitude, width):
    return amplitude * np.exp(-((t - center) ** 2) / (2 * width ** 2))


def generate_ecg(duration_sec, fs):
    """
    Generate a simple synthetic ECG-like signal using Gaussian waves.

    Parameters:
        duration_sec (int): Signal duration in seconds
        fs (int): Sampling frequency in Hz

    Returns:
        tuple: time array and synthetic ECG signal
    """
    t = np.linspace(0, duration_sec, duration_sec * fs, endpoint=False)
    ecg = np.zeros_like(t)

    beat_times = np.arange(0.8, duration_sec, 1.0)

    for beat in beat_times:
        ecg += gaussian_wave(t, beat - 0.2, 0.12, 0.025)   # P wave
        ecg += gaussian_wave(t, beat - 0.04, -0.15, 0.008) # Q wave
        ecg += gaussian_wave(t, beat, 1.2, 0.01)           # R wave
        ecg += gaussian_wave(t, beat + 0.04, -0.25, 0.01)  # S wave
        ecg += gaussian_wave(t, beat + 0.28, 0.35, 0.06)   # T wave

    baseline = 0.03 * np.sin(2 * np.pi * 0.3 * t)
    ecg += baseline

    return t, ecg


def plot_ecg(time, signal):
    """
    Plot the ECG signal.
    """
    plt.figure(figsize=(12, 4))
    plt.plot(time, signal, color="red", linewidth=1.5)
    plt.title("Synthetic ECG Signal")
    plt.xlabel("Time (sec)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    fs = 500
    duration = 5

    time, signal = generate_ecg(duration, fs)
    plot_ecg(time, signal)


if __name__ == "__main__":
    main()
