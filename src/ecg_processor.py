import numpy as np
import matplotlib.pyplot as plt

def generate_ecg(duration_sec, fs):
    """تولید یک سیگنال ECG مصنوعی ساده"""
    t = np.linspace(0, duration_sec, duration_sec * fs)
    # یک موج سینوسی پایه به همراه نویز و پیک‌های فرضی
    ecg = 0.5 * np.sin(2 * np.pi * 1.5 * t)  # ضربان پایه
    
    # اضافه کردن پیک‌های R (به صورت ساده برای تست)
    for i in range(1, duration_sec):
        idx = i * fs
        ecg[idx:idx+10] += 2.0
        
    return t, ecg

# تنظیمات
fs = 500  # فرکانس نمونه‌برداری
duration = 5 # ثانیه

time, signal = generate_ecg(duration, fs)

# رسم نمودار
plt.figure(figsize=(12, 4))
plt.plot(time, signal, color='red', linewidth=1)
plt.title('Synthetic ECG Signal Test')
plt.xlabel('Time (sec)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()
