# ECG Signal Analyzer

An introductory ECG signal processing project developed for learning, practice, and academic coursework.

## Features
- Generate a synthetic ECG-like signal with simple P-Q-R-S-T morphology
- Add Gaussian noise to simulate a noisy acquisition
- Apply basic low-pass filtering (Butterworth) using SciPy
- Visualize clean vs noisy vs filtered signals
- Detect R-peaks from the filtered ECG signal
- Estimate heart rate in beats per minute (BPM)
- Extract basic ECG features from detected R-peaks
- Compute RR interval statistics:
  - Mean RR interval
  - Standard deviation of RR intervals
  - Minimum RR interval
  - Maximum RR interval
    
## Project Structure
- `src/`: source code
- `Images/`: output figures and results
- `data/`: extracted feature files

## Requirements
Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
```bash
python src/ecg_processor.py
```
The script generates and saves the following outputs:

- `Images/Figure_1.png` → clean vs noisy vs filtered ECG comparison
- `Images/Figure_2.png` → detected R-peaks and estimated heart rate
- `Images/Figure_3.png` → RR interval analysis and R-peak amplitudes
- `data/ecg_features.csv` → extracted ECG feature values

## Feature Extraction Output
The current version of the project extracts the following ECG features:
- R-peak time positions
- R-peak amplitudes
- RR intervals
- Mean RR interval
- RR interval standard deviation
- Minimum RR interval
- Maximum RR interval

These extracted values are saved in:
data/ecg_features.csv




## Sample Output
## Figure 1 - Signal Comparison
![Synthetic ECG Output](Images/Figure_1.png)

## Figure 2 - R-peak Detection
![R-peak Detection](Images/Figure_2.png)

## Figure 3 - Feature Extraction
![Feature Extraction](Images/Figure_3.png)

## Project Context
Developed for academic practice in Biomedical Signal Processing (2023).
