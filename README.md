# ECG Signal Analyzer

An introductory ECG signal processing project developed for learning, practice, and academic coursework.

## Features
- Generate a synthetic ECG-like signal (simple P-Q-R-S-T morphology)
- Add Gaussian noise to simulate a noisy acquisition
- Apply basic low-pass filtering (Butterworth) using SciPy
- Visualize clean vs noisy vs filtered signals

## Project Structure
- `src/`: source code
- `Images/`: output figures and results

## Requirements
Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
Run the script:
```bash
python src/ecg_processor.py
```
The script saves an output figure to `Images/Figure_1.png`.

## Sample Output
![Synthetic ECG Output](Images/Figure_1.png)

## Project Context
Developed for academic practice in Biomedical Signal Processing (2023).
