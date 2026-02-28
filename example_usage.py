"""
Example usage of the MNE ICA GUI.

This script shows how to call launch_ica_gui() from your own code.
Replace the synthetic data below with your actual Raw instance.
"""

import mne
from ica_gui import launch_ica_gui

# ---------------------------------------------------------------
# Option A: Load your own data
# ---------------------------------------------------------------
# raw = mne.io.read_raw_fif("your_data.fif", preload=True)
# raw.filter(1.0, 40.0)  # recommended before ICA

# ---------------------------------------------------------------
# Option B: Synthetic demo data (for testing)
# ---------------------------------------------------------------
import numpy as np

rng = np.random.RandomState(42)
n_channels = 32
sfreq = 256.0
duration = 60  # seconds
n_samples = int(sfreq * duration)

# Create channel info with standard 10-20 montage
montage = mne.channels.make_standard_montage("standard_1020")
ch_names = montage.ch_names[:n_channels]
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")

# Generate synthetic signals
t = np.arange(n_samples) / sfreq
data = rng.randn(n_channels, n_samples) * 1e-6

# Inject some structured components
data[0] += 5e-6 * np.sin(2 * np.pi * 10 * t)    # ~10 Hz alpha
data[1] += 8e-6 * np.sin(2 * np.pi * 1.0 * t)   # ~1 Hz slow (blink-like)
data[2] += 3e-6 * np.sin(2 * np.pi * 50 * t)     # 50 Hz line noise
data[3] += 4e-6 * np.sin(2 * np.pi * 1.2 * t)    # ~1.2 Hz (heartbeat-like)

raw = mne.io.RawArray(data, info)
raw.set_montage(montage, match_case=False)

# ---------------------------------------------------------------
# Launch the ICA GUI
# ---------------------------------------------------------------
ica, cleaned_raw = launch_ica_gui(raw)

# ---------------------------------------------------------------
# Use the results
# ---------------------------------------------------------------
if ica is not None:
    print(f"\nICA method: {ica.method}")
    print(f"Components: {ica.n_components_}")
    print(f"Excluded:   {ica.exclude}")

if cleaned_raw is not None:
    print(f"Cleaned Raw: {cleaned_raw.info['nchan']} channels, {cleaned_raw.n_times} samples")
    # Continue your analysis pipeline with cleaned_raw...
else:
    print("No cleaned data returned (ICA was not applied or window was closed).")
