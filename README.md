# MNE ICA GUI

A PyQt5-based graphical interface for performing Independent Component Analysis (ICA) on MNE Raw data. Designed to be called directly from your Python scripts.

## Requirements

```
mne
PyQt5
matplotlib
numpy
scipy        # optional, for better PSD computation
```

Install with:
```bash
pip install mne PyQt5 matplotlib numpy scipy
```

## Quick Start

```python
import mne
from ica_gui import launch_ica_gui

# Load and preprocess your data
raw = mne.io.read_raw_fif("my_data.fif", preload=True)
raw.filter(1.0, 40.0)

# Launch the GUI — blocks until the window is closed
ica, cleaned_raw = launch_ica_gui(raw)

# Use the results
print(f"Excluded components: {ica.exclude}")
if cleaned_raw is not None:
    # Continue your pipeline with cleaned_raw
    ...
```

### With a pre-fitted ICA

```python
from mne.preprocessing import ICA

ica = ICA(n_components=20, method="fastica", random_state=42)
ica.fit(raw)

# Open the GUI with the existing ICA — skips straight to visualization
ica, cleaned_raw = launch_ica_gui(raw, ica=ica)
```

## Features

| Tab | Description |
|---|---|
| **⚙ Fit ICA** | Configure method, n_components, max_iter, decimation, optional pre-filtering |
| **🗺 Topomaps** | Grid of component topographic maps — click to toggle exclusion |
| **📈 Sources** | Scrollable time-series of ICA sources with time/component navigation |
| **🔬 Properties** | Per-component detail: topomap, power spectrum, time course, kurtosis |
| **✓ Apply/Export** | Before/after preview, apply ICA, save cleaned Raw or ICA solution to `.fif` |

**Exclusion sidebar** (always visible):
- Manual add/remove of components
- Auto-detect EOG artifacts (`find_bads_eog`)
- Auto-detect ECG artifacts (`find_bads_ecg`)
- Adjustable detection threshold
- Select specific EOG/ECG channel names

## API

```python
launch_ica_gui(raw, ica=None) -> (ica, cleaned_raw)
```

| Parameter | Type | Description |
|---|---|---|
| `raw` | `mne.io.Raw` | Preloaded raw data |
| `ica` | `ICA` or `None` | Optional pre-fitted ICA object |
| **Returns** | `(ICA, Raw or None)` | The ICA with `.exclude` set, and cleaned Raw if applied |
