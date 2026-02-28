"""
MNE ICA GUI — A Qt-based graphical interface for ICA analysis on MNE Raw data.
Supports PySide6 and PyQt5 (auto-detected to match matplotlib's backend).

Usage:
    from ica_gui import launch_ica_gui
    ica, cleaned_raw = launch_ica_gui(raw)
"""
import sys, warnings, traceback
from functools import partial
import numpy as np
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import mne
from mne.preprocessing import ICA

# --- Qt compat layer ---
_QT = None
try:
    _m = sys.modules.get("matplotlib.backends.backend_qt")
    if _m and hasattr(_m, "QtWidgets"):
        _n = _m.QtWidgets.__name__.split(".")[0]
        if "PySide6" in _n: _QT = "PySide6"
        elif "PyQt5" in _n: _QT = "PyQt5"
        elif "PyQt6" in _n: _QT = "PyQt6"
except Exception: pass
if _QT is None:
    try: import PySide6; _QT = "PySide6"
    except ImportError:
        try: import PyQt5; _QT = "PyQt5"
        except ImportError: raise ImportError("Install PySide6 or PyQt5")

if _QT == "PySide6":
    from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QTabWidget,
        QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout, QGroupBox, QLabel,
        QPushButton, QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox, QListWidget,
        QListWidgetItem, QSplitter, QScrollArea, QSlider, QProgressBar, QTextEdit,
        QFileDialog, QMessageBox, QStatusBar, QFrame, QSizePolicy, QAbstractItemView)
    from PySide6.QtCore import Qt, QThread, Signal as pyqtSignal, QTimer, QEventLoop
    from PySide6.QtGui import QFont, QColor, QIcon
elif _QT == "PyQt5":
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QTabWidget,
        QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout, QGroupBox, QLabel,
        QPushButton, QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox, QListWidget,
        QListWidgetItem, QSplitter, QScrollArea, QSlider, QProgressBar, QTextEdit,
        QFileDialog, QMessageBox, QStatusBar, QFrame, QSizePolicy, QAbstractItemView)
    from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QEventLoop
    from PyQt5.QtGui import QFont, QColor, QIcon
else:
    from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QTabWidget,
        QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout, QGroupBox, QLabel,
        QPushButton, QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox, QListWidget,
        QListWidgetItem, QSplitter, QScrollArea, QSlider, QProgressBar, QTextEdit,
        QFileDialog, QMessageBox, QStatusBar, QFrame, QSizePolicy, QAbstractItemView)
    from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QEventLoop
    from PyQt6.QtGui import QFont, QColor, QIcon

# --- ICLabel support ---
try:
    from mne_icalabel import label_components as _label_components
    HAS_ICALABEL = True
except ImportError:
    HAS_ICALABEL = False

# --- ICA fit worker thread ---
class _ICAFitWorker(QThread):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)
    def __init__(self, raw, method, n_components, max_iter, random_state, decim):
        super().__init__()
        self.raw, self.method = raw, method
        self.n_components, self.max_iter = n_components, max_iter
        self.random_state, self.decim = random_state, decim
    def run(self):
        try:
            self.progress.emit("Creating ICA object…")
            ica = ICA(n_components=self.n_components, method=self.method,
                      max_iter=self.max_iter, random_state=self.random_state)
            self.progress.emit(f"Fitting ICA ({self.method}, n={self.n_components})…")
            ica.fit(self.raw, decim=self.decim)
            self.progress.emit("Fitting complete.")
            self.finished.emit(ica)
        except Exception as exc:
            self.error.emit(f"{exc}\n{traceback.format_exc()}")

# --- Matplotlib canvas widget ---
class _MplCanvas(QWidget):
    def __init__(self, parent=None, width=6, height=4, dpi=100, toolbar=False):
        super().__init__(parent)
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.canvas = FigureCanvas(self.fig)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        if toolbar:
            self.toolbar = NavigationToolbar(self.canvas, self)
            layout.addWidget(self.toolbar)
        else:
            self.toolbar = None
        layout.addWidget(self.canvas)
    def clear(self):
        self.fig.clear(); self.canvas.draw_idle()
    def draw(self):
        self.fig.tight_layout(); self.canvas.draw_idle()

# ====================== 1. FIT TAB ======================
class FitTab(QWidget):
    ica_fitted = pyqtSignal(object)
    def __init__(self, raw, parent=None):
        super().__init__(parent)
        self.raw = raw; self._worker = None; self._build_ui()
    def _build_ui(self):
        layout = QVBoxLayout(self)
        # Parameters
        pg = QGroupBox("ICA Parameters"); form = QFormLayout()
        self.method_combo = QComboBox()
        self.method_combo.addItems(["fastica", "infomax", "picard"])
        form.addRow("Method:", self.method_combo)
        self.n_comp_spin = QSpinBox(); self.n_comp_spin.setRange(1, 500); self.n_comp_spin.setValue(20)
        form.addRow("n_components:", self.n_comp_spin)
        self.max_iter_spin = QSpinBox(); self.max_iter_spin.setRange(100, 10000)
        self.max_iter_spin.setValue(1000); self.max_iter_spin.setSingleStep(100)
        form.addRow("max_iter:", self.max_iter_spin)
        self.seed_spin = QSpinBox(); self.seed_spin.setRange(0, 99999); self.seed_spin.setValue(42)
        form.addRow("Random seed:", self.seed_spin)
        self.decim_spin = QSpinBox(); self.decim_spin.setRange(1, 50); self.decim_spin.setValue(3)
        form.addRow("Decimation:", self.decim_spin)
        pg.setLayout(form); layout.addWidget(pg)
        # Pre-filtering
        fg = QGroupBox("Pre-filtering (before fit)"); ff = QFormLayout()
        self.filter_check = QCheckBox("Apply band-pass filter before fitting"); self.filter_check.setChecked(False)
        ff.addRow(self.filter_check)
        self.hp_spin = QDoubleSpinBox(); self.hp_spin.setRange(0, 50); self.hp_spin.setValue(1.0)
        self.hp_spin.setSingleStep(0.5); self.hp_spin.setDecimals(1); ff.addRow("High-pass (Hz):", self.hp_spin)
        self.lp_spin = QDoubleSpinBox(); self.lp_spin.setRange(1, 500); self.lp_spin.setValue(40.0)
        self.lp_spin.setSingleStep(5); self.lp_spin.setDecimals(1); ff.addRow("Low-pass (Hz):", self.lp_spin)
        fg.setLayout(ff); layout.addWidget(fg)
        # Fit button
        self.fit_btn = QPushButton("▶  Fit ICA"); self.fit_btn.setMinimumHeight(40)
        self.fit_btn.setStyleSheet(
            "QPushButton{background:#2d7d46;color:white;font-weight:bold;font-size:14px;border-radius:6px}"
            "QPushButton:hover{background:#36964f}QPushButton:disabled{background:#555}")
        self.fit_btn.clicked.connect(self._on_fit); layout.addWidget(self.fit_btn)
        self.progress_bar = QProgressBar(); self.progress_bar.setRange(0, 0); self.progress_bar.hide()
        layout.addWidget(self.progress_bar)
        self.log = QTextEdit(); self.log.setReadOnly(True); self.log.setMaximumHeight(180)
        self.log.setStyleSheet("background:#1e1e1e;color:#d4d4d4;font-family:Consolas,monospace")
        layout.addWidget(self.log); layout.addStretch()
    def _log(self, msg): self.log.append(msg)
    def _on_fit(self):
        self.fit_btn.setEnabled(False); self.progress_bar.show()
        raw_to_fit = self.raw.copy()
        if self.filter_check.isChecked():
            hp, lp = self.hp_spin.value(), self.lp_spin.value()
            self._log(f"Filtering copy: {hp}–{lp} Hz …"); raw_to_fit.filter(hp, lp, verbose=False)
        self._worker = _ICAFitWorker(raw_to_fit, self.method_combo.currentText(),
            self.n_comp_spin.value(), self.max_iter_spin.value(),
            self.seed_spin.value(), self.decim_spin.value())
        self._worker.progress.connect(self._log)
        self._worker.finished.connect(self._on_fit_done)
        self._worker.error.connect(self._on_fit_error); self._worker.start()
    def _on_fit_done(self, ica):
        self.progress_bar.hide(); self.fit_btn.setEnabled(True)
        self._log(f"✓ ICA fitted — {ica.n_components_} components extracted.")
        self.ica_fitted.emit(ica)
    def _on_fit_error(self, msg):
        self.progress_bar.hide(); self.fit_btn.setEnabled(True)
        self._log(f"✗ Error:\n{msg}")

# ====================== 2. TOPOMAP TAB ======================
class TopomapTab(QWidget):
    exclusion_changed = pyqtSignal()
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ica = self.raw = None; self._n_cols = 5; self._current_page = 0; self._per_page = 20
        self._build_ui()
    def _build_ui(self):
        layout = QVBoxLayout(self)
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Cols:"))
        self.cols_spin = QSpinBox(); self.cols_spin.setRange(2, 10); self.cols_spin.setValue(5)
        self.cols_spin.valueChanged.connect(self._refresh); ctrl.addWidget(self.cols_spin)
        ctrl.addWidget(QLabel("Per page:"))
        self.pp_spin = QSpinBox(); self.pp_spin.setRange(4, 60); self.pp_spin.setValue(20)
        self.pp_spin.setSingleStep(4); self.pp_spin.valueChanged.connect(self._refresh); ctrl.addWidget(self.pp_spin)
        self.prev_btn = QPushButton("◀ Prev"); self.prev_btn.clicked.connect(lambda: self._page(-1)); ctrl.addWidget(self.prev_btn)
        self.page_lbl = QLabel("1/1"); ctrl.addWidget(self.page_lbl)
        self.next_btn = QPushButton("Next ▶"); self.next_btn.clicked.connect(lambda: self._page(1)); ctrl.addWidget(self.next_btn)
        ctrl.addStretch(); layout.addLayout(ctrl)
        self.mpl = _MplCanvas(self, width=10, height=7, toolbar=True)
        self.mpl.canvas.mpl_connect("button_press_event", self._on_click); layout.addWidget(self.mpl)
    def set_data(self, ica, raw): self.ica = ica; self.raw = raw; self._current_page = 0; self._refresh()
    def _pages(self):
        if not self.ica: return 1
        return max(1, int(np.ceil(self.ica.n_components_ / self.pp_spin.value())))
    def _page(self, d): self._current_page = max(0, min(self._current_page + d, self._pages() - 1)); self._refresh()
    def _refresh(self):
        if not self.ica: return
        nc = self.cols_spin.value(); pp = self.pp_spin.value()
        n = self.ica.n_components_; s = self._current_page * pp; e = min(s + pp, n)
        picks = list(range(s, e)); nr = int(np.ceil(len(picks) / nc))
        self.mpl.fig.clear(); self._axes_map = {}
        for idx, comp in enumerate(picks):
            ax = self.mpl.fig.add_subplot(nr, nc, idx + 1)
            try:
                data = self.ica.get_components()[:, comp]
                mne.viz.plot_topomap(data, self.ica.info, axes=ax, show=False)
                ax.set_title(f"IC {comp}", fontsize=9)
            except Exception:
                ax.text(0.5, 0.5, f"IC {comp}", ha="center", va="center", transform=ax.transAxes)
            if comp in self.ica.exclude:
                for sp in ax.spines.values(): sp.set_edgecolor("red"); sp.set_linewidth(3)
                ax.set_title(f"IC {comp} [X]", fontsize=9, color="red", fontweight="bold")
            self._axes_map[ax] = comp
        self.page_lbl.setText(f"{self._current_page+1}/{self._pages()}"); self.mpl.draw()
    def _on_click(self, event):
        if event.inaxes is None or not self.ica: return
        comp = self._axes_map.get(event.inaxes)
        if comp is None: return
        if comp in self.ica.exclude: self.ica.exclude.remove(comp)
        else: self.ica.exclude.append(comp)
        self.exclusion_changed.emit(); self._refresh()

# ====================== 3. SOURCES TAB ======================
class SourcesTab(QWidget):
    exclusion_changed = pyqtSignal()
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ica = self.raw = self._sources = None
        self._n_display = 10; self._offset = 0; self._t_start = 0.0; self._t_win = 10.0
        self._build_ui()
    def _build_ui(self):
        layout = QVBoxLayout(self)
        ctrl = QHBoxLayout(); ctrl.addWidget(QLabel("Show:"))
        self.n_spin = QSpinBox(); self.n_spin.setRange(1, 60); self.n_spin.setValue(10)
        self.n_spin.valueChanged.connect(self._params); ctrl.addWidget(self.n_spin)
        ctrl.addWidget(QLabel("Window (s):"))
        self.win_spin = QDoubleSpinBox(); self.win_spin.setRange(0.5, 300); self.win_spin.setValue(10)
        self.win_spin.valueChanged.connect(self._params); ctrl.addWidget(self.win_spin)
        ctrl.addStretch(); layout.addLayout(ctrl)
        tr = QHBoxLayout(); tr.addWidget(QLabel("Time:"))
        self.t_slider = QSlider(Qt.Horizontal); self.t_slider.setRange(0, 1000)
        self.t_slider.valueChanged.connect(self._on_t); tr.addWidget(self.t_slider)
        self.t_lbl = QLabel("0.0 s"); tr.addWidget(self.t_lbl); layout.addLayout(tr)
        cr = QHBoxLayout(); cr.addWidget(QLabel("Components ↕:"))
        self.c_slider = QSlider(Qt.Horizontal); self.c_slider.setRange(0, 0)
        self.c_slider.valueChanged.connect(self._on_c); cr.addWidget(self.c_slider)
        self.c_lbl = QLabel("0–9"); cr.addWidget(self.c_lbl); layout.addLayout(cr)
        self.mpl = _MplCanvas(self, width=12, height=6, toolbar=True)
        self.mpl.canvas.mpl_connect("button_press_event", self._on_click); layout.addWidget(self.mpl)
    def set_data(self, ica, raw):
        self.ica = ica; self.raw = raw; self._sources = ica.get_sources(raw)
        self._t_start = 0.0; self._offset = 0
        self.c_slider.setRange(0, max(0, ica.n_components_ - self.n_spin.value()))
        self.t_slider.setRange(0, int(raw.times[-1] * 10)); self._refresh()
    def _params(self):
        self._n_display = self.n_spin.value(); self._t_win = self.win_spin.value()
        if self.ica: self.c_slider.setRange(0, max(0, self.ica.n_components_ - self._n_display))
        self._refresh()
    def _on_t(self, v):
        self._t_start = v / 10.0; self.t_lbl.setText(f"{self._t_start:.1f} s"); self._refresh()
    def _on_c(self, v):
        self._offset = v
        end = min(self._offset + self._n_display, self.ica.n_components_ if self.ica else 0) - 1
        self.c_lbl.setText(f"{self._offset}–{end}"); self._refresh()
    def _refresh(self):
        if self._sources is None or not self.ica: return
        self.mpl.fig.clear(); ax = self.mpl.fig.add_subplot(111)
        comps = list(range(self._offset, min(self._offset + self._n_display, self.ica.n_components_)))
        sfreq = self.raw.info["sfreq"]; t0 = self._t_start; t1 = min(t0 + self._t_win, self.raw.times[-1])
        s0, s1 = int(t0 * sfreq), int(t1 * sfreq)
        if s1 <= s0: return
        data = self._sources.get_data(start=s0, stop=s1); times = np.linspace(t0, t1, s1 - s0)
        sub = data[comps]
        spacing = np.max(np.ptp(sub, axis=1)) * 1.2 if sub.size > 0 and np.max(np.ptp(sub, axis=1)) > 0 else 1.0
        self._lpos = {}
        for i, comp in enumerate(comps):
            yo = -i * spacing; color = "red" if comp in self.ica.exclude else "#4fc3f7"
            ax.plot(times, data[comp] + yo, color=color, linewidth=0.6)
            lbl = f"IC {comp}" + (" [X]" if comp in self.ica.exclude else "")
            ax.text(t0 - (t1 - t0) * 0.02, yo, lbl, ha="right", va="center", fontsize=8,
                    color="red" if comp in self.ica.exclude else "white",
                    fontweight="bold" if comp in self.ica.exclude else "normal")
            self._lpos[comp] = yo
        ax.set_xlim(t0, t1); ax.set_xlabel("Time (s)"); ax.set_yticks([])
        ax.set_facecolor("#1a1a2e"); self.mpl.fig.patch.set_facecolor("#1a1a2e")
        ax.tick_params(colors="white"); ax.xaxis.label.set_color("white")
        for sp in ax.spines.values(): sp.set_color("#444")
        self.mpl.draw()
    def _on_click(self, event):
        if event.inaxes is None or not self.ica: return
        y = event.ydata; best_c, best_d = None, float("inf")
        for c, yp in self._lpos.items():
            d = abs(y - yp)
            if d < best_d: best_d = d; best_c = c
        if best_c is not None:
            if best_c in self.ica.exclude: self.ica.exclude.remove(best_c)
            else: self.ica.exclude.append(best_c)
            self.exclusion_changed.emit(); self._refresh()

# ====================== 4. PROPERTIES TAB ======================
class PropertiesTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent); self.ica = self.raw = None; self._build_ui()
    def _build_ui(self):
        layout = QVBoxLayout(self)
        ctrl = QHBoxLayout(); ctrl.addWidget(QLabel("Component:"))
        self.comp_spin = QSpinBox(); self.comp_spin.setRange(0, 0)
        self.comp_spin.valueChanged.connect(self._refresh); ctrl.addWidget(self.comp_spin)
        self.excl_lbl = QLabel(""); self.excl_lbl.setStyleSheet("color:red;font-weight:bold")
        ctrl.addWidget(self.excl_lbl); ctrl.addStretch(); layout.addLayout(ctrl)
        cr = QHBoxLayout()
        self.topo_c = _MplCanvas(self, width=3.5, height=3.5); cr.addWidget(self.topo_c)
        self.psd_c = _MplCanvas(self, width=4.5, height=3.5); cr.addWidget(self.psd_c)
        layout.addLayout(cr)
        self.ts_c = _MplCanvas(self, width=12, height=3, toolbar=True); layout.addWidget(self.ts_c)
        self.stats_lbl = QLabel(""); self.stats_lbl.setStyleSheet("font-family:Consolas;font-size:12px")
        self.stats_lbl.setWordWrap(True); layout.addWidget(self.stats_lbl)
    def set_data(self, ica, raw):
        self.ica = ica; self.raw = raw; self.comp_spin.setRange(0, ica.n_components_ - 1); self._refresh()
    def _refresh(self):
        if not self.ica or not self.raw: return
        comp = self.comp_spin.value()
        self.excl_lbl.setText("⚠ EXCLUDED" if comp in self.ica.exclude else "")
        # Topomap
        self.topo_c.fig.clear(); ax = self.topo_c.fig.add_subplot(111)
        try:
            mne.viz.plot_topomap(self.ica.get_components()[:, comp], self.ica.info, axes=ax, show=False)
        except Exception: ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"IC {comp} Topomap", fontsize=10); self.topo_c.draw()
        # PSD
        self.psd_c.fig.clear(); ax = self.psd_c.fig.add_subplot(111)
        try:
            src = self.ica.get_sources(self.raw).get_data(picks=[comp])[0]
            sfreq = self.raw.info["sfreq"]
            try:
                from scipy.signal import welch
                f, p = welch(src, fs=sfreq, nperseg=min(int(sfreq * 2), len(src)))
            except ImportError:
                ns = min(int(sfreq * 2), len(src))
                fv = np.fft.rfft(src[:ns]); f = np.fft.rfftfreq(ns, 1.0 / sfreq)
                p = np.abs(fv) ** 2 / ns
            ax.semilogy(f, p, color="#4fc3f7", linewidth=1)
            ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("PSD")
            ax.set_title(f"IC {comp} Power Spectrum", fontsize=10)
            ax.set_xlim(0, min(sfreq / 2, 80)); ax.grid(True, alpha=0.3)
        except Exception: ax.text(0.5, 0.5, "PSD N/A", ha="center", va="center", transform=ax.transAxes)
        self.psd_c.draw()
        # Time series
        self.ts_c.fig.clear(); ax = self.ts_c.fig.add_subplot(111)
        try:
            sources = self.ica.get_sources(self.raw); sfreq = self.raw.info["sfreq"]
            ns = min(int(10 * sfreq), sources.n_times)
            td = sources.get_data(picks=[comp], start=0, stop=ns)[0]; t = np.arange(ns) / sfreq
            ax.plot(t, td, color="red" if comp in self.ica.exclude else "#4fc3f7", linewidth=0.5)
            ax.set_xlabel("Time (s)"); ax.set_title(f"IC {comp} (first {ns/sfreq:.1f}s)", fontsize=10)
            ax.set_facecolor("#1a1a2e"); self.ts_c.fig.patch.set_facecolor("#1a1a2e")
            ax.tick_params(colors="white"); ax.xaxis.label.set_color("white")
            for sp in ax.spines.values(): sp.set_color("#444")
        except Exception: ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
        self.ts_c.draw()
        # Stats
        try:
            sd = self.ica.get_sources(self.raw).get_data(picks=[comp])[0]
            v = np.var(sd); k = float(np.mean(((sd - sd.mean()) / sd.std()) ** 4) - 3)
            self.stats_lbl.setText(f"IC {comp} | Var: {v:.6e} | Kurt: {k:.2f} | {'EXCLUDED' if comp in self.ica.exclude else 'Included'}")
        except Exception: self.stats_lbl.setText("")
