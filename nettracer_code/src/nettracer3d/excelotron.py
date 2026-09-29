"""
excelotron.py - Spreadsheet -> NetTracer3D property importer.

Loads a .csv/.xlsx exported from an external program (QuPath, CellProfiler, HALO,
etc.) and turns selected columns into a NetTracer3D node property dictionary.

Supported properties
--------------------
Node Identities   : three import styles (see below)
Node Centroids    : ID + Z/Y/X columns
Node Communities  : ID + community column

Node Identity import styles
---------------------------
1. Single identity column
       one column of text labels; each node gets that one identity.
2. Identity matrix (0/1)
       several columns, one per marker; column header is the identity name and a
       1 means the node carries it. Nodes may end up with several identities.
3. Raw intensity columns
       several numeric columns; optionally z-scored, then thresholded one marker
       at a time in a histogram GUI. Values at/above the threshold are positive.

Emitted data
------------
`data_exported(result_dict, property_name, add_flag)` keeps the original contract:
parallel lists keyed by the template field names, e.g.

    {'Numerical IDs': [1, 2, 3], 'Identity Column': ['CD3', ['CD3', 'CD20'], 'CD20']}

For convenience `identities_exported(identity_dict, add_flag)` is also emitted for
Node Identities, carrying the plain {node_id: [identity, ...]} mapping.
Set ALWAYS_LIST = True below if NetTracer3D would rather always receive lists.
"""

import sys
import os
import math
import builtins

import numpy as np
import pandas as pd

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QFont
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QDialog, QWidget, QFrame, QSplitter,
    QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QPushButton, QLineEdit, QComboBox, QCheckBox, QRadioButton,
    QButtonGroup, QListWidget, QListWidgetItem, QTableWidget, QTableWidgetItem,
    QStackedWidget, QScrollArea, QMessageBox, QFileDialog, QTextEdit,
    QAbstractItemView, QGroupBox, QSizePolicy
)

try:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    MATPLOTLIB_AVAILABLE = True
except Exception:  # pragma: no cover - matplotlib is optional
    MATPLOTLIB_AVAILABLE = False


# If True, every node's identity value is a list, even when it has only one.
ALWAYS_LIST = False

MAX_PREVIEW_ROWS = 200
MAX_HIST_SAMPLES = 2_000_000

MODE_SINGLE = 0
MODE_MATRIX = 1
MODE_INTENSITY = 2


# --------------------------------------------------------------------------- #
#  helpers
# --------------------------------------------------------------------------- #

def _is_nan_or_empty(val):
    """True for None, NaN, blank strings, 'nan', and lists made only of those."""
    if val is None:
        return True
    if isinstance(val, float) and math.isnan(val):
        return True
    if isinstance(val, str):
        s = val.strip()
        return s == '' or s.lower() == 'nan'
    if isinstance(val, (list, tuple)):
        if len(val) == 0:
            return True
        return all(_is_nan_or_empty(v) for v in val)
    return False


def _coerce_node_id(val):
    """Node IDs are label values in a segmentation, so prefer ints."""
    try:
        f = float(val)
    except (TypeError, ValueError):
        return str(val).strip()
    if math.isnan(f):
        return None
    if f.is_integer():
        return int(f)
    return f


def _numeric_column(series):
    """Series -> float ndarray, non-numeric entries become NaN."""
    return pd.to_numeric(series, errors='coerce').to_numpy(dtype=float)


def _looks_binary(series):
    """True if the column only holds 0/1 (or False/True) plus blanks."""
    vals = pd.unique(series.dropna())
    if len(vals) == 0 or len(vals) > 3:
        return False
    allowed = {0, 1, 0.0, 1.0, True, False, '0', '1'}
    for v in vals:
        if isinstance(v, str):
            v = v.strip()
        if v not in allowed:
            return False
    return True


def _looks_numeric(series):
    conv = pd.to_numeric(series, errors='coerce')
    return conv.notna().sum() >= max(1, int(0.5 * len(series)))


def _truthy(val):
    """Positive call for a 0/1 matrix cell."""
    if _is_nan_or_empty(val):
        return False
    if isinstance(val, str):
        s = val.strip().lower()
        if s in ('1', 'true', 'yes', 'y', 'pos', 'positive', '+'):
            return True
        if s in ('0', 'false', 'no', 'n', 'neg', 'negative', '-'):
            return False
        try:
            return float(s) >= 0.5
        except ValueError:
            return False
    try:
        return float(val) >= 0.5
    except (TypeError, ValueError):
        return False


def _zscore(arr):
    """Z-score a float array, ignoring NaNs. Flat columns come back as zeros."""
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return arr
    mu = float(np.mean(finite))
    sd = float(np.std(finite))
    if sd == 0:
        return np.where(np.isfinite(arr), 0.0, arr)
    return (arr - mu) / sd


def _otsu_threshold(values, nbins=256):
    """Otsu's method on a 1D array; used as the default intensity cut."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0
    if np.all(v == v[0]):
        return float(v[0])
    counts, edges = np.histogram(v, bins=nbins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    w = counts.astype(float)
    total = w.sum()
    if total == 0:
        return float(np.mean(v))
    omega = np.cumsum(w) / total
    mu = np.cumsum(w * centers) / total
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    with np.errstate(divide='ignore', invalid='ignore'):
        sigma_b = ((mu_t * omega - mu) ** 2) / denom
    sigma_b[~np.isfinite(sigma_b)] = -1.0
    return float(centers[int(np.argmax(sigma_b))])


def _guess_column(columns, keywords, exclude=()):
    """Best-effort header match, used to pre-fill the pickers on load."""
    lowered = [(str(c), str(c).strip().lower()) for c in columns]
    for kw in keywords:
        for original, low in lowered:
            if original in exclude:
                continue
            if low == kw:
                return original
    for kw in keywords:
        for original, low in lowered:
            if original in exclude:
                continue
            if kw in low:
                return original
    return None


def _fmt(value, places=4):
    try:
        return f"{float(value):.{places}g}"
    except (TypeError, ValueError):
        return str(value)


# --------------------------------------------------------------------------- #
#  small reusable widgets
# --------------------------------------------------------------------------- #

class DropZoneWidget(QFrame):
    """Click-or-drop target for a spreadsheet file."""

    file_dropped = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(64)
        self.setStyleSheet("""
            QFrame {
                border: 2px dashed #aaa;
                border-radius: 6px;
                background-color: #fafafa;
            }
            QFrame:hover {
                border-color: #007acc;
                background-color: #f0f8ff;
            }
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        self.label = QLabel("Click to browse, or drop a .csv / .xlsx here")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet("border: none; color: #555; font-size: 13px;")
        layout.addWidget(self.label)

    def set_message(self, text):
        self.label.setText(text)

    def mousePressEvent(self, event):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open spreadsheet", "", "Spreadsheets (*.csv *.xlsx *.xls);;All files (*)"
        )
        if path:
            self.file_dropped.emit(path)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if len(urls) == 1 and urls[0].toLocalFile().lower().endswith(('.xlsx', '.xls', '.csv')):
                event.acceptProposedAction()
                return
        event.ignore()

    def dropEvent(self, event: QDropEvent):
        if event.mimeData().hasUrls():
            self.file_dropped.emit(event.mimeData().urls()[0].toLocalFile())
            event.acceptProposedAction()


class ColumnCombo(QComboBox):
    """Column picker. Index 0 is a placeholder / '(none)' entry."""

    def __init__(self, placeholder="Select column...", parent=None):
        super().__init__(parent)
        self.placeholder = placeholder
        self.setMinimumWidth(180)
        self.set_columns([])

    def set_columns(self, columns):
        current = self.current_column()
        self.blockSignals(True)
        self.clear()
        self.addItem(self.placeholder)
        for col in columns:
            self.addItem(str(col))
        if current is not None:
            idx = self.findText(str(current))
            if idx >= 0:
                self.setCurrentIndex(idx)
        self.blockSignals(False)

    def current_column(self):
        if self.currentIndex() <= 0:
            return None
        return self.currentText()

    def select(self, column):
        if column is None:
            self.setCurrentIndex(0)
            return
        idx = self.findText(str(column))
        self.setCurrentIndex(idx if idx >= 0 else 0)


class ColumnCheckList(QWidget):
    """Filterable checkbox list of columns, with select-all / auto-detect."""

    selection_changed = pyqtSignal()

    def __init__(self, auto_label="Auto-detect", parent=None):
        super().__init__(parent)
        self._auto_predicate = None
        self._summaries = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        top = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("Filter columns...")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._apply_filter)
        top.addWidget(self.search)

        self.auto_btn = QPushButton(auto_label)
        self.auto_btn.setToolTip("Tick every column that looks like it belongs here")
        self.auto_btn.clicked.connect(self._auto_select)
        top.addWidget(self.auto_btn)

        all_btn = QPushButton("All")
        all_btn.setFixedWidth(46)
        all_btn.clicked.connect(lambda: self._set_all(True))
        top.addWidget(all_btn)

        none_btn = QPushButton("None")
        none_btn.setFixedWidth(52)
        none_btn.clicked.connect(lambda: self._set_all(False))
        top.addWidget(none_btn)
        layout.addLayout(top)

        self.list = QListWidget()
        self.list.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.list.setAlternatingRowColors(True)
        self.list.setMinimumHeight(110)
        self.list.setMaximumHeight(150)
        self.list.itemChanged.connect(lambda _: self.selection_changed.emit())
        layout.addWidget(self.list)

        self.count_label = QLabel("0 columns selected")
        self.count_label.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(self.count_label)
        self.selection_changed.connect(self._update_count)

    def set_auto_predicate(self, predicate):
        self._auto_predicate = predicate

    def set_columns(self, columns, summaries=None):
        checked = set(self.checked_columns())
        self._summaries = summaries or {}
        self.list.blockSignals(True)
        self.list.clear()
        for col in columns:
            item = QListWidgetItem(str(col))
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked if str(col) in checked else Qt.CheckState.Unchecked
            )
            summary = self._summaries.get(str(col))
            if summary:
                item.setToolTip(summary)
            self.list.addItem(item)
        self.list.blockSignals(False)
        self._apply_filter(self.search.text())
        self.selection_changed.emit()

    def checked_columns(self):
        out = []
        for i in range(self.list.count()):
            item = self.list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                out.append(item.text())
        return out

    def set_checked(self, columns):
        wanted = {str(c) for c in columns}
        self.list.blockSignals(True)
        for i in range(self.list.count()):
            item = self.list.item(i)
            item.setCheckState(
                Qt.CheckState.Checked if item.text() in wanted else Qt.CheckState.Unchecked
            )
        self.list.blockSignals(False)
        self.selection_changed.emit()

    def _set_all(self, state):
        self.list.blockSignals(True)
        for i in range(self.list.count()):
            item = self.list.item(i)
            if item.isHidden():
                continue
            item.setCheckState(Qt.CheckState.Checked if state else Qt.CheckState.Unchecked)
        self.list.blockSignals(False)
        self.selection_changed.emit()

    def _auto_select(self):
        if self._auto_predicate is None:
            return
        matches = [
            self.list.item(i).text()
            for i in range(self.list.count())
            if self._auto_predicate(self.list.item(i).text())
        ]
        if not matches:
            QMessageBox.information(self, "Nothing found",
                                    "No columns in this file matched that pattern.")
            return
        self.set_checked(matches)

    def _apply_filter(self, text):
        needle = (text or "").strip().lower()
        for i in range(self.list.count()):
            item = self.list.item(i)
            item.setHidden(bool(needle) and needle not in item.text().lower())

    def _update_count(self):
        n = len(self.checked_columns())
        self.count_label.setText(f"{n} column{'' if n == 1 else 's'} selected")


class HintLabel(QLabel):
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setWordWrap(True)
        self.setStyleSheet("color: #666; font-size: 11px;")


# --------------------------------------------------------------------------- #
#  intensity thresholding dialog
# --------------------------------------------------------------------------- #

class ThresholdDialog(QDialog):
    """
    Walk through each intensity column and pick the cut that separates positive
    from negative nodes. Left-drag moves the low line, right-drag the high line.
    """

    def __init__(self, data, thresholds=None, parent=None):
        """
        data       : {column_name: 1D float ndarray}
        thresholds : {column_name: (low, high_or_None)} starting values
        """
        super().__init__(parent)
        self.setWindowTitle("Set identity thresholds")
        self.setMinimumSize(820, 540)

        self.data = data
        self.names = list(data.keys())
        self.thresholds = {}
        for name in self.names:
            if thresholds and name in thresholds:
                self.thresholds[name] = tuple(thresholds[name])
            else:
                self.thresholds[name] = (_otsu_threshold(data[name]), None)

        self.current = None
        self._dragging = None
        self._hist_cache = {}

        root = QHBoxLayout(self)

        # ---- left: marker list -------------------------------------------- #
        left = QVBoxLayout()
        left.addWidget(QLabel("<b>Identities</b>"))
        self.list = QListWidget()
        self.list.setMinimumWidth(230)
        self.list.currentRowChanged.connect(self._select_row)
        left.addWidget(self.list)
        left.addWidget(HintLabel("Positive nodes are shown in brackets. "
                                 "Click an identity to adjust its cut."))
        root.addLayout(left, 0)

        # ---- right: plot + controls --------------------------------------- #
        right = QVBoxLayout()

        self.title_label = QLabel("")
        self.title_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        right.addWidget(self.title_label)

        if MATPLOTLIB_AVAILABLE:
            self.figure = Figure(figsize=(5, 3.2), tight_layout=True)
            self.canvas = FigureCanvas(self.figure)
            self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            self.ax = self.figure.add_subplot(111)
            self.canvas.mpl_connect('button_press_event', self._on_press)
            self.canvas.mpl_connect('motion_notify_event', self._on_motion)
            self.canvas.mpl_connect('button_release_event', self._on_release)
            right.addWidget(self.canvas, 1)
        else:
            self.canvas = None
            note = QLabel("matplotlib is not available - type thresholds by hand below.")
            note.setStyleSheet("color: #b36b00;")
            right.addWidget(note)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Positive at or above:"))
        self.low_edit = QLineEdit()
        self.low_edit.setFixedWidth(110)
        self.low_edit.editingFinished.connect(self._low_typed)
        controls.addWidget(self.low_edit)

        self.use_high = QCheckBox("and at or below")
        self.use_high.toggled.connect(self._toggle_high)
        controls.addWidget(self.use_high)

        self.high_edit = QLineEdit()
        self.high_edit.setFixedWidth(110)
        self.high_edit.setEnabled(False)
        self.high_edit.editingFinished.connect(self._high_typed)
        controls.addWidget(self.high_edit)
        controls.addStretch()

        self.log_check = QCheckBox("Log y")
        self.log_check.setChecked(True)
        self.log_check.toggled.connect(lambda _: self._redraw())
        controls.addWidget(self.log_check)
        right.addLayout(controls)

        buttons = QHBoxLayout()
        otsu_btn = QPushButton("Auto (Otsu)")
        otsu_btn.setToolTip("Pick the cut that best splits this column into two populations")
        otsu_btn.clicked.connect(self._auto_current)
        buttons.addWidget(otsu_btn)

        mean_btn = QPushButton("Mean + 1 SD")
        mean_btn.clicked.connect(self._mean_sd_current)
        buttons.addWidget(mean_btn)

        auto_all_btn = QPushButton("Auto for all")
        auto_all_btn.clicked.connect(self._auto_all)
        buttons.addWidget(auto_all_btn)

        copy_btn = QPushButton("Copy value to all")
        copy_btn.clicked.connect(self._copy_to_all)
        buttons.addWidget(copy_btn)
        buttons.addStretch()
        right.addLayout(buttons)

        self.stats_label = HintLabel("")
        right.addWidget(self.stats_label)

        nav = QHBoxLayout()
        prev_btn = QPushButton("< Previous")
        prev_btn.clicked.connect(lambda: self._step(-1))
        nav.addWidget(prev_btn)
        next_btn = QPushButton("Next >")
        next_btn.clicked.connect(lambda: self._step(1))
        nav.addWidget(next_btn)
        nav.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        nav.addWidget(cancel_btn)

        ok_btn = QPushButton("Done")
        ok_btn.setDefault(True)
        ok_btn.setStyleSheet(
            "QPushButton { background-color: #28a745; color: white; font-weight: bold;"
            " padding: 6px 18px; border: none; border-radius: 4px; }"
            "QPushButton:hover { background-color: #218838; }"
        )
        ok_btn.clicked.connect(self.accept)
        nav.addWidget(ok_btn)
        right.addLayout(nav)

        root.addLayout(right, 1)

        self._refresh_list()
        if self.names:
            self.list.setCurrentRow(0)

    # -- data helpers ------------------------------------------------------- #

    def _positive_count(self, name):
        low, high = self.thresholds[name]
        arr = self.data[name]
        mask = np.isfinite(arr) & (arr >= low)
        if high is not None:
            mask &= arr <= high
        return int(np.count_nonzero(mask))

    def _histogram(self, name):
        if name not in self._hist_cache:
            arr = self.data[name]
            arr = arr[np.isfinite(arr)]
            if arr.size > MAX_HIST_SAMPLES:
                step = int(np.ceil(arr.size / MAX_HIST_SAMPLES))
                arr = arr[::step]
            if arr.size == 0:
                self._hist_cache[name] = (np.array([0]), np.array([0.0, 1.0]))
            else:
                bins = int(np.clip(np.sqrt(arr.size), 32, 256))
                self._hist_cache[name] = np.histogram(arr, bins=bins)
        return self._hist_cache[name]

    # -- list --------------------------------------------------------------- #

    def _refresh_list(self):
        row = self.list.currentRow()
        self.list.blockSignals(True)
        self.list.clear()
        for name in self.names:
            n = self._positive_count(name)
            self.list.addItem(f"{name}   [{n}]")
        self.list.blockSignals(False)
        if 0 <= row < self.list.count():
            self.list.setCurrentRow(row)

    def _refresh_current_list_row(self):
        if self.current is None:
            return
        idx = self.names.index(self.current)
        item = self.list.item(idx)
        if item is not None:
            item.setText(f"{self.current}   [{self._positive_count(self.current)}]")

    def _select_row(self, row):
        if not (0 <= row < len(self.names)):
            return
        self.current = self.names[row]
        low, high = self.thresholds[self.current]
        self.title_label.setText(self.current)
        self.low_edit.setText(_fmt(low))
        self.use_high.blockSignals(True)
        self.use_high.setChecked(high is not None)
        self.use_high.blockSignals(False)
        self.high_edit.setEnabled(high is not None)
        arr = self.data[self.current]
        finite = arr[np.isfinite(arr)]
        self.high_edit.setText(_fmt(high if high is not None
                                    else (finite.max() if finite.size else 0.0)))
        self._redraw()

    def _step(self, delta):
        row = self.list.currentRow() + delta
        if 0 <= row < self.list.count():
            self.list.setCurrentRow(row)

    # -- threshold edits ---------------------------------------------------- #

    def _set_threshold(self, low=None, high=..., redraw=True):
        if self.current is None:
            return
        cur_low, cur_high = self.thresholds[self.current]
        new_low = cur_low if low is None else float(low)
        new_high = cur_high if high is ... else high
        if new_high is not None and new_high < new_low:
            new_high = new_low
        self.thresholds[self.current] = (new_low, new_high)
        self.low_edit.setText(_fmt(new_low))
        if new_high is not None:
            self.high_edit.setText(_fmt(new_high))
        self._refresh_current_list_row()
        if redraw:
            self._redraw()

    def _low_typed(self):
        try:
            self._set_threshold(low=float(self.low_edit.text()))
        except ValueError:
            if self.current is not None:
                self.low_edit.setText(_fmt(self.thresholds[self.current][0]))

    def _high_typed(self):
        if not self.use_high.isChecked():
            return
        try:
            self._set_threshold(high=float(self.high_edit.text()))
        except ValueError:
            if self.current is not None:
                self.high_edit.setText(_fmt(self.thresholds[self.current][1] or 0.0))

    def _toggle_high(self, checked):
        self.high_edit.setEnabled(checked)
        if self.current is None:
            return
        if checked:
            try:
                value = float(self.high_edit.text())
            except ValueError:
                finite = self.data[self.current][np.isfinite(self.data[self.current])]
                value = float(finite.max()) if finite.size else 0.0
            self._set_threshold(high=value)
        else:
            self._set_threshold(high=None)

    def _auto_current(self):
        if self.current is not None:
            self._set_threshold(low=_otsu_threshold(self.data[self.current]))

    def _mean_sd_current(self):
        if self.current is None:
            return
        arr = self.data[self.current]
        finite = arr[np.isfinite(arr)]
        if finite.size:
            self._set_threshold(low=float(np.mean(finite) + np.std(finite)))

    def _auto_all(self):
        for name in self.names:
            self.thresholds[name] = (_otsu_threshold(self.data[name]), self.thresholds[name][1])
        self._refresh_list()
        if self.current is not None:
            self._select_row(self.names.index(self.current))

    def _copy_to_all(self):
        if self.current is None:
            return
        low, high = self.thresholds[self.current]
        for name in self.names:
            self.thresholds[name] = (low, high)
        self._refresh_list()
        self._select_row(self.names.index(self.current))

    # -- plotting ----------------------------------------------------------- #

    def _redraw(self):
        if self.current is None:
            return
        name = self.current
        arr = self.data[name]
        finite = arr[np.isfinite(arr)]
        low, high = self.thresholds[name]
        n_pos = self._positive_count(name)
        total = int(finite.size)
        pct = (100.0 * n_pos / total) if total else 0.0
        self.stats_label.setText(
            f"{n_pos:,} of {total:,} nodes positive ({pct:.1f}%)   |   "
            f"range {_fmt(finite.min()) if total else 'n/a'} to "
            f"{_fmt(finite.max()) if total else 'n/a'}   |   "
            f"mean {_fmt(np.mean(finite)) if total else 'n/a'}"
        )

        if not MATPLOTLIB_AVAILABLE:
            return

        counts, edges = self._histogram(name)
        centers = 0.5 * (edges[:-1] + edges[1:])
        widths = np.diff(edges)

        self.ax.clear()
        pos_mask = centers >= low
        if high is not None:
            pos_mask = pos_mask & (centers <= high)
        colors = np.where(pos_mask, '#28a745', '#b0b0b0')
        self.ax.bar(centers, counts, width=widths, align='center', color=list(colors), alpha=0.85)
        if self.log_check.isChecked():
            self.ax.set_yscale('log')
        self.low_line = self.ax.axvline(low, color='#d62728', linewidth=2.2, zorder=10)
        if high is not None:
            self.high_line = self.ax.axvline(high, color='#1f77b4', linewidth=2.2, zorder=10)
        else:
            self.high_line = None
        self.ax.set_xlabel(name)
        self.ax.set_ylabel("nodes")
        self.ax.set_title("left-drag = lower cut     right-drag = upper cut", fontsize=9, color='#666')
        self.canvas.draw_idle()

    def _on_press(self, event):
        if event.inaxes != getattr(self, 'ax', None) or event.xdata is None:
            return
        if event.button == 1:
            self._dragging = 'low'
            self._set_threshold(low=event.xdata, redraw=False)
            self._quick_move()
        elif event.button == 3 and self.use_high.isChecked():
            self._dragging = 'high'
            self._set_threshold(high=float(event.xdata), redraw=False)
            self._quick_move()

    def _on_motion(self, event):
        if not self._dragging or event.inaxes != getattr(self, 'ax', None) or event.xdata is None:
            return
        if self._dragging == 'low':
            self._set_threshold(low=event.xdata, redraw=False)
        else:
            self._set_threshold(high=float(event.xdata), redraw=False)
        self._quick_move()

    def _quick_move(self):
        """Cheap update while dragging - move the line, skip the full replot."""
        if not MATPLOTLIB_AVAILABLE or self.current is None:
            return
        low, high = self.thresholds[self.current]
        if getattr(self, 'low_line', None) is not None:
            self.low_line.set_xdata([low, low])
        if getattr(self, 'high_line', None) is not None and high is not None:
            self.high_line.set_xdata([high, high])
        self.canvas.draw_idle()

    def _on_release(self, event):
        if self._dragging:
            self._dragging = None
            self._redraw()

    def get_thresholds(self):
        return dict(self.thresholds)


# --------------------------------------------------------------------------- #
#  property panels
# --------------------------------------------------------------------------- #

class BasePanel(QWidget):
    changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.df = None

    def set_dataframe(self, df):
        self.df = df
        self.on_dataframe(df)
        self.changed.emit()

    def on_dataframe(self, df):
        pass

    def node_ids(self, column):
        """IDs from `column`, or sequential 1..N when column is None."""
        if column is None:
            return [int(i) for i in range(1, len(self.df) + 1)]
        return [_coerce_node_id(v) for v in self.df[column].tolist()]

    def build(self):
        """-> (result_dict, summary_string). Raise ValueError with a friendly message."""
        raise NotImplementedError


class SimplePanel(BasePanel):
    """One combo per required field - used for Centroids and Communities."""

    def __init__(self, fields, guesses, parent=None):
        """
        fields  : list of (key_name, label, required)
        guesses : {key_name: [header keywords]}
        """
        super().__init__(parent)
        self.fields = fields
        self.guesses = guesses
        self.combos = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        for key, label, required in fields:
            placeholder = ("Auto: 1, 2, 3, ..." if key == 'Numerical IDs'
                           else "Select column...")
            combo = ColumnCombo(placeholder)
            combo.currentIndexChanged.connect(lambda _: self.changed.emit())
            self.combos[key] = combo
            form.addRow(f"{label}{'' if required else ' (optional)'}:", combo)
        layout.addLayout(form)
        layout.addWidget(HintLabel(
            "Leave the node ID column empty to number nodes 1, 2, 3 ... in row order."))
        layout.addStretch()

    def on_dataframe(self, df):
        cols = list(df.columns) if df is not None else []
        for key, combo in self.combos.items():
            combo.set_columns(cols)
            if df is not None and combo.current_column() is None:
                guess = _guess_column(cols, self.guesses.get(key, []))
                if guess is not None:
                    combo.select(guess)

    def build(self):
        if self.df is None:
            raise ValueError("Load a spreadsheet first.")
        result = {}
        id_col = self.combos['Numerical IDs'].current_column()
        result['Numerical IDs'] = self.node_ids(id_col)
        for key, label, required in self.fields:
            if key == 'Numerical IDs':
                continue
            col = self.combos[key].current_column()
            if col is None:
                if required:
                    raise ValueError(f"Choose a column for '{label}'.")
                continue
            result[key] = self.df[col].tolist()

        # drop rows where any non-ID field is blank
        keys = [k for k in result if k != 'Numerical IDs']
        keep = [i for i in range(len(result['Numerical IDs']))
                if not any(_is_nan_or_empty(result[k][i]) for k in keys)
                and result['Numerical IDs'][i] is not None]
        for k in result:
            result[k] = [result[k][i] for i in keep]

        n = len(result['Numerical IDs'])
        summary = (f"{n:,} nodes ready.\n"
                   f"IDs from: {id_col or 'sequential 1..N'}\n"
                   + "\n".join(f"{k}: {self.combos[k].current_column()}"
                               for k, _, _ in self.fields if k != 'Numerical IDs'
                               and self.combos[k].current_column()))
        return result, summary


class IdentityPanel(BasePanel):
    """Three ways to turn columns into node identities."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.mode = MODE_SINGLE
        self.thresholds = {}          # column -> (low, high or None)
        self.zscore = False
        self._asked_zscore = False
        self._numeric_cache = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # --- node id column ------------------------------------------------ #
        id_box = QGroupBox("1.  Node ID column")
        id_layout = QVBoxLayout(id_box)
        self.id_combo = ColumnCombo("Auto: number rows 1, 2, 3, ...")
        self.id_combo.currentIndexChanged.connect(lambda _: self.changed.emit())
        id_layout.addWidget(self.id_combo)
        id_layout.addWidget(HintLabel(
            "Leave empty to number nodes 1, 2, 3 ... in row order."))
        layout.addWidget(id_box)

        # --- mode ----------------------------------------------------------- #
        mode_box = QGroupBox("2.  How are identities stored in this file?")
        mode_layout = QVBoxLayout(mode_box)
        self.mode_group = QButtonGroup(self)

        self.radio_single = QRadioButton("One identity column")
        self.radio_matrix = QRadioButton("Several 0 / 1 columns (identity matrix)")
        self.radio_intensity = QRadioButton("Several raw intensity columns (threshold them)")
        self.radio_single.setChecked(True)

        self.MODE_HINTS = (
            "A single column of labels, e.g. \"Class\" holding Tumor / Stroma / Immune.",
            "One column per marker; the header becomes the identity name and a 1 "
            "means the node carries it. Nodes may get several identities.",
            "One column per marker holding measured intensities. You set a cut-off "
            "per marker in a histogram; at or above it counts as positive.",
        )
        for i, radio in enumerate((self.radio_single, self.radio_matrix, self.radio_intensity)):
            radio.setToolTip(self.MODE_HINTS[i])
            self.mode_group.addButton(radio, i)
            mode_layout.addWidget(radio)

        self.mode_hint = HintLabel(self.MODE_HINTS[0])
        self.mode_hint.setContentsMargins(20, 4, 0, 0)
        mode_layout.addWidget(self.mode_hint)

        self.mode_group.idToggled.connect(self._mode_changed)
        layout.addWidget(mode_box)

        # --- mode-specific -------------------------------------------------- #
        self.detail_box = QGroupBox("3.  Identity columns")
        detail_layout = QVBoxLayout(self.detail_box)
        self.stack = QStackedWidget()

        # single
        single_page = QWidget()
        single_form = QFormLayout(single_page)
        self.identity_combo = ColumnCombo("Select column...")
        self.identity_combo.currentIndexChanged.connect(lambda _: self.changed.emit())
        single_form.addRow("Identity column:", self.identity_combo)
        self.stack.addWidget(single_page)

        # matrix
        matrix_page = QWidget()
        matrix_layout = QVBoxLayout(matrix_page)
        matrix_layout.setContentsMargins(0, 0, 0, 0)
        self.matrix_list = ColumnCheckList("Auto-detect 0/1 columns")
        self.matrix_list.set_auto_predicate(self._is_binary_column)
        self.matrix_list.selection_changed.connect(self.changed.emit)
        matrix_layout.addWidget(self.matrix_list)
        matrix_layout.addWidget(HintLabel(
            "Tick every marker column. A node gets each identity whose column holds a 1."))
        self.stack.addWidget(matrix_page)

        # intensity
        intensity_page = QWidget()
        intensity_layout = QVBoxLayout(intensity_page)
        intensity_layout.setContentsMargins(0, 0, 0, 0)
        self.intensity_list = ColumnCheckList("Auto-detect numeric columns")
        self.intensity_list.set_auto_predicate(self._is_intensity_column)
        self.intensity_list.selection_changed.connect(self._intensity_selection_changed)
        intensity_layout.addWidget(self.intensity_list)

        action_row = QHBoxLayout()
        self.zscore_check = QCheckBox("Z-score normalise first")
        self.zscore_check.setToolTip(
            "Centre and scale each selected column before thresholding.\n"
            "Leave off if these values are already normalised.")
        self.zscore_check.toggled.connect(self._zscore_toggled)
        action_row.addWidget(self.zscore_check)
        action_row.addStretch()

        self.threshold_btn = QPushButton("Set thresholds...")
        self.threshold_btn.setStyleSheet(
            "QPushButton { background-color: #007acc; color: white; font-weight: bold;"
            " padding: 7px 16px; border: none; border-radius: 4px; }"
            "QPushButton:hover { background-color: #005a9e; }"
        )
        self.threshold_btn.clicked.connect(self.open_threshold_dialog)
        action_row.addWidget(self.threshold_btn)
        intensity_layout.addLayout(action_row)

        self.threshold_summary = HintLabel(
            "Thresholds start at an automatic estimate; open the dialog to refine them.")
        intensity_layout.addWidget(self.threshold_summary)
        self.stack.addWidget(intensity_page)

        detail_layout.addWidget(self.stack)
        layout.addWidget(self.detail_box)
        layout.addStretch()

    # -- column helpers ----------------------------------------------------- #

    def _is_binary_column(self, name):
        if self.df is None or name not in self.df.columns:
            return False
        if name == self.id_combo.current_column():
            return False
        return _looks_binary(self.df[name])

    def _is_intensity_column(self, name):
        if self.df is None or name not in self.df.columns:
            return False
        if name == self.id_combo.current_column():
            return False
        series = self.df[name]
        return _looks_numeric(series) and not _looks_binary(series)

    def _column_values(self, name):
        if name not in self._numeric_cache:
            self._numeric_cache[name] = _numeric_column(self.df[name])
        arr = self._numeric_cache[name]
        return _zscore(arr) if self.zscore else arr

    def _column_summaries(self, df):
        summaries = {}
        sample = df.head(2000)
        for col in df.columns:
            series = sample[col]
            try:
                if _looks_numeric(series):
                    conv = pd.to_numeric(series, errors='coerce')
                    summaries[str(col)] = (f"numeric | {_fmt(conv.min())} to {_fmt(conv.max())}"
                                           f"{' | looks binary' if _looks_binary(series) else ''}")
                else:
                    uniques = pd.unique(series.dropna())[:6]
                    summaries[str(col)] = "text | " + ", ".join(str(u) for u in uniques)
            except Exception:
                summaries[str(col)] = ""
        return summaries

    # -- events -------------------------------------------------------------- #

    def on_dataframe(self, df):
        cols = list(df.columns) if df is not None else []
        self._numeric_cache.clear()
        self.thresholds.clear()
        self._asked_zscore = False

        self.id_combo.set_columns(cols)
        self.identity_combo.set_columns(cols)
        if df is not None:
            guess = _guess_column(cols, ['object id', 'label', 'node id', 'cell id',
                                         'id', 'node', 'object'])
            self.id_combo.select(guess)
            id_guess = self.id_combo.current_column()
            identity_guess = _guess_column(
                cols, ['classification', 'class', 'identity', 'cell type', 'celltype',
                       'type', 'phenotype', 'name'], exclude=(id_guess,) if id_guess else ())
            self.identity_combo.select(identity_guess)

        summaries = self._column_summaries(df) if df is not None else {}
        self.matrix_list.set_columns(cols, summaries)
        self.intensity_list.set_columns(cols, summaries)
        self._update_threshold_summary()

    def _mode_changed(self, mode_id, checked):
        if not checked:
            return
        self.mode = mode_id
        self.stack.setCurrentIndex(mode_id)
        self.mode_hint.setText(self.MODE_HINTS[mode_id])
        if mode_id == MODE_INTENSITY:
            self._maybe_ask_zscore()
        self.changed.emit()

    def _maybe_ask_zscore(self):
        """The z-score question, asked once when intensity mode is first chosen."""
        if self._asked_zscore or self.df is None:
            return
        self._asked_zscore = True
        answer = QMessageBox.question(
            self,
            "Normalise intensities?",
            "Z-score normalise the selected intensity columns before thresholding?\n\n"
            "Choose No if these values are already normalised.\n"
            "You can change this any time with the checkbox.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        self.zscore_check.setChecked(answer == QMessageBox.StandardButton.Yes)

    def _zscore_toggled(self, checked):
        self.zscore = checked
        self.thresholds.clear()   # old cuts are in the wrong units now
        self._update_threshold_summary()
        self.changed.emit()

    def _intensity_selection_changed(self):
        selected = set(self.intensity_list.checked_columns())
        for name in list(self.thresholds):
            if name not in selected:
                del self.thresholds[name]
        self._update_threshold_summary()
        self.changed.emit()

    def _ensure_thresholds(self):
        """Fill in Otsu defaults so a preview exists before the dialog is opened."""
        for name in self.intensity_list.checked_columns():
            if name not in self.thresholds:
                self.thresholds[name] = (_otsu_threshold(self._column_values(name)), None)

    def _update_threshold_summary(self):
        cols = self.intensity_list.checked_columns()
        if not cols:
            self.threshold_summary.setText("Tick the intensity columns you want to import.")
            return
        set_count = sum(1 for c in cols if c in self.thresholds)
        unit = " (z-scored)" if self.zscore else ""
        self.threshold_summary.setText(
            f"{set_count} of {len(cols)} thresholds set{unit}. "
            "Unset ones fall back to an automatic estimate."
        )

    def open_threshold_dialog(self):
        if self.df is None:
            QMessageBox.warning(self, "No data", "Load a spreadsheet first.")
            return
        self._maybe_ask_zscore()
        cols = self.intensity_list.checked_columns()
        if not cols:
            QMessageBox.warning(self, "No columns",
                                "Tick at least one intensity column first.")
            return
        data = {name: self._column_values(name) for name in cols}
        dialog = ThresholdDialog(data, self.thresholds, parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.thresholds = dialog.get_thresholds()
            self._update_threshold_summary()
            self.changed.emit()

    # -- build ---------------------------------------------------------------- #

    def _identity_map(self):
        """-> ordered {node_id: [identity, ...]} plus a per-identity tally."""
        id_col = self.id_combo.current_column()
        ids = self.node_ids(id_col)
        mapping = {}
        tally = {}

        def identity_columns(picked):
            # the ID column is never an identity, even if it got ticked
            return [c for c in picked if c != id_col]

        def add(node, label):
            if node is None or _is_nan_or_empty(label):
                return
            label = str(label).strip()
            bucket = mapping.setdefault(node, [])
            if label not in bucket:
                bucket.append(label)
                tally[label] = tally.get(label, 0) + 1

        if self.mode == MODE_SINGLE:
            col = self.identity_combo.current_column()
            if col is None:
                raise ValueError("Choose the column holding the identities.")
            for node, label in zip(ids, self.df[col].tolist()):
                add(node, label)

        elif self.mode == MODE_MATRIX:
            cols = identity_columns(self.matrix_list.checked_columns())
            if not cols:
                raise ValueError("Tick at least one 0/1 identity column.")
            for col in cols:
                flags = self.df[col].tolist()
                for node, flag in zip(ids, flags):
                    if _truthy(flag):
                        add(node, col)

        else:
            cols = identity_columns(self.intensity_list.checked_columns())
            if not cols:
                raise ValueError("Tick at least one intensity column.")
            self._ensure_thresholds()
            id_array = np.array(ids, dtype=object)
            for col in cols:
                low, high = self.thresholds[col]
                values = self._column_values(col)
                mask = np.isfinite(values) & (values >= low)
                if high is not None:
                    mask &= values <= high
                for node in id_array[mask]:
                    add(node, col)

        return mapping, tally

    def build(self):
        if self.df is None:
            raise ValueError("Load a spreadsheet first.")
        mapping, tally = self._identity_map()
        if not mapping:
            raise ValueError("No node ended up with an identity. Check the columns "
                             "you picked (and your thresholds, in intensity mode).")

        node_list = list(mapping.keys())
        identity_values = []
        for node in node_list:
            labels = mapping[node]
            identity_values.append(labels if (ALWAYS_LIST or len(labels) > 1) else labels[0])

        result = {'Numerical IDs': node_list, 'Identity Column': identity_values}

        multi = sum(1 for node in node_list if len(mapping[node]) > 1)
        lines = [
            f"{len(node_list):,} nodes with at least one identity.",
            f"IDs from: {self.id_combo.current_column() or 'sequential 1..N'}",
            f"{len(tally)} distinct identit{'y' if len(tally) == 1 else 'ies'}"
            + (f", {multi:,} nodes carry more than one." if multi else "."),
            "",
        ]
        for label, count in sorted(tally.items(), key=lambda kv: -kv[1])[:15]:
            lines.append(f"  {label}: {count:,}")
        if len(tally) > 15:
            lines.append(f"  ... and {len(tally) - 15} more")

        sample = node_list[:5]
        lines.append("")
        lines.append("Sample: " + "; ".join(f"{n} -> {mapping[n]}" for n in sample))
        return result, "\n".join(lines), mapping


# --------------------------------------------------------------------------- #
#  main window
# --------------------------------------------------------------------------- #

class ExcelToDictGUI(QMainWindow):
    # dictionary, property_name, add_status  (unchanged contract)
    data_exported = pyqtSignal(dict, str, bool)
    # {node_id: [identity, ...]}, add_status  (convenience, Node Identities only)
    identities_exported = pyqtSignal(dict, bool)

    PROPERTIES = ['Node Identities', 'Node Centroids', 'Node Communities']

    def __init__(self):
        super().__init__()
        self.df = None
        self.add = False
        self.identity_dict = {}

        self.setWindowTitle("Spreadsheet Importer - NetTracer3D")
        self.setGeometry(100, 100, 1250, 780)
        self._build_ui()
        self._refresh_preview()

    # -- ui ------------------------------------------------------------------ #

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(6)

        # ---- left: file + table ------------------------------------------- #
        left = QWidget()
        left.setMinimumWidth(420)
        left_layout = QVBoxLayout(left)

        heading = QLabel("Spreadsheet")
        heading.setStyleSheet("font-weight: bold; font-size: 15px;")
        left_layout.addWidget(heading)

        self.drop_zone = DropZoneWidget()
        self.drop_zone.file_dropped.connect(self.load_file)
        left_layout.addWidget(self.drop_zone)

        self.file_label = HintLabel("No file loaded.")
        left_layout.addWidget(self.file_label)

        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        left_layout.addWidget(self.table)

        splitter.addWidget(left)

        # ---- right: property builder --------------------------------------- #
        right = QWidget()
        right.setMinimumWidth(430)
        right_layout = QVBoxLayout(right)

        prop_row = QHBoxLayout()
        prop_label = QLabel("Import as:")
        prop_label.setStyleSheet("font-weight: bold; font-size: 15px;")
        prop_row.addWidget(prop_label)
        self.property_combo = QComboBox()
        self.property_combo.addItems(self.PROPERTIES)
        self.property_combo.currentIndexChanged.connect(self._property_changed)
        prop_row.addWidget(self.property_combo, 1)
        right_layout.addLayout(prop_row)

        self.identity_panel = IdentityPanel()
        self.centroid_panel = SimplePanel(
            fields=[('Numerical IDs', 'Node ID column', False),
                    ('Z', 'Z', True), ('Y', 'Y', True), ('X', 'X', True)],
            guesses={'Numerical IDs': ['object id', 'label', 'node id', 'id'],
                     'Z': ['centroid z', 'z position', ' z', 'z'],
                     'Y': ['centroid y', 'y position', ' y', 'y'],
                     'X': ['centroid x', 'x position', ' x', 'x']},
        )
        self.community_panel = SimplePanel(
            fields=[('Numerical IDs', 'Node ID column', False),
                    ('Community Identifier', 'Community column', True)],
            guesses={'Numerical IDs': ['object id', 'label', 'node id', 'id'],
                     'Community Identifier': ['community', 'cluster', 'group', 'region']},
        )

        self.panels = {
            'Node Identities': self.identity_panel,
            'Node Centroids': self.centroid_panel,
            'Node Communities': self.community_panel,
        }

        panel_stack_host = QScrollArea()
        panel_stack_host.setWidgetResizable(True)
        panel_stack_host.setFrameShape(QFrame.Shape.NoFrame)
        self.panel_stack = QStackedWidget()
        for name in self.PROPERTIES:
            panel = self.panels[name]
            panel.changed.connect(self._refresh_preview)
            self.panel_stack.addWidget(panel)
        panel_stack_host.setWidget(self.panel_stack)
        right_layout.addWidget(panel_stack_host, 1)

        preview_box = QGroupBox("Preview")
        preview_layout = QVBoxLayout(preview_box)
        self.preview_text = QTextEdit()
        self.preview_text.setReadOnly(True)
        self.preview_text.setFixedHeight(132)
        self.preview_text.setFont(QFont("Monospace", 9))
        preview_layout.addWidget(self.preview_text)
        right_layout.addWidget(preview_box)

        export_row = QHBoxLayout()
        self.add_check = QCheckBox("Add to existing properties")
        self.add_check.setToolTip("Off: replace what NetTracer3D already holds.\n"
                                  "On: merge with it.")
        self.add_check.toggled.connect(self._toggle_add)
        export_row.addWidget(self.add_check)
        export_row.addStretch()

        self.export_btn = QPushButton("Send to NetTracer3D")
        self.export_btn.setStyleSheet(
            "QPushButton { background-color: #28a745; color: white; font-weight: bold;"
            " padding: 10px 22px; border: none; border-radius: 5px; font-size: 13px; }"
            "QPushButton:hover { background-color: #218838; }"
            "QPushButton:disabled { background-color: #b5b5b5; }"
        )
        self.export_btn.clicked.connect(self.export_dictionary)
        export_row.addWidget(self.export_btn)
        right_layout.addLayout(export_row)

        splitter.addWidget(right)
        splitter.setSizes([680, 570])
        outer.addWidget(splitter)

    # -- file ---------------------------------------------------------------- #

    def load_file(self, file_path):
        try:
            lowered = file_path.lower()
            if lowered.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            elif lowered.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif lowered.endswith(('.tsv', '.txt')):
                df = pd.read_csv(file_path, sep='\t')
            else:
                QMessageBox.warning(self, "Unsupported file",
                                    "Please choose a .csv, .tsv or .xlsx file.")
                return
        except Exception as exc:
            QMessageBox.critical(self, "Could not read file", str(exc))
            return

        if df.empty or len(df.columns) == 0:
            QMessageBox.warning(self, "Empty file", "That file has no usable rows.")
            return

        df.columns = [str(c) for c in df.columns]
        self.df = df
        self.drop_zone.set_message(os.path.basename(file_path))
        self.file_label.setText(
            f"{os.path.basename(file_path)} - {len(df):,} rows, {len(df.columns)} columns"
        )
        self._populate_table()
        for panel in self.panels.values():
            panel.set_dataframe(df)
        self._refresh_preview()

    def _populate_table(self):
        df = self.df
        rows = min(MAX_PREVIEW_ROWS, len(df))
        self.table.clear()
        self.table.setRowCount(rows)
        self.table.setColumnCount(len(df.columns))
        self.table.setHorizontalHeaderLabels([str(c) for c in df.columns])
        for i in range(rows):
            for j in range(len(df.columns)):
                value = df.iat[i, j]
                if isinstance(value, float) and not math.isnan(value):
                    text = f"{value:.6g}"
                else:
                    text = "" if _is_nan_or_empty(value) else str(value)
                self.table.setItem(i, j, QTableWidgetItem(text))
        self.table.resizeColumnsToContents()

    # -- state --------------------------------------------------------------- #

    def _property_changed(self, index):
        self.panel_stack.setCurrentIndex(index)
        self._refresh_preview()

    def _toggle_add(self, checked):
        self.add = checked

    def _current_panel(self):
        return self.panels[self.property_combo.currentText()]

    def _current_result(self):
        """-> (result_dict, summary, identity_map_or_None) or (None, message, None)."""
        panel = self._current_panel()
        if self.df is None:
            return None, "Load a spreadsheet to get started.", None
        try:
            out = panel.build()
        except ValueError as exc:
            return None, str(exc), None
        except Exception as exc:  # unexpected - still shouldn't crash the window
            return None, f"Could not build the dictionary: {exc}", None
        if len(out) == 3:
            result, summary, mapping = out
        else:
            result, summary = out
            mapping = None
        return result, summary, mapping

    def _refresh_preview(self):
        result, summary, _ = self._current_result()
        self.preview_text.setPlainText(summary)
        self.export_btn.setEnabled(result is not None)

    # -- export --------------------------------------------------------------- #

    def export_dictionary(self):
        result, summary, mapping = self._current_result()
        if result is None:
            QMessageBox.warning(self, "Not ready", summary)
            return

        property_name = self.property_combo.currentText()
        self.identity_dict = mapping or {}

        self.data_exported.emit(result, property_name, self.add)
        if property_name == 'Node Identities' and mapping is not None:
            self.identities_exported.emit(mapping, self.add)

        # kept for backwards compatibility with older call sites
        builtins.excel_dict = result
        builtins.target_property = property_name
        builtins.add = self.add

        QMessageBox.information(
            self, "Sent",
            f"{property_name} sent to NetTracer3D "
            f"({'added to' if self.add else 'replacing'} existing).\n\n{summary}"
        )


def main(standalone=True):
    if standalone:
        app = QApplication(sys.argv)
        app.setStyle('Fusion')
        window = ExcelToDictGUI()
        window.show()
        sys.exit(app.exec())
    return ExcelToDictGUI


if __name__ == "__main__":
    main(True)
