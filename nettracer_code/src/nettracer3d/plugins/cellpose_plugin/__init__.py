"""
Cellpose Segmentation Plugin for NetTracer3D
=============================================

Provides a comprehensive cellpose interface as a plugin, including:
- All major cellpose parameters exposed as controls
- Input / context / output channel selection
- Built-in and custom model support
- Chunked (piecemeal) segmentation for large images on limited GPUs
- Option to open the standalone cellpose GUI
"""

import subprocess
import sys
import threading
import math
import traceback
from pathlib import Path

import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QComboBox, QPushButton, QSpinBox, QDoubleSpinBox,
    QCheckBox, QGroupBox, QFileDialog, QMessageBox, QProgressBar,
    QSizePolicy, QFrame, QWidget,
)
from PyQt6.QtCore import Qt, pyqtSignal, QObject, QThread

# ──────────────────────────────────────────────────────────────────────
#  Plugin metadata
# ──────────────────────────────────────────────────────────────────────

PLUGIN_INFO = {
    "name": "Cellpose Segmentation",
    "version": "1.0.0",
    "author": "NetTracer3D Contributors",
    "description": (
        "Integrates Cellpose segmentation into NetTracer3D.  "
        "Run cellpose directly from the app with full parameter control, "
        "chunked processing for large volumes, and custom model support."
    ),
    "api_version": (1, 0),
    "requires": [],
    "category": "processing",
}

_api = None


# ──────────────────────────────────────────────────────────────────────
#  register / unregister
# ──────────────────────────────────────────────────────────────────────

def register(api):
    global _api
    _api = api

    api.register_menu_action(
        "Extensions/Cellpose/Open Cellpose Panel...",
        _open_cellpose_dialog,
        tooltip="Full cellpose segmentation interface",
    )
    api.register_menu_action(
        "Extensions/Cellpose/Quick Launch Cellpose GUI",
        _quick_launch_gui,
        tooltip="Open the standalone cellpose GUI (requires cellpose[gui])",
    )
    api.print("Cellpose plugin loaded.")


def unregister(api):
    api.print("Cellpose plugin unloaded.")


# ──────────────────────────────────────────────────────────────────────
#  Callbacks wired by register()
# ──────────────────────────────────────────────────────────────────────

def _open_cellpose_dialog():
    win = _api.get_unsafe_window()
    dialog = CellposeDialog(_api, parent=win)
    dialog.show()
    dialog.raise_()


def _quick_launch_gui():
    shape = _api.get_shape()
    use_3d = True
    if shape is not None:
        use_3d = shape[0] > 1
    _launch_cellpose_gui_subprocess(use_3d=use_3d, parent=_api.get_unsafe_window())


# ──────────────────────────────────────────────────────────────────────
#  Subprocess GUI launcher  (kept from original cellpose_manager)
# ──────────────────────────────────────────────────────────────────────

def _launch_cellpose_gui_subprocess(use_3d=False, parent=None):
    """Fire-and-forget launch of the standalone cellpose Qt GUI."""

    def _run():
        try:
            cmd = [sys.executable, "-m", "cellpose"]
            if use_3d:
                cmd.append("--Zstack")
            subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except Exception as exc:
            tag = "3D " if use_3d else ""
            print(f"Failed to launch cellpose {tag}GUI: {exc}")

    try:
        t = threading.Thread(target=_run, daemon=True)
        t.start()
    except Exception as exc:
        if parent:
            QMessageBox.critical(
                parent, "Cellpose Error",
                f"Could not start cellpose GUI:\n{exc}\n\n"
                "You may need to install cellpose first:\n"
                "  pip install 'cellpose[gui]'\n\n"
                "For GPU support see https://pytorch.org/get-started/locally/",
            )


# ──────────────────────────────────────────────────────────────────────
#  Worker that runs cellpose in a QThread so the GUI stays responsive
# ──────────────────────────────────────────────────────────────────────

class _SegmentationWorker(QObject):
    """Runs cellpose segmentation off the main thread."""

    progress = pyqtSignal(int, int)        # (current_chunk, total_chunks)
    status = pyqtSignal(str)               # human-readable status text
    finished = pyqtSignal(object)          # final label array (or None)
    error = pyqtSignal(str)                # traceback string

    def __init__(self, params: dict):
        super().__init__()
        self.p = params

    def run(self):
        try:
            from cellpose import models

            image = self.p["image"]                  # always 3-D from NetTracer (Z,Y,X)
            context_image = self.p["context_image"]  # optional second channel, also 3-D
            model_path = self.p["model_path"]        # None ⇒ use built-in
            model_type = self.p["model_type"]        # e.g. "cyto3"
            gpu = self.p["use_gpu"]
            diameter = self.p["diameter"]
            flow_threshold = self.p["flow_threshold"]
            cellprob_threshold = self.p["cellprob_threshold"]
            native_3d = self.p["native_3d"]
            stitch_threshold = self.p["stitch_threshold"]
            min_size = self.p["min_size"]
            chunk_count = self.p["chunk_count"]      # 1 = no chunking

            # ── Squeeze out artificial Z=1 dimension ─────────────────
            # NetTracer3D always stores images as (Z,Y,X) even for 2-D,
            # adding Z=1.  Cellpose needs the real dimensionality.
            image = np.squeeze(image)
            if context_image is not None:
                context_image = np.squeeze(context_image)

            # ── Auto-detect dimensionality ───────────────────────────
            is_2d = image.ndim == 2
            is_3d = image.ndim == 3

            if is_2d:
                print(f"[Cellpose] Detected 2D image: {image.shape} (Y, X)")
            elif is_3d:
                print(f"[Cellpose] Detected 3D image: {image.shape} (Z, Y, X)")
                if native_3d:
                    print("[Cellpose] Mode: native 3D convolutions (slow)")
                else:
                    print(f"[Cellpose] Mode: slice-by-slice + stitch "
                          f"(stitch_threshold={stitch_threshold})")
            else:
                print(f"[Cellpose] Unexpected dimensionality: {image.shape}")

            # ── Build model ──────────────────────────────────────────
            self.status.emit("Loading cellpose model...")
            print(f"[Cellpose] Loading model: "
                  f"{model_path or model_type}, GPU={gpu}")

            if model_path:
                model = models.CellposeModel(
                    pretrained_model=str(model_path), gpu=gpu)
            else:
                model = models.CellposeModel(
                    pretrained_model=model_type, gpu=gpu)

            # ── Prepare the image / dual-channel stack ───────────────
            has_context = context_image is not None
            if has_context:
                combined = np.stack([image, context_image], axis=-1)
                # (Y,X,2) for 2-D  or  (Z,Y,X,2) for 3-D
                print(f"[Cellpose] Stacked with context → {combined.shape}")
            else:
                combined = image

            # ── Chunk planning ───────────────────────────────────────
            if chunk_count <= 1 or is_2d:
                chunks = [(combined, (0, 0, 0))]
                total = 1
            else:
                chunks, chunk_grid = self._plan_chunks(
                    combined, chunk_count, overlap=diameter or 30)
                total = len(chunks)
                self.status.emit(
                    f"Planned {total} chunks "
                    f"({chunk_grid[0]}×{chunk_grid[1]}×{chunk_grid[2]})")
                print(f"[Cellpose] Chunked into {total} pieces "
                      f"({chunk_grid[0]}×{chunk_grid[1]}×{chunk_grid[2]})")

            # ── Build eval kwargs ────────────────────────────────────
            eval_kwargs = dict(
                diameter=diameter,
                flow_threshold=flow_threshold,
                cellprob_threshold=cellprob_threshold,
                min_size=min_size,
            )

            if is_2d:
                eval_kwargs["do_3D"] = False
                if has_context:
                    eval_kwargs["channel_axis"] = -1
            elif is_3d:
                eval_kwargs["z_axis"] = 0
                if native_3d:
                    # Native 3D: full volumetric convolutions
                    eval_kwargs["do_3D"] = True
                else:
                    # Slice-by-slice: segment each Z independently, then stitch
                    eval_kwargs["do_3D"] = False
                    eval_kwargs["stitch_threshold"] = stitch_threshold
                if has_context:
                    eval_kwargs["channel_axis"] = -1

            print(f"[Cellpose] eval kwargs: {eval_kwargs}")
            sys.stdout.flush()

            # ── Segment each chunk ───────────────────────────────────
            output_shape = image.shape[:3] if is_3d else image.shape[:2]
            output = np.zeros(output_shape, dtype=np.int32)
            label_offset = 0

            for idx, (chunk_data, origin) in enumerate(chunks):
                self.progress.emit(idx + 1, total)
                self.status.emit(f"Segmenting chunk {idx + 1}/{total}...")
                print(f"[Cellpose] Segmenting chunk {idx + 1}/{total}, "
                      f"shape={chunk_data.shape}...")
                sys.stdout.flush()

                masks, flows, styles = model.eval(
                    chunk_data, **eval_kwargs)

                masks = np.asarray(masks, dtype=np.int32)
                n_objects = len(np.unique(masks)) - (1 if 0 in masks else 0)
                print(f"[Cellpose] Chunk {idx + 1} done: "
                      f"{n_objects} objects, mask shape={masks.shape}")

                # Re-label so IDs don't collide across chunks
                if masks.max() > 0:
                    masks[masks > 0] += label_offset
                    label_offset = masks.max()

                # Write into output (trimming overlap)
                if is_2d:
                    oz, oy, ox = origin
                    sy, sx = masks.shape[:2]
                    out_y = slice(oy, min(oy + sy, output.shape[0]))
                    out_x = slice(ox, min(ox + sx, output.shape[1]))

                    region = output[out_y, out_x]
                    new_mask = masks[:region.shape[0], :region.shape[1]]
                    merge_mask = (region == 0) & (new_mask > 0)
                    region[merge_mask] = new_mask[merge_mask]
                    output[out_y, out_x] = region
                else:
                    oz, oy, ox = origin
                    sz, sy, sx = masks.shape[:3]
                    out_z = slice(oz, min(oz + sz, output.shape[0]))
                    out_y = slice(oy, min(oy + sy, output.shape[1]))
                    out_x = slice(ox, min(ox + sx, output.shape[2]))

                    region = output[out_z, out_y, out_x]
                    new_mask = masks[:region.shape[0],
                                     :region.shape[1],
                                     :region.shape[2]]
                    merge_mask = (region == 0) & (new_mask > 0)
                    region[merge_mask] = new_mask[merge_mask]
                    output[out_z, out_y, out_x] = region

            # ── Re-expand to match NetTracer3D's 3-D convention ──────
            if output.ndim == 2:
                output = output[np.newaxis, :, :]

            n_total = len(np.unique(output)) - (1 if 0 in output else 0)
            print(f"[Cellpose] Segmentation complete: "
                  f"{n_total} objects, output shape={output.shape}")
            self.status.emit("Segmentation complete.")
            self.finished.emit(output)

        except ImportError:
            self.error.emit(
                "cellpose is not installed.\n\n"
                "Install it with:\n"
                "  pip install cellpose\n\n"
                "For GPU support:\n"
                "  pip install cellpose[gui]\n"
                "  # and install the matching PyTorch — see\n"
                "  # https://pytorch.org/get-started/locally/")
        except Exception:
            self.error.emit(traceback.format_exc())

    # ── chunk planner ────────────────────────────────────────────────

    @staticmethod
    def _plan_chunks(volume, n_chunks, overlap=30):
        """
        Divide a volume into approximately *n_chunks* cuboid sub-volumes
        that are as close to cubic as possible, with *overlap* voxels of
        padding between neighbours so that border objects are segmented
        properly.

        Returns
        -------
        chunks : list of (sub_array, (oz, oy, ox))
        grid   : (nz, ny, nx) — how many divisions along each axis
        """
        if volume.ndim == 2 or (volume.ndim == 3 and volume.shape[-1] in (1, 2, 3, 4)):
            # Effectively 2-D (or 2-D + channels) — no Z chunking
            h, w = volume.shape[:2]
            nz = 1
            ny = max(1, round(math.sqrt(n_chunks * h / w)))
            nx = max(1, round(n_chunks / ny))
        else:
            d, h, w = volume.shape[:3]
            # Optimise grid so each chunk is as cuboid as possible
            best, best_score = (1, 1, 1), float("inf")
            for nz in range(1, n_chunks + 1):
                rem = n_chunks / nz
                for ny in range(1, int(rem) + 2):
                    nx = rem / ny
                    if nx < 1 or nx != int(nx) and abs(nx - round(nx)) > 0.5:
                        continue
                    nx = max(1, round(nx))
                    if nz * ny * nx > n_chunks * 1.25:
                        continue
                    # Score: how far from cubic each chunk is
                    cz = d / nz
                    cy = h / ny
                    cx = w / nx
                    mean = (cz + cy + cx) / 3
                    score = ((cz - mean) ** 2 + (cy - mean) ** 2 + (cx - mean) ** 2)
                    if score < best_score and nz * ny * nx >= n_chunks:
                        best, best_score = (nz, ny, nx), score
            nz, ny, nx = best

        d = volume.shape[0] if volume.ndim >= 3 and volume.shape[-1] not in (1, 2, 3, 4) else 1
        h, w = volume.shape[0] if d == 1 else volume.shape[1], \
               volume.shape[1] if d == 1 else volume.shape[2]

        def _slices(length, n):
            step = length / n
            out = []
            for i in range(n):
                s = max(0, int(i * step) - (overlap if i > 0 else 0))
                e = min(length, int((i + 1) * step) + (overlap if i < n - 1 else 0))
                out.append((s, e))
            return out

        z_slices = _slices(d, nz) if d > 1 else [(0, d)]
        y_slices = _slices(h, ny)
        x_slices = _slices(w, nx)

        chunks = []
        for zs, ze in z_slices:
            for ys, ye in y_slices:
                for xs, xe in x_slices:
                    if d == 1:
                        sub = volume[ys:ye, xs:xe] if volume.ndim <= 3 else volume[ys:ye, xs:xe, :]
                        origin = (0, ys, xs)
                    else:
                        if volume.ndim > 3:
                            sub = volume[zs:ze, ys:ye, xs:xe, :]
                        else:
                            sub = volume[zs:ze, ys:ye, xs:xe]
                        origin = (zs, ys, xs)
                    chunks.append((sub, origin))

        return chunks, (nz, ny, nx)


# ──────────────────────────────────────────────────────────────────────
#  Main dialog
# ──────────────────────────────────────────────────────────────────────

class CellposeDialog(QDialog):
    """Full-featured cellpose control panel."""

    def __init__(self, api, parent=None):
        super().__init__(parent)
        self.api = api
        self.setWindowTitle("Cellpose Segmentation")
        self.setMinimumWidth(520)
        self._worker = None
        self._thread = None
        self._custom_models = {}  # display_name → path

        self._build_ui()

    # ── UI construction ──────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)

        # ── Channel selection ────────────────────────────────────────
        chan_box = QGroupBox("Channel Selection")
        chan_grid = QGridLayout(chan_box)

        channel_items = self._channel_items()

        chan_grid.addWidget(QLabel("Image to segment:"), 0, 0)
        self.input_combo = QComboBox()
        self.input_combo.addItems(channel_items)
        chan_grid.addWidget(self.input_combo, 0, 1)

        chan_grid.addWidget(QLabel("Secondary context image:"), 1, 0)
        self.context_combo = QComboBox()
        self.context_combo.addItems(["None"] + channel_items)
        chan_grid.addWidget(self.context_combo, 1, 1)

        chan_grid.addWidget(QLabel("Send output to:"), 2, 0)
        self.output_combo = QComboBox()
        self.output_combo.addItems(channel_items)
        self.output_combo.setCurrentIndex(0)
        chan_grid.addWidget(self.output_combo, 2, 1)

        root.addWidget(chan_box)

        # ── Model selection ──────────────────────────────────────────
        model_box = QGroupBox("Model")
        model_grid = QGridLayout(model_box)

        model_grid.addWidget(QLabel("Model:"), 0, 0)
        self.model_combo = QComboBox()
        self._builtin_models = [
            "cyto3", "cyto2", "cyto", "nuclei",
            "tissuenet_cp3", "livecell_cp3",
            "yeast_PhC_cp3", "yeast_BF_cp3",
            "bact_phase_cp3", "bact_fluor_cp3",
            "deepbacs_cp3", "cyto2_cp3",
        ]
        self.model_combo.addItems(self._builtin_models)
        self.model_combo.setCurrentText("cyto3")
        model_grid.addWidget(self.model_combo, 0, 1)

        self.load_model_btn = QPushButton("Load Custom Model...")
        self.load_model_btn.clicked.connect(self._load_custom_model)
        model_grid.addWidget(self.load_model_btn, 1, 0, 1, 2)

        root.addWidget(model_box)

        # ── Parameters ───────────────────────────────────────────────
        param_box = QGroupBox("Segmentation Parameters")
        param_grid = QGridLayout(param_box)
        row = 0

        param_grid.addWidget(QLabel("Diameter (0 = auto):"), row, 0)
        self.diameter_spin = QDoubleSpinBox()
        self.diameter_spin.setRange(0, 9999)
        self.diameter_spin.setValue(30)
        self.diameter_spin.setDecimals(1)
        self.diameter_spin.setToolTip(
            "Expected cell diameter in pixels.  0 lets cellpose estimate it.")
        param_grid.addWidget(self.diameter_spin, row, 1)
        row += 1

        param_grid.addWidget(QLabel("Flow threshold:"), row, 0)
        self.flow_spin = QDoubleSpinBox()
        self.flow_spin.setRange(0, 10)
        self.flow_spin.setValue(0.4)
        self.flow_spin.setSingleStep(0.1)
        self.flow_spin.setDecimals(2)
        self.flow_spin.setToolTip(
            "Maximum allowed error of the flow field.  Increase to get more "
            "cells; decrease for stricter segmentation.")
        param_grid.addWidget(self.flow_spin, row, 1)
        row += 1

        param_grid.addWidget(QLabel("Cell probability threshold:"), row, 0)
        self.cellprob_spin = QDoubleSpinBox()
        self.cellprob_spin.setRange(-6.0, 6.0)
        self.cellprob_spin.setValue(0.0)
        self.cellprob_spin.setSingleStep(0.5)
        self.cellprob_spin.setDecimals(1)
        self.cellprob_spin.setToolTip(
            "Pixels with probability above this are considered part of a cell.  "
            "Decrease to include dimmer cells.")
        param_grid.addWidget(self.cellprob_spin, row, 1)
        row += 1

        param_grid.addWidget(QLabel("Minimum object size (px):"), row, 0)
        self.minsize_spin = QSpinBox()
        self.minsize_spin.setRange(0, 999999)
        self.minsize_spin.setValue(15)
        self.minsize_spin.setToolTip(
            "Objects smaller than this many pixels are removed.")
        param_grid.addWidget(self.minsize_spin, row, 1)
        row += 1

        param_grid.addWidget(QLabel("Stitch threshold:"), row, 0)
        self.stitch_spin = QDoubleSpinBox()
        self.stitch_spin.setRange(0, 1)
        self.stitch_spin.setValue(0.5)
        self.stitch_spin.setSingleStep(0.1)
        self.stitch_spin.setDecimals(2)
        self.stitch_spin.setToolTip(
            "IoU threshold for stitching 2-D masks across Z slices.\n"
            "Higher = stricter matching (fewer stitched objects).\n"
            "Only used for 3-D images when 'Native 3D' is off.\n"
            "0 = no stitching (each slice independent).")
        param_grid.addWidget(self.stitch_spin, row, 1)
        row += 1

        self.gpu_check = QCheckBox("Use GPU")
        self.gpu_check.setChecked(True)
        self.gpu_check.setToolTip(
            "Use CUDA GPU if available.  Falls back to CPU automatically.")
        param_grid.addWidget(self.gpu_check, row, 0)

        self.native3d_check = QCheckBox("Native 3D (Slower, Uncheck for Quick 3D)")
        self.native3d_check.setChecked(False)
        self.native3d_check.setToolTip(
            "Use cellpose's native 3-D convolutions instead of the default\n"
            "slice-by-slice + stitch approach.  MUCH slower and uses more\n"
            "memory, but can give better results for roughly spherical\n"
            "objects.  Only applies to 3-D images (auto-detected).")
        param_grid.addWidget(self.native3d_check, row, 1)
        row += 1

        root.addWidget(param_box)

        # ── Chunked / piecemeal ──────────────────────────────────────
        chunk_box = QGroupBox("Chunked Processing")
        chunk_layout = QHBoxLayout(chunk_box)

        self.chunk_check = QCheckBox("Enable")
        self.chunk_check.setToolTip(
            "Divide the image into sub-volumes and segment each one\n"
            "separately.  Useful when the full volume is too large for\n"
            "your GPU's VRAM.")
        self.chunk_check.toggled.connect(self._on_chunk_toggled)
        chunk_layout.addWidget(self.chunk_check)

        chunk_layout.addWidget(QLabel("Number of chunks:"))
        self.chunk_spin = QSpinBox()
        self.chunk_spin.setRange(2, 512)
        self.chunk_spin.setValue(8)
        self.chunk_spin.setEnabled(False)
        self.chunk_spin.setToolTip(
            "Total number of sub-volumes.  The plugin will try to make\n"
            "each chunk as cuboid as possible.")
        chunk_layout.addWidget(self.chunk_spin)

        root.addWidget(chunk_box)

        # ── Progress ─────────────────────────────────────────────────
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        root.addWidget(self.progress_bar)

        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(self.status_label)

        # ── Action buttons ───────────────────────────────────────────
        btn_row = QHBoxLayout()

        self.run_btn = QPushButton("▶  Run Segmentation")
        self.run_btn.setMinimumHeight(38)
        self.run_btn.clicked.connect(self._run)
        btn_row.addWidget(self.run_btn)

        self.gui_btn = QPushButton("Open Cellpose GUI")
        self.gui_btn.setToolTip(
            "Launch the standalone cellpose Qt GUI in a separate process.\n"
            "Requires cellpose[gui] to be installed.")
        self.gui_btn.clicked.connect(self._open_gui)
        btn_row.addWidget(self.gui_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel)
        btn_row.addWidget(self.cancel_btn)

        root.addLayout(btn_row)

    # ── helpers ──────────────────────────────────────────────────────

    def _channel_items(self):
        names = self.api.get_channel_names()
        items = []
        for i, name in enumerate(names):
            has_data = self.api.get_channel_data(i) is not None
            tag = "" if has_data else " (empty)"
            items.append(f"{i}: {name}{tag}")
        return items

    def _on_chunk_toggled(self, checked):
        self.chunk_spin.setEnabled(checked)

    def _load_custom_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Custom Cellpose Model", "",
            "All Files (*);;Model Files (*.pth *.pt *.npy)")
        if not path:
            return

        display = Path(path).stem
        # Avoid duplicate display names
        orig = display
        counter = 2
        while display in self._custom_models or display in self._builtin_models:
            display = f"{orig} ({counter})"
            counter += 1

        self._custom_models[display] = path
        self.model_combo.addItem(f"✦ {display}")
        self.model_combo.setCurrentText(f"✦ {display}")

    def _get_selected_model_info(self):
        """Return (model_type_str_or_None, custom_path_or_None)."""
        text = self.model_combo.currentText()
        if text.startswith("✦ "):
            key = text[2:]
            return None, self._custom_models.get(key)
        return text, None

    # ── run / cancel ─────────────────────────────────────────────────

    def _run(self):
        # Gather input image
        input_idx = self.input_combo.currentIndex()
        image = self.api.get_channel_data(input_idx)
        if image is None:
            QMessageBox.warning(
                self, "No Data",
                f"Channel {input_idx} has no data loaded.")
            return

        # Flatten RGB for cellpose if needed
        if image.ndim == 4 and image.shape[-1] in (3, 4):
            pass  # cellpose handles RGB natively
        elif image.ndim == 4:
            QMessageBox.warning(
                self, "Unsupported Shape",
                f"Channel shape {image.shape} is not supported.")
            return

        # Context image
        context_image = None
        ctx_idx = self.context_combo.currentIndex() - 1  # "None" is index 0
        if ctx_idx >= 0:
            context_image = self.api.get_channel_data(ctx_idx)
            if context_image is None:
                QMessageBox.warning(
                    self, "No Data",
                    f"Context channel {ctx_idx} has no data loaded.")
                return
            if context_image.shape[:3] != image.shape[:3]:
                QMessageBox.warning(
                    self, "Shape Mismatch",
                    "Context image must have the same Z/Y/X dimensions "
                    "as the primary image.")
                return

        model_type, model_path = self._get_selected_model_info()

        params = dict(
            image=image,
            context_image=context_image,
            model_type=model_type,
            model_path=model_path,
            use_gpu=self.gpu_check.isChecked(),
            diameter=self.diameter_spin.value() or None,  # 0 → auto
            flow_threshold=self.flow_spin.value(),
            cellprob_threshold=self.cellprob_spin.value(),
            native_3d=self.native3d_check.isChecked(),
            stitch_threshold=self.stitch_spin.value(),
            min_size=self.minsize_spin.value(),
            chunk_count=self.chunk_spin.value() if self.chunk_check.isChecked() else 1,
        )

        # Disable controls while running
        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Starting...")

        # Spin up worker thread
        self._thread = QThread()
        self._worker = _SegmentationWorker(params)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.status.connect(self._on_status)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)

        # Clean up thread after worker finishes
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)

        self._thread.start()

    def _cancel(self):
        if self._thread and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait(3000)
        self._reset_ui()
        self.status_label.setText("Cancelled.")

    def _on_progress(self, current, total):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)

    def _on_status(self, text):
        self.status_label.setText(text)

    def _on_finished(self, masks):
        self._reset_ui()
        if masks is None:
            self.status_label.setText("Segmentation returned no output.")
            return

        # Determine best dtype
        max_val = masks.max()
        if max_val < 256:
            masks = masks.astype(np.uint8)
        elif max_val < 65536:
            masks = masks.astype(np.uint16)
        else:
            masks = masks.astype(np.uint32)

        output_idx = self.output_combo.currentIndex()
        self.api.set_channel_data(output_idx, masks)

        n_objects = len(np.unique(masks)) - (1 if 0 in masks else 0)
        self.status_label.setText(
            f"Done — {n_objects} objects → channel {output_idx}")

    def _on_error(self, msg):
        self._reset_ui()
        self.status_label.setText("Error — see details below.")
        QMessageBox.critical(self, "Cellpose Error", msg)

    def _reset_ui(self):
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress_bar.setVisible(False)

    # ── standalone GUI ───────────────────────────────────────────────

    def _open_gui(self):
        shape = self.api.get_shape()
        use_3d = shape is not None and shape[0] > 1
        _launch_cellpose_gui_subprocess(use_3d=use_3d, parent=self)
