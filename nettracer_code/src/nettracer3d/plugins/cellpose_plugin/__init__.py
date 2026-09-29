"""
Cellpose Segmentation Plugin for NetTracer3D
=============================================

Provides a comprehensive cellpose interface as a plugin, including:
- All major cellpose parameters exposed as controls
- Input / context / output channel selection
- Built-in and custom model support
- Chunked (piecemeal) segmentation for large images on limited GPUs,
  with optional overlapping padding so objects are not clipped at seams
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
    "version": "1.1.0",
    "author": "NetTracer3D Contributors",
    "description": (
        "Integrates Cellpose segmentation into NetTracer3D.  "
        "Run cellpose directly from the app with full parameter control, "
        "chunked processing (with padded, overlap-aware reassembly) for "
        "large volumes, and custom model support."
    ),
    "api_version": (1, 0),
    "requires": [],
    "category": "processing",
}

_api = None


# ──────────────────────────────────────────────────────────────────────
#  Chunk-grid / padding helpers
#  (module level so both the worker and the dialog agree on the layout)
# ──────────────────────────────────────────────────────────────────────

def _compute_chunk_grid(shape_zyx, n_chunks):
    """
    Choose a (nz, ny, nx) division of a (Z, Y, X) volume whose product is
    >= n_chunks and whose resulting sub-volumes are as close to cubic as
    possible.

    Returns
    -------
    (nz, ny, nx)
    """
    d, h, w = (max(1, int(s)) for s in shape_zyx)
    n_chunks = max(1, int(n_chunks))

    if n_chunks == 1:
        return (1, 1, 1)

    best = (1, 1, 1)
    best_score = float("inf")

    max_z = min(n_chunks, d)
    for nz in range(1, max_z + 1):
        max_y = min(n_chunks, h)
        for ny in range(1, max_y + 1):
            if nz * ny > n_chunks * 2:
                break
            nx = max(1, math.ceil(n_chunks / (nz * ny)))
            nx = min(nx, w)
            total = nz * ny * nx
            if total < n_chunks:
                continue
            cz, cy, cx = d / nz, h / ny, w / nx
            mean = (cz + cy + cx) / 3.0
            # How far from cubic (normalised so it is scale-independent)
            aspect = ((cz - mean) ** 2 + (cy - mean) ** 2 + (cx - mean) ** 2) \
                / (mean ** 2 + 1e-9)
            # Penalise producing many more chunks than the user asked for
            excess = (total - n_chunks) / float(n_chunks)
            score = aspect + 2.0 * excess
            if score < best_score:
                best, best_score = (nz, ny, nx), score

    return best


def _suggest_padding(shape_zyx, n_chunks, fraction=0.10,
                     min_pad=8, max_fraction=0.25):
    """
    Estimate a sensible overlap (padding) for chunked segmentation.

    The guess is a fraction of the *chunk* extent along each axis, so it
    automatically shrinks as the user asks for more chunks, and it is hard
    capped at ``max_fraction`` of the chunk extent so that the padding can
    never dwarf the chunk it is padding.  The cap matters: padding grows
    the block actually handed to cellpose, and chunking is usually being
    used precisely because that block has to stay small.

    XY and Z are estimated separately because the data may be anisotropic.

    Returns
    -------
    (pad_xy, pad_z) : ints, in voxels
    """
    if not shape_zyx:
        return 32, 4

    d, h, w = (max(1, int(s)) for s in shape_zyx)
    nz, ny, nx = _compute_chunk_grid((d, h, w), n_chunks)

    cz, cy, cx = d / nz, h / ny, w / nx

    def _est(extent, n_div):
        if n_div <= 1 or extent <= 1:
            return 0
        cap = int(extent * max_fraction)
        if cap < 1:
            return 0
        pad = int(round(extent * fraction))
        pad = max(pad, min(min_pad, cap))   # floor, but never above the cap
        return int(max(0, min(pad, cap)))

    pad_xy = max(_est(cy, ny), _est(cx, nx))
    pad_z = _est(cz, nz)

    return pad_xy, pad_z


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
    #api.print("Cellpose plugin loaded.")


def unregister(api):
    #api.print("Cellpose plugin unloaded.")
    pass


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

            # ── Padding (overlap) settings ───────────────────────────
            use_padding = bool(self.p.get("use_padding", True))
            pad_xy = int(self.p.get("pad_xy", 0)) if use_padding else 0
            pad_z = int(self.p.get("pad_z", 0)) if use_padding else 0
            pad_xy = max(0, pad_xy)
            pad_z = max(0, pad_z)
            recover_seams = bool(self.p.get("recover_seams", True))
            filter_artifacts = bool(self.p.get("filter_artifacts", True))
            artifact_percent = float(self.p.get("artifact_percent", 10.0))

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
            # NB: 2-D images are chunked too (tiled in Y/X only) — the
            # planner forces nz = 1 and pad_z = 0 for them.
            if chunk_count <= 1:
                # One "chunk" covering everything — keeps the write-back
                # loop below uniform.
                chunks = [{
                    "data": combined,
                    "core_slice": (slice(None), slice(None), slice(None)),
                    "core_origin": (0, 0, 0),
                    "pad_origin": (0, 0, 0),
                }]
                total = 1
                pad_xy = pad_z = 0
            else:
                chunks, chunk_grid = self._plan_chunks(
                    combined, chunk_count,
                    is_2d=is_2d, has_context=has_context,
                    pad_xy=pad_xy, pad_z=pad_z)
                total = len(chunks)
                pad_msg = (f"padding XY={pad_xy}, Z={pad_z}"
                           if (pad_xy or pad_z) else "no padding")
                self.status.emit(
                    f"Planned {total} chunks "
                    f"({chunk_grid[0]}×{chunk_grid[1]}×{chunk_grid[2]}), "
                    f"{pad_msg}")
                print(f"[Cellpose] Chunked into {total} pieces "
                      f"({chunk_grid[0]}×{chunk_grid[1]}×{chunk_grid[2]}), "
                      f"{pad_msg}")
                if pad_xy or pad_z:
                    print("[Cellpose] Padded regions are segmented but "
                          "discarded on reassembly"
                          + (" (objects crossing a seam are kept whole)"
                             if recover_seams else ""))

            # Seam recovery is only meaningful when there is padding to
            # recover the object from.
            recover_seams = recover_seams and (pad_xy > 0 or pad_z > 0)

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
            # Work in 3-D internally (Z=1 for 2-D images) so the write-back
            # bookkeeping only has to be written once.
            if is_2d:
                output_shape = (1,) + tuple(image.shape[:2])
            else:
                output_shape = tuple(image.shape[:3])
            output = np.zeros(output_shape, dtype=np.int32)
            label_offset = 0

            for idx, chunk in enumerate(chunks):
                chunk_data = chunk["data"]
                self.progress.emit(idx + 1, total)
                self.status.emit(f"Segmenting chunk {idx + 1}/{total}...")
                print(f"[Cellpose] Segmenting chunk {idx + 1}/{total}, "
                      f"shape={chunk_data.shape}...")
                sys.stdout.flush()

                masks, flows, styles = model.eval(
                    chunk_data, **eval_kwargs)

                masks = np.asarray(masks, dtype=np.int32)
                if masks.ndim == 2:                 # 2-D → (1, Y, X)
                    masks = masks[np.newaxis, :, :]

                n_objects = len(np.unique(masks)) - (1 if 0 in masks else 0)
                print(f"[Cellpose] Chunk {idx + 1} done: "
                      f"{n_objects} objects, mask shape={masks.shape}")

                # Re-label so IDs don't collide across chunks
                if masks.max() > 0:
                    masks[masks > 0] += label_offset
                    label_offset = int(masks.max())

                # ── Discard the padding ──────────────────────────────
                # The padded border was only ever there to give cellpose
                # the surrounding context; the labels we keep come from
                # the chunk's central (core) region, which tiles the
                # volume exactly once with no overlap.
                cz_sl, cy_sl, cx_sl = chunk["core_slice"]
                core = masks[cz_sl, cy_sl, cx_sl]

                self._merge_block(output, chunk["core_origin"], core)

                # ── Optionally keep seam-crossing objects whole ───────
                # An object straddling a core boundary was segmented in
                # full inside this chunk's padded field of view, so let
                # it claim its voxels beyond the core as long as no other
                # chunk has written there yet.  Whichever chunk reaches
                # the object first keeps it in one piece.
                if recover_seams and masks.max() > 0:
                    core_labels = np.unique(core)
                    core_labels = core_labels[core_labels > 0]
                    if core_labels.size:
                        lut = np.zeros(int(masks.max()) + 1, dtype=bool)
                        lut[core_labels] = True
                        spill = np.where(lut[masks], masks, 0)
                        self._merge_block(output, chunk["pad_origin"], spill)

            # ── Clean up seam artifacts ──────────────────────────────
            if filter_artifacts and total > 1:
                removed, n_kept, thresh = self._filter_small_objects(
                    output, artifact_percent)
                if removed:
                    msg = (f"Removed {removed} fragment(s) smaller than "
                           f"{thresh} voxels ({artifact_percent}% of the "
                           f"median object)")
                    print(f"[Cellpose] {msg}; {n_kept} objects remain")
                    self.status.emit(msg)

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

    # ── reassembly helpers ───────────────────────────────────────────

    @staticmethod
    def _merge_block(output, origin, block, absorb_fraction=0.5):
        """
        Write *block*'s labels into *output* at *origin*, filling only
        voxels that are still background.

        The important part is what happens on collision.  Two chunks that
        both see the same object will not agree on its boundary to the
        voxel, so once one chunk has claimed the object, the other chunk's
        copy has a thin rim of voxels that are still background.  Filling
        those naively stamps a *new* label along the seam — the one- or
        two-voxel strands clinging to chunk borders.

        So: if a label in *block* lands mostly on top of a label already
        present in *output*, the two are taken to be the same object and
        the leftover rim adopts the existing id instead of becoming a new
        object.  ``absorb_fraction`` is how much of the incoming label has
        to overlap before that kicks in; a genuinely distinct neighbour
        only grazes its neighbour's boundary and stays independent.
        """
        oz, oy, ox = origin
        region = output[oz:oz + block.shape[0],
                        oy:oy + block.shape[1],
                        ox:ox + block.shape[2]]
        block = block[:region.shape[0], :region.shape[1], :region.shape[2]]

        if block.size == 0 or block.max() <= 0:
            return

        fill = (region == 0) & (block > 0)
        if not fill.any():
            return

        collide = (region > 0) & (block > 0)
        if absorb_fraction and collide.any():
            b_hit = block[collide]
            e_hit = region[collide]

            # Compress the label ids before cross-tabulating — raw ids can
            # be in the millions after many chunks, so a dense id×id table
            # is not an option.
            b_vals, b_inv = np.unique(b_hit, return_inverse=True)
            e_vals, e_inv = np.unique(e_hit, return_inverse=True)
            counts = np.bincount(b_inv * len(e_vals) + e_inv)

            # Total size of each colliding label within this block
            flat = block[block > 0]
            pos = np.searchsorted(b_vals, flat)
            np.clip(pos, 0, len(b_vals) - 1, out=pos)
            valid = b_vals[pos] == flat
            totals = np.bincount(pos[valid], minlength=len(b_vals))

            # Walk pairs by descending overlap so each incoming label is
            # matched to the existing label it overlaps most.
            nz = np.nonzero(counts)[0]
            order = nz[np.argsort(-counts[nz])]
            claimed = np.zeros(len(b_vals), dtype=bool)
            src, dst = [], []
            for k in order:
                bi, ei = divmod(int(k), len(e_vals))
                if claimed[bi]:
                    continue
                claimed[bi] = True
                if counts[k] >= absorb_fraction * max(1, totals[bi]):
                    src.append(int(b_vals[bi]))
                    dst.append(int(e_vals[ei]))

            if src:
                src_arr = np.asarray(src)
                order = np.argsort(src_arr)
                src_arr = src_arr[order]
                dst_arr = np.asarray(dst)[order]

                pos = np.searchsorted(src_arr, block)
                np.clip(pos, 0, len(src_arr) - 1, out=pos)
                hit = src_arr[pos] == block
                block = np.where(hit, dst_arr[pos], block)

        region[fill] = block[fill]

    @staticmethod
    def _filter_small_objects(output, percent_of_median):
        """
        Drop labels whose volume is below *percent_of_median* percent of
        the median object volume, in place.

        The median is used rather than the mean because the fragments we
        are trying to remove are both numerous and tiny, which drags a
        mean down toward them; and because real object-size distributions
        are right-skewed, so a large share of perfectly good objects sit
        below the mean anyway.

        Returns (n_removed, n_remaining, threshold_voxels).
        """
        counts = np.bincount(output.ravel())
        if counts.size < 2:
            return 0, 0, 0
        counts[0] = 0                       # ignore background

        labels = np.nonzero(counts)[0]
        if labels.size < 2:
            return 0, int(labels.size), 0

        sizes = counts[labels]
        threshold = int(np.median(sizes) * (percent_of_median / 100.0))
        if threshold < 1:
            return 0, int(labels.size), 0

        small = labels[sizes < threshold]
        if small.size == 0:
            return 0, int(labels.size), threshold

        drop = np.zeros(counts.size, dtype=bool)
        drop[small] = True
        output[drop[output]] = 0

        return int(small.size), int(labels.size - small.size), threshold

    # ── chunk planner ────────────────────────────────────────────────

    @staticmethod
    def _plan_chunks(volume, n_chunks, is_2d=False, has_context=False,
                     pad_xy=0, pad_z=0):
        """
        Divide a volume into approximately *n_chunks* sub-volumes that are
        as close to cubic as possible.

        Each chunk has two extents:

          * the **core** — the chunk's own exclusive slab.  The cores tile
            the volume exactly once, with no overlap and no gaps.
          * the **padded block** — the core grown by ``pad_xy`` voxels in
            Y/X and ``pad_z`` voxels in Z (clipped at the volume border).
            This is what actually gets handed to cellpose, so objects
            sitting on a core boundary are seen in full rather than being
            cut off by an artificial image edge.

        The padding is thrown away when the volume is reassembled: only
        the labels falling inside the core are written back.

        Returns
        -------
        chunks : list of dicts with keys
                 'data'        padded sub-array to feed cellpose
                 'core_slice'  (z, y, x) slices selecting the core out of
                               a mask of the padded block
                 'core_origin' (oz, oy, ox) of the core in the full volume
                 'pad_origin'  (pz, py, px) of the padded block
        grid   : (nz, ny, nx) — how many divisions along each axis
        """
        # Trailing channel axis (from a stacked context image) rides along
        # with the spatial slicing and is not chunked.
        spatial = volume.shape[:-1] if has_context else volume.shape

        if is_2d:
            d = 1
            h, w = spatial[:2]
        else:
            d, h, w = spatial[:3]

        nz, ny, nx = _compute_chunk_grid((d, h, w), n_chunks)
        if is_2d or d <= 1:
            nz = 1

        pad_xy = max(0, int(pad_xy))
        pad_z = 0 if nz <= 1 else max(0, int(pad_z))

        def _axis(length, n, pad):
            """Yield (core_start, core_end, pad_start, pad_end) per division."""
            out = []
            for i in range(n):
                s = int(round(i * length / n))
                e = length if i == n - 1 else int(round((i + 1) * length / n))
                if e <= s:
                    continue
                out.append((s, e, max(0, s - pad), min(length, e + pad)))
            return out

        z_ax = _axis(d, nz, pad_z) if d > 1 else [(0, max(1, d), 0, max(1, d))]
        y_ax = _axis(h, ny, pad_xy)
        x_ax = _axis(w, nx, pad_xy)

        chunks = []
        for zs, ze, pzs, pze in z_ax:
            for ys, ye, pys, pye in y_ax:
                for xs, xe, pxs, pxe in x_ax:
                    if is_2d or d <= 1:
                        sub = volume[pys:pye, pxs:pxe]
                        core_slice = (slice(0, 1),
                                      slice(ys - pys, ye - pys),
                                      slice(xs - pxs, xe - pxs))
                        core_origin = (0, ys, xs)
                        pad_origin = (0, pys, pxs)
                    else:
                        sub = volume[pzs:pze, pys:pye, pxs:pxe]
                        core_slice = (slice(zs - pzs, ze - pzs),
                                      slice(ys - pys, ye - pys),
                                      slice(xs - pxs, xe - pxs))
                        core_origin = (zs, ys, xs)
                        pad_origin = (pzs, pys, pxs)

                    chunks.append({
                        "data": sub,
                        "core_slice": core_slice,
                        "core_origin": core_origin,
                        "pad_origin": pad_origin,
                    })

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
        # Padding boxes track the chunk count automatically until the user
        # types their own value into one of them.
        self._padding_is_auto = True

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
        self.input_combo.currentIndexChanged.connect(self._on_input_changed)
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
        chunk_grid = QGridLayout(chunk_box)
        crow = 0

        self.chunk_check = QCheckBox("Enable")
        self.chunk_check.setToolTip(
            "Divide the image into sub-volumes and segment each one\n"
            "separately.  Useful when the full volume is too large for\n"
            "your GPU's VRAM.")
        self.chunk_check.toggled.connect(self._on_chunk_toggled)
        chunk_grid.addWidget(self.chunk_check, crow, 0)

        chunk_grid.addWidget(QLabel("Number of chunks:"), crow, 1)
        self.chunk_spin = QSpinBox()
        self.chunk_spin.setRange(2, 512)
        self.chunk_spin.setValue(8)
        self.chunk_spin.setEnabled(False)
        self.chunk_spin.setToolTip(
            "Total number of sub-volumes.  The plugin will try to make\n"
            "each chunk as cuboid as possible.")
        self.chunk_spin.valueChanged.connect(self._on_chunk_count_changed)
        chunk_grid.addWidget(self.chunk_spin, crow, 2)
        crow += 1

        self.pad_check = QCheckBox("Use padding")
        self.pad_check.setChecked(True)
        self.pad_check.setEnabled(False)
        self.pad_check.setToolTip(
            "Give each chunk an overlapping border of surrounding voxels\n"
            "when it is sent to cellpose, then throw that border away when\n"
            "the volume is put back together.\n\n"
            "Objects sitting on a chunk boundary are then segmented with\n"
            "their real surroundings visible instead of being cut off by an\n"
            "artificial image edge, so they are far less likely to be\n"
            "clipped.\n\n"
            "Note that padding enlarges the block sent to cellpose, so it\n"
            "raises peak VRAM per chunk — reduce it (or add chunks) if you\n"
            "run out of memory.")
        self.pad_check.toggled.connect(self._on_pad_toggled)
        chunk_grid.addWidget(self.pad_check, crow, 0)

        chunk_grid.addWidget(QLabel("XY padding (voxels):"), crow, 1)
        self.pad_xy_spin = QSpinBox()
        self.pad_xy_spin.setRange(0, 4096)
        self.pad_xy_spin.setEnabled(False)
        self.pad_xy_spin.setToolTip(
            "Overlap added on each side of every chunk in Y and X.\n\n"
            "Auto-filled from the image size and chunk count, and capped so\n"
            "the padding can never exceed half the chunk itself.  Editing it\n"
            "by hand switches this field to manual — press 'Auto' to hand it\n"
            "back to the estimator.\n\n"
            "A good manual value is roughly one object diameter.")
        self.pad_xy_spin.valueChanged.connect(self._on_padding_edited)
        chunk_grid.addWidget(self.pad_xy_spin, crow, 2)
        crow += 1

        self.pad_auto_btn = QPushButton("Auto")
        self.pad_auto_btn.setEnabled(False)
        self.pad_auto_btn.setToolTip(
            "Re-estimate both padding values from the current image size\n"
            "and chunk count, and resume updating them automatically.")
        self.pad_auto_btn.clicked.connect(self._reset_padding_to_auto)
        chunk_grid.addWidget(self.pad_auto_btn, crow, 0)

        chunk_grid.addWidget(QLabel("Z padding (voxels):"), crow, 1)
        self.pad_z_spin = QSpinBox()
        self.pad_z_spin.setRange(0, 4096)
        self.pad_z_spin.setEnabled(False)
        self.pad_z_spin.setToolTip(
            "Overlap added above and below every chunk in Z.\n\n"
            "Kept separate from XY because the data may be anisotropic —\n"
            "with thick slices you usually want far less padding here.\n"
            "Ignored when the chunk grid does not divide Z.")
        self.pad_z_spin.valueChanged.connect(self._on_padding_edited)
        chunk_grid.addWidget(self.pad_z_spin, crow, 2)
        crow += 1

        self.seam_check = QCheckBox("Keep objects crossing chunk seams whole")
        self.seam_check.setChecked(True)
        self.seam_check.setEnabled(False)
        self.seam_check.setToolTip(
            "An object straddling a chunk boundary was segmented in full\n"
            "inside the padded region, so let the first chunk that sees it\n"
            "keep the whole object rather than splitting it at the seam.\n\n"
            "Turn this off for a strict crop, where each chunk contributes\n"
            "only the labels inside its own core region.\n"
            "Requires padding.")
        chunk_grid.addWidget(self.seam_check, crow, 0, 1, 3)
        crow += 1

        self.artifact_check = QCheckBox("Filter seam artifacts")
        self.artifact_check.setChecked(True)
        self.artifact_check.setEnabled(False)
        self.artifact_check.setToolTip(
            "After the volume is reassembled, drop labels far smaller than\n"
            "a typical object.  These are usually thin slivers left along a\n"
            "chunk seam where two chunks segmented the same object and\n"
            "disagreed about its boundary by a voxel or two.\n\n"
            "Turn this off if you would rather keep every label and filter\n"
            "the noise yourself downstream — expect some very small\n"
            "fragments along the chunk borders if you do.")
        self.artifact_check.toggled.connect(self._on_artifact_toggled)
        chunk_grid.addWidget(self.artifact_check, crow, 0)

        chunk_grid.addWidget(QLabel("Drop below (% of median):"), crow, 1)
        self.artifact_spin = QDoubleSpinBox()
        self.artifact_spin.setRange(0.1, 100.0)
        self.artifact_spin.setValue(10.0)
        self.artifact_spin.setSingleStep(5.0)
        self.artifact_spin.setDecimals(1)
        self.artifact_spin.setEnabled(False)
        self.artifact_spin.setToolTip(
            "An object is discarded if its volume is below this percentage\n"
            "of the median object volume.\n\n"
            "The median is used rather than the mean because seam fragments\n"
            "are numerous and tiny, which drags a mean down toward them, and\n"
            "because object sizes are right-skewed — plenty of legitimate\n"
            "objects sit below the mean.  10% is conservative; raise it if\n"
            "fragments survive, lower it if real objects are disappearing.")
        chunk_grid.addWidget(self.artifact_spin, crow, 2)

        root.addWidget(chunk_box)

        # Seed the padding estimates from the image that is loaded now.
        self._reset_padding_to_auto()

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

    # ── chunking / padding ───────────────────────────────────────────

    def _on_chunk_toggled(self, checked):
        self.chunk_spin.setEnabled(checked)
        self.pad_check.setEnabled(checked)
        self.pad_auto_btn.setEnabled(checked)
        self.artifact_check.setEnabled(checked)
        self._on_artifact_toggled(self.artifact_check.isChecked())
        self._on_pad_toggled(self.pad_check.isChecked())

    def _on_artifact_toggled(self, checked):
        self.artifact_spin.setEnabled(
            checked and self.chunk_check.isChecked())

    def _on_pad_toggled(self, checked):
        on = checked and self.chunk_check.isChecked()
        self.pad_xy_spin.setEnabled(on)
        self.pad_z_spin.setEnabled(on)
        self.pad_auto_btn.setEnabled(on)
        self.seam_check.setEnabled(on)

    def _on_chunk_count_changed(self, _value):
        # Sensible padding depends on how big a chunk ends up being, so it
        # has to follow the chunk count — unless the user took the wheel.
        if self._padding_is_auto:
            self._apply_auto_padding()

    def _on_input_changed(self, _index):
        if self._padding_is_auto:
            self._apply_auto_padding()

    def _on_padding_edited(self, _value):
        # Only fires for genuine user edits; programmatic updates below are
        # made with signals blocked.
        self._padding_is_auto = False

    def _reset_padding_to_auto(self):
        self._padding_is_auto = True
        self._apply_auto_padding()

    def _apply_auto_padding(self):
        """Fill the padding boxes from the image size and chunk count."""
        shape = self._current_shape()
        if shape is None:
            pad_xy, pad_z = 32, 4
        else:
            pad_xy, pad_z = _suggest_padding(shape, self.chunk_spin.value())

        for spin, value in ((self.pad_xy_spin, pad_xy),
                            (self.pad_z_spin, pad_z)):
            blocked = spin.blockSignals(True)
            spin.setValue(int(value))
            spin.blockSignals(blocked)

    def _current_shape(self):
        """Best available (Z, Y, X) of the image to be segmented."""
        try:
            data = self.api.get_channel_data(self.input_combo.currentIndex())
            if data is not None and getattr(data, "ndim", 0) >= 2:
                shape = tuple(int(s) for s in data.shape[:3])
                if len(shape) == 2:          # (Y, X) → (1, Y, X)
                    shape = (1,) + shape
                return shape
        except Exception:
            pass

        try:
            shape = self.api.get_shape()
            if shape:
                shape = tuple(int(s) for s in shape[:3])
                if len(shape) == 2:
                    shape = (1,) + shape
                if len(shape) == 3:
                    return shape
        except Exception:
            pass

        return None

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
            use_padding=self.pad_check.isChecked(),
            pad_xy=self.pad_xy_spin.value(),
            pad_z=self.pad_z_spin.value(),
            recover_seams=self.seam_check.isChecked(),
            filter_artifacts=self.artifact_check.isChecked(),
            artifact_percent=self.artifact_spin.value(),
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
