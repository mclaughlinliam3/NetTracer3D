import argparse
import sys
import warnings
import numpy as np
from scipy import ndimage
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Sequence
import os

try:
    import tifffile
except ImportError:
    print("ERROR: tifffile is required.  Install with:  pip install tifffile")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Per-axis 2-D EDT stacks
# ---------------------------------------------------------------------------

AXIS_CONFIG = {
    # axis: (label, collapse_axes, spacing_indices)
    0: ('Z', (1, 2), (1, 2)),   # Z: iterate Z slices, EDT in YX plane
    1: ('Y', (0, 2), (0, 2)),   # Y: iterate Y slices, EDT in ZX plane
    2: ('X', (0, 1), (0, 1)),   # X: iterate X slices, EDT in ZY plane
}

# Beyond this, thread overhead and memory contention dominate.
MAX_WORKERS = 16


def compute_axis_edt(mask: np.ndarray, axis: int,
                     voxel_spacing: tuple,
                     max_workers: Optional[int] = None) -> np.ndarray:
    """
    Compute a stack of 2-D EDTs perpendicular to the given axis.

    Uses float32 to halve memory vs float64.  Slices are computed in
    parallel via ThreadPoolExecutor; scipy's distance_transform_edt
    releases the GIL so threads run concurrently in native code.
    """
    _, _, sp_idx = AXIS_CONFIG[axis]
    plane_spacing = (voxel_spacing[sp_idx[0]], voxel_spacing[sp_idx[1]])
    edt = np.zeros(mask.shape, dtype=np.float32)
    n_slices = mask.shape[axis]

    def _process_slice(i: int) -> None:
        slc = [slice(None)] * 3
        slc[axis] = i
        slc = tuple(slc)
        plane_mask = mask[slc]
        if plane_mask.any():
            edt[slc] = ndimage.distance_transform_edt(
                plane_mask, sampling=plane_spacing
            ).astype(np.float32)

    workers = max_workers or min(n_slices, min(os.cpu_count() or 4, MAX_WORKERS))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_process_slice, i) for i in range(n_slices)]
        for f in futs:
            f.result()

    return edt


# ---------------------------------------------------------------------------
# Core normalisation
# ---------------------------------------------------------------------------

def normalize_brightness(
    img: np.ndarray,
    voxel_spacing: tuple,
    n_shells: int = 20,
    max_correction: float = 10.0,
    boost_only: bool = False,
    axes: Sequence[int] = (0, 1, 2),
) -> np.ndarray:
    """
    Normalize illumination gradients along the specified axes.

    Parameters
    ----------
    img : 3-D array (background = 0, tissue > 0).
    voxel_spacing : (z, y, x) physical voxel sizes.
    n_shells : number of concentric shells per axis.
    max_correction : cap on per-voxel correction factor.
    boost_only : if True, only brighten dim regions (never dim bright ones).
                 This preserves SNR in well-illuminated areas.
    axes : which axes to correct and in what order (default: Z→Y→X).

    Returns
    -------
    corrected : 3-D float64 array, same shape as input.
    """
    # Pad so EDT always has a background boundary
    padded = np.pad(img, pad_width=1, mode='constant', constant_values=0)
    mask = padded > 0
    result = padded.astype(np.float64)

    for axis in axes:
        if axis not in AXIS_CONFIG:
            raise ValueError(f"Invalid axis {axis}; must be 0, 1, or 2.")
        label, collapse_axes, _ = AXIS_CONFIG[axis]
        n_pos = result.shape[axis]
        print(f"\n  [{label} axis]")

        # --- 1. Per-axis 2-D EDT stack (float32) ---
        edt = compute_axis_edt(mask, axis, voxel_spacing)
        edt_tissue = edt[mask]
        max_dist = float(edt_tissue.max()) if edt_tissue.size > 0 else 0.0

        if max_dist < 1e-6:
            print(f"    WARNING: {label} EDT max ~0, skipping.")
            del edt
            continue

        # --- 2. Single-pass shell assignment via digitize ---
        shell_edges = np.linspace(0, max_dist, n_shells + 1)
        shell_edges[-1] += 1e-9

        # shell_labels: 0 = background, 1..n_shells = shell id
        shell_labels = np.digitize(edt, shell_edges)
        shell_labels[~mask] = 0
        np.clip(shell_labels, 0, n_shells, out=shell_labels)

        print(f"    EDT range: [0, {max_dist:.2f}], "
              f"{n_shells} shells (width {max_dist / n_shells:.2f})")

        del edt  # free EDT before shell loop

        # --- 3. Per-shell correction (v1-style nanmedian + broadcast) ---
        correction = np.ones(result.shape, dtype=np.float64)
        active = 0

        for s in range(1, n_shells + 1):
            shell_mask = shell_labels == s
            if not shell_mask.any():
                continue

            # Mask image, collapse to 1-D profile along target axis
            masked = np.where(shell_mask, result, np.nan)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                profile = np.nanmedian(masked, axis=collapse_axes)

            valid = ~np.isnan(profile)
            if valid.sum() < 2:
                continue

            shell_ref = np.nanmedian(profile[valid])
            if shell_ref < 1e-6:
                continue

            # Ratios: positions dimmer than reference get boosted
            ratios = np.where(valid & (profile > 1e-6),
                              shell_ref / profile, 1.0)

            if boost_only:
                ratios = np.clip(ratios, 1.0, max_correction)
            else:
                ratios = np.clip(ratios, 1.0 / max_correction, max_correction)

            # Broadcast along target axis, apply to shell voxels only
            bcast = [1, 1, 1]
            bcast[axis] = n_pos
            ratio_3d = np.broadcast_to(ratios.reshape(bcast), result.shape)
            correction[shell_mask] = ratio_3d[shell_mask]
            active += 1

        # --- 4. Apply and clean up ---
        result *= correction
        result[~mask] = 0.0
        np.clip(result, 0, None, out=result)

        corr_tissue = correction[mask]
        print(f"    Shells corrected: {active}/{n_shells}")
        print(f"    Correction range: "
              f"[{corr_tissue.min():.4f}, {corr_tissue.max():.4f}]")

        del shell_labels, correction

    # Strip padding
    return result[1:-1, 1:-1, 1:-1]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

AXIS_NAMES = {'z': 0, 'y': 1, 'x': 2}


def parse_axes(s: str) -> tuple:
    """Parse axis string like 'zyx' or 'zy' into a tuple of ints."""
    axes = []
    for ch in s.lower():
        if ch not in AXIS_NAMES:
            raise argparse.ArgumentTypeError(
                f"Invalid axis '{ch}'. Use z, y, x.")
        a = AXIS_NAMES[ch]
        if a in axes:
            raise argparse.ArgumentTypeError(f"Duplicate axis '{ch}'.")
        axes.append(a)
    return tuple(axes)


def main():
    parser = argparse.ArgumentParser(
        description="Normalize illumination gradients in a 3-D microscopy TIFF.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python normalize_3d_brightness.py input.tif output.tif \\
      --spacing-z 2.0 --spacing-xy 0.5 --shells 25

  # Correct only Z and Y, boost-only mode:
  python normalize_3d_brightness.py input.tif output.tif \\
      --axes zy --boost-only

Background must be 0, tissue > 0.
        """,
    )
    parser.add_argument("input", help="Input 3-D TIFF.")
    parser.add_argument("output", help="Output TIFF path.")
    parser.add_argument("--spacing-z", type=float, default=1.0,
                        help="Voxel size along Z (default: 1.0).")
    parser.add_argument("--spacing-xy", type=float, default=1.0,
                        help="Voxel size along X and Y (default: 1.0).")
    parser.add_argument("--shells", type=int, default=20,
                        help="Number of shells per axis (default: 20).")
    parser.add_argument("--max-correction", type=float, default=10.0,
                        help="Max per-voxel correction factor (default: 10.0).")
    parser.add_argument("--boost-only", action="store_true",
                        help="Only brighten dim regions, never dim bright ones.")
    parser.add_argument("--axes", type=parse_axes, default=(0, 1, 2),
                        help="Axes to correct and order, e.g. 'zyx' (default), "
                             "'zy', 'z'.")
    parser.add_argument("--output-dtype",
                        choices=["same", "uint8", "uint16", "float32"],
                        default="same", help="Output dtype (default: same).")

    args = parser.parse_args()

    axis_str = ''.join({0: 'Z', 1: 'Y', 2: 'X'}[a] for a in args.axes)

    print(f"\n{'='*60}")
    print("3-D Brightness Normalization (v3)")
    print(f"{'='*60}")

    raw = tifffile.imread(args.input)
    orig_dtype = raw.dtype
    img = raw.astype(np.float64)
    if img.ndim != 3:
        print(f"ERROR: Expected 3-D, got {img.ndim}-D ({img.shape})")
        sys.exit(1)

    tissue = img > 0
    spacing = (args.spacing_z, args.spacing_xy, args.spacing_xy)
    print(f"  Input:    {args.input}")
    print(f"  Shape:    {img.shape}, dtype: {orig_dtype}")
    print(f"  Tissue:   {tissue.sum():,} voxels")
    print(f"  Range:    [{img[tissue].min():.1f}, {img[tissue].max():.1f}]")
    print(f"  Spacing:  Z={args.spacing_z}, XY={args.spacing_xy}")
    print(f"  Shells:   {args.shells}, max correction: {args.max_correction}x")
    print(f"  Axes:     {axis_str}, boost-only: {args.boost_only}")

    corrected = normalize_brightness(
        img, spacing,
        n_shells=args.shells,
        max_correction=args.max_correction,
        boost_only=args.boost_only,
        axes=args.axes,
    )

    # Cast
    out_dtype = orig_dtype if args.output_dtype == "same" \
        else np.dtype(args.output_dtype)
    if np.issubdtype(out_dtype, np.integer):
        info = np.iinfo(out_dtype)
        corrected = np.clip(corrected, info.min, info.max)
    corrected = corrected.astype(out_dtype)

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(out_path), corrected, imagej=True)
    cmask = corrected > 0
    print(f"\n{'='*60}")
    print(f"  Saved:  {out_path}")
    print(f"  Shape:  {corrected.shape}, dtype: {corrected.dtype}")
    if cmask.any():
        print(f"  Range:  [{corrected[cmask].min()}, {corrected[cmask].max()}]")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()