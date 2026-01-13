"""
Automatically inspect all groups of a hdf5-file for NaNs, Infs, as well as their lenghts. Taken from Chat, tested and adapted.
"""
import argparse
import h5py
import numpy as np
from collections import Counter
from typing import Iterator, Tuple, List, Optional

def iter_datasets(g: h5py.Group, prefix: str = "") -> Iterator[Tuple[str, h5py.Dataset]]:
    """Yield (relative_path, dataset) for all datasets under group g (recursive)."""
    for k, v in g.items():
        p = f"{prefix}/{k}" if prefix else k
        if isinstance(v, h5py.Dataset):
            yield p, v
        elif isinstance(v, h5py.Group):
            yield from iter_datasets(v, p)

def infer_time_length(dsets: List[h5py.Dataset]) -> int:
    """
    Infer demo length as the MODE of the first dimension across all multi-d datasets.
    Falls back to 0 if no candidate has ndim > 0.
    """
    lengths = [d.shape[0] for d in dsets if getattr(d, "ndim", 0) > 0 and len(d.shape) > 0]
    if not lengths:
        return 0
    counts = Counter(lengths)
    mode_len = max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
    return int(mode_len)

# --------------------- Depth flatness helpers ---------------------

def _pick_time_axis(
    dset: h5py.Dataset,
    demo_len: int,
    forced_axis: Optional[int] = None,
) -> Optional[int]:
    """
    Heuristically choose the time axis for a depth dataset.
    Priority:
      1) forced_axis if provided
      2) axis with size == demo_len (if demo_len > 0 and matches uniquely)
      3) largest axis with size >= 8
    Returns None if cannot decide on an axis with size >= 2.
    """
    if forced_axis is not None:
        if 0 <= forced_axis < dset.ndim and dset.shape[forced_axis] >= 2:
            return forced_axis
        return None

    if dset.ndim == 0:
        return None

    # Prefer axis == demo_len when it uniquely matches
    if demo_len > 0:
        matches = [ax for ax, sz in enumerate(dset.shape) if sz == demo_len]
        if len(matches) == 1 and dset.shape[matches[0]] >= 2:
            return matches[0]

    # Else: largest axis with reasonable length
    candidates = [(sz, ax) for ax, sz in enumerate(dset.shape) if sz >= 8]
    if candidates:
        return max(candidates)[1]

    # Fallback: any axis with size >= 2
    candidates = [ax for ax, sz in enumerate(dset.shape) if sz >= 2]
    return candidates[0] if candidates else None

def _frame_from_dset(dset: h5py.Dataset, t: int, time_axis: int) -> np.ndarray:
    """
    Slice out frame t along time_axis, keeping spatial dims. Works for 3D/4D.
    """
    slc = [slice(None)] * dset.ndim
    slc[time_axis] = slice(t, t + 1)
    arr = dset[tuple(slc)]
    # squeeze time axis
    arr = np.squeeze(arr, axis=time_axis)
    return arr

def _is_near_constant_frame(
    frame: np.ndarray,
    tol_abs: float,
    tol_rel: float,
    pixel_frac: float,
    cv_tol: float,
    iqr_rel_tol: float,
    hist_bins: int,
    hist_top_frac: float,
    unique_frac_tol: float,
    spatial_stride: int,
) -> bool:
    """
    Decide if a depth frame is near-constant using multiple robust cues.
    Flags True if ANY of these are true:
      - abs range <= tol_abs
      - rel range (max-min)/max(|median|, eps) <= tol_rel
      - CV (std/|mean|) <= cv_tol (or std <= tol_abs if mean ~ 0)
      - >= pixel_frac within tol_abs of the median
      - relative IQR (q99-q1)/max(|median|, eps) <= iqr_rel_tol
      - a histogram bin holds >= hist_top_frac of pixels
      - unique value fraction <= unique_frac_tol
    """
    a = np.squeeze(frame)
    # ensure spatial plane (H,W) after squeeze; tolerate (H,W,1)
    if a.ndim >= 3:
        # drop trailing singleton channels if any
        while a.ndim > 2 and a.shape[-1] == 1:
            a = a[..., 0]
    if a.ndim != 2:
        return False

    if spatial_stride > 1:
        a = a[::spatial_stride, ::spatial_stride]

    # convert to float for stats
    a = a.astype(np.float32, copy=False)

    finite = np.isfinite(a)
    if not finite.any():
        return True
    a = a[finite]

    a_min = float(np.min(a))
    a_max = float(np.max(a))
    med = float(np.median(a))
    mean = float(np.mean(a))
    std = float(np.std(a))
    scale = max(abs(med), 1e-6)

    # quick unique fraction check (use small sample for large arrays)
    n = a.size
    if n > 200_000:
        # sample uniformly ~200k pixels
        idx = np.random.choice(n, 200_000, replace=False)
        au = a.reshape(-1)[idx]
    else:
        au = a.reshape(-1)

    # 7) unique fraction
    unique_frac = float(np.unique(au).size) / float(au.size)
    if unique_frac <= unique_frac_tol:
        return True

    # 1) absolute range
    if (a_max - a_min) <= tol_abs:
        return True

    # 2) relative range
    if (a_max - a_min) / scale <= tol_rel:
        return True

    # 3) coefficient of variation
    if abs(mean) > 1e-9:
        if (std / abs(mean)) <= cv_tol:
            return True
    else:
        if std <= tol_abs:
            return True

    # 4) majority near median
    within = np.abs(a - med) <= tol_abs
    if float(np.mean(within)) >= pixel_frac:
        return True

    # 5) IQR (robust spread)
    q1 = float(np.quantile(a, 0.01))
    q99 = float(np.quantile(a, 0.99))
    if (q99 - q1) / scale <= iqr_rel_tol:
        return True

    # 6) dominant histogram bin
    try:
        counts, _ = np.histogram(a, bins=max(8, min(hist_bins, a.size)))
        if counts.max() / counts.sum() >= hist_top_frac:
            return True
    except Exception:
        pass

    return False

def _depth_flat_fraction_for_dataset(
    dset: h5py.Dataset,
    demo_len: int,
    time_stride: int,
    tol_abs: float,
    tol_rel: float,
    pixel_frac: float,
    cv_tol: float,
    iqr_rel_tol: float,
    hist_bins: int,
    hist_top_frac: float,
    unique_frac_tol: float,
    spatial_stride: int,
    forced_time_axis: Optional[int],
    debug: bool,
    debug_all: bool,
    dpath: str,
) -> float:
    """
    Returns fraction of frames in a *_depth dataset that are near-constant.
    Auto-detects the time axis unless forced.
    """
    if dset.ndim < 3:
        return 0.0

    time_axis = _pick_time_axis(dset, demo_len, forced_axis=forced_time_axis)
    if time_axis is None:
        return 0.0

    T = dset.shape[time_axis]
    if T == 0:
        return 0.0

    flat = 0
    total = 0
    step = max(1, time_stride)
    for t in range(0, T, step):
        frame = _frame_from_dset(dset, t, time_axis)
        is_flat = _is_near_constant_frame(
            frame=frame,
            tol_abs=tol_abs,
            tol_rel=tol_rel,
            pixel_frac=pixel_frac,
            cv_tol=cv_tol,
            iqr_rel_tol=iqr_rel_tol,
            hist_bins=hist_bins,
            hist_top_frac=hist_top_frac,
            unique_frac_tol=unique_frac_tol,
            spatial_stride=spatial_stride,
        )
        if is_flat:
            flat += 1
        total += 1

        # Optional per-frame debug
        if debug_all:
            _print_depth_stats(frame, dpath, t, spatial_stride)

    frac = float(flat) / float(total) if total else 0.0
    # If flagged overall, print a compact summary of a mid frame to help tuning
    if debug and frac > 0.0:
        mid = min(T - 1, max(0, T // 2))
        _print_depth_stats(_frame_from_dset(dset, mid, time_axis), dpath, mid, spatial_stride)
    return frac

def _print_depth_stats(frame: np.ndarray, dpath: str, t: int, spatial_stride: int) -> None:
    """Helper to print min/max/median/std/IQR/unique frac and top hist bin share for a frame (debugging)."""
    a = np.squeeze(frame)
    if a.ndim >= 3 and a.shape[-1] == 1:
        a = a[..., 0]
    if spatial_stride > 1:
        a = a[::spatial_stride, ::spatial_stride]
    a = a.astype(np.float32, copy=False)
    a = a[np.isfinite(a)]
    if a.size == 0:
        print(f"[depth-debug] {dpath} t={t}: all non-finite")
        return
    a_min, a_max = float(np.min(a)), float(np.max(a))
    med, mean, std = float(np.median(a)), float(np.mean(a)), float(np.std(a))
    q1, q99 = float(np.quantile(a, 0.01)), float(np.quantile(a, 0.99))
    n = a.size
    au = a.reshape(-1)[: min(n, 200_000)]
    uniq_frac = float(np.unique(au).size) / float(au.size)
    counts, _ = np.histogram(a, bins=max(8, min(128, a.size)))
    top_bin = float(counts.max()) / float(counts.sum())
    print(f"[depth-debug] {dpath} t={t}: min={a_min:.6f} max={a_max:.6f} med={med:.6f} mean={mean:.6f} "
          f"std={std:.6e} iqr={q99-q1:.6e} uniq={uniq_frac:.4f} topbin={top_bin:.3f}")

# -----------------------------------------------------------------

def scan_h5(
    path: str,
    root_group: str,
    demo_prefix: str,
    short_threshold: int,
    chunk_size: int,
    verbose: bool,
    check_depth: bool,
    # depth params
    depth_tol_abs: float,
    depth_tol_rel: float,
    depth_pixel_frac: float,
    depth_flag_frac: float,
    depth_spatial_stride: int,
    depth_time_stride: int,
    depth_cv_tol: float,
    depth_iqr_rel_tol: float,
    depth_hist_bins: int,
    depth_hist_top_frac: float,
    depth_unique_frac_tol: float,
    depth_time_axis: Optional[int],
    depth_debug: bool,
    depth_debug_all: bool,
) -> int:
    problems = 0
    rows = []

    with h5py.File(path, "r") as f:
        if root_group not in f or not isinstance(f[root_group], h5py.Group):
            print(f"Root group '{root_group}' not found.")
            return 1

        root = f[root_group]
        demo_groups = [(name, obj) for name, obj in root.items()
                       if isinstance(obj, h5py.Group) and name.startswith(demo_prefix)]

        if not demo_groups:
            print(f"No groups matching '{root_group}/{demo_prefix}*' found.")
            return 0

        for demo_name, demo_grp in sorted(demo_groups, key=lambda x: x[0]):
            dsets_list = list(iter_datasets(demo_grp))
            dsets_only = [d for _, d in dsets_list]

            demo_len = infer_time_length(dsets_only)

            # scan float datasets for NaNs and Infs (chunk along axis 0)
            nan_count = 0
            inf_count = 0
            for dpath, dset in dsets_list:
                if not np.issubdtype(dset.dtype, np.floating):
                    continue
                if dset.ndim == 0:
                    arr = np.asarray(dset[()])
                    nan_count += int(np.isnan(arr).sum())
                    inf_count += int(np.isinf(arr).sum())
                else:
                    n0 = dset.shape[0]
                    if not isinstance(n0, (int, np.integer)) or n0 is None:
                        arr = dset[()]
                        nan_count += int(np.isnan(arr).sum())
                        inf_count += int(np.isinf(arr).sum())
                    else:
                        start = 0
                        while start < n0:
                            stop = min(start + chunk_size, n0)
                            slc = (slice(start, stop),) + tuple(slice(None) for _ in range(dset.ndim - 1))
                            arr = dset[slc]
                            nan_count += int(np.isnan(arr).sum())
                            inf_count += int(np.isinf(arr).sum())
                            start = stop

            depth_flag = False
            depth_flag_keys = []
            if check_depth:
                for dpath, dset in dsets_list:
                    if not dpath.endswith("_depth"):
                        continue
                    try:
                        frac_flat = _depth_flat_fraction_for_dataset(
                            dset=dset,
                            demo_len=demo_len,
                            time_stride=depth_time_stride,
                            tol_abs=depth_tol_abs,
                            tol_rel=depth_tol_rel,
                            pixel_frac=depth_pixel_frac,
                            cv_tol=depth_cv_tol,
                            iqr_rel_tol=depth_iqr_rel_tol,
                            hist_bins=depth_hist_bins,
                            hist_top_frac=depth_hist_top_frac,
                            unique_frac_tol=depth_unique_frac_tol,
                            spatial_stride=depth_spatial_stride,
                            forced_time_axis=depth_time_axis,
                            debug=depth_debug,
                            debug_all=depth_debug_all,
                            dpath=f"{root_group}/{demo_name}/{dpath}",
                        )
                    except Exception:
                        frac_flat = 1.0
                    if frac_flat >= depth_flag_frac:
                        depth_flag = True
                        depth_flag_keys.append(dpath)

            too_short = demo_len <= short_threshold
            bad = (nan_count > 0) or (inf_count > 0) or too_short or depth_flag
            problems += int(bad)

            rows.append({
                "demo": f"{root_group}/{demo_name}",
                "length": demo_len,
                "NaNs": nan_count,
                "Infs": inf_count,
                "too_short(≤%d)" % short_threshold: too_short,
                "depth_flat": depth_flag if check_depth else False,
                "depth_keys_flagged": ",".join(depth_flag_keys) if depth_flag_keys else "",
                "bad": bad,
            })

            if verbose:
                msg = (f"{root_group}/{demo_name}: len={demo_len}, NaNs={nan_count}, Infs={inf_count}, "
                       f"too_short={too_short}")
                if check_depth:
                    msg += f", depth_flat={depth_flag}"
                    if depth_flag_keys:
                        msg += f" (keys: {depth_flag_keys})"
                msg += f", bad={bad}"
                print(msg)

    # Summary
    total = len(rows)
    short = sum(r["too_short(≤%d)" % short_threshold] for r in rows)
    any_nan = sum(r["NaNs"] > 0 for r in rows)
    any_inf = sum(r["Infs"] > 0 for r in rows)
    any_depth = sum(r["depth_flat"] for r in rows) if check_depth else 0

    print("\nSummary")
    print("=======")
    print(f"Total demos: {total}")
    print(f"Bad demos:   {problems}  (too short: {short}, any NaNs: {any_nan}, any Infs: {any_inf}"
          + (f", depth flat: {any_depth}" if check_depth else "") + ")")

    header = f"{'demo':<24} {'len':>6} {'NaN':>8} {'Inf':>8} "
    if check_depth:
        header += f"{'depth':>7} "
    header += f"{'bad':>5}"
    print("\n" + header)
    print("-" * len(header))
    for r in rows[:300]:
        line = f"{r['demo']:<24} {r['length']:>6} {r['NaNs']:>8} {r['Infs']:>8} "
        if check_depth:
            line += f"{str(r['depth_flat']):>7} "
        line += f"{str(r['bad']):>5}"
        print(line)
        if check_depth and r["depth_keys_flagged"]:
            print(f"{'':<24} {'':>6} {'':>8} {'':>8} {'':>7} -> {r['depth_keys_flagged']}")
    if total > 300:
        print(f"... ({total-300} more)")

    return 0 if problems == 0 else 2

def main():
    ap = argparse.ArgumentParser(
        description="READ-ONLY: Check HDF5 demos for short length, NaNs/±Infs, and robust depth flatness (auto-detect time axis)."
    )
    ap.add_argument("path", type=str, help="Path to the HDF5 file.")
    ap.add_argument("--root-group", type=str, default="data",
                    help="Root group containing demo_* (default: 'data').")
    ap.add_argument("--demo-prefix", type=str, default="demo_",
                    help="Prefix of demo groups under --root-group (default: 'demo_').")
    ap.add_argument("--short-threshold", type=int, default=10,
                    help="Flag demos with length ≤ this value as too short (default: 10).")
    ap.add_argument("--chunk-size", type=int, default=128,
                    help="Chunk size along axis 0 when scanning float datasets (default: 128).")
    ap.add_argument("--verbose", action="store_true", help="Per-demo details.")

    # Depth flatness options (robust)
    ap.add_argument("--check-depth", action="store_true",
                    help="Enable depth flatness checks on datasets whose names end with '_depth'.")
    ap.add_argument("--depth-tol-abs", type=float, default=1e-3,
                    help="Absolute tolerance for pixel equality (default: 1e-3).")
    ap.add_argument("--depth-tol-rel", type=float, default=1e-3,
                    help="Relative tolerance on range: (max-min)/|median| (default: 1e-3).")
    ap.add_argument("--depth-pixel-frac", type=float, default=0.995,
                    help="Fraction of pixels within abs tol of median to call frame 'flat' (default: 0.995).")
    ap.add_argument("--depth-flag-frac", type=float, default=0.9,
                    help="If ≥ this fraction of frames are flat, flag the demo (default: 0.9).")
    ap.add_argument("--depth-spatial-stride", type=int, default=8,
                    help="Spatial subsampling stride (default: 8).")
    ap.add_argument("--depth-time-stride", type=int, default=1,
                    help="Temporal subsampling stride (default: 1).")
    ap.add_argument("--depth-cv-tol", type=float, default=5e-3,
                    help="Coefficient of variation (std/|mean|) threshold (default: 5e-3).")
    ap.add_argument("--depth-iqr-rel-tol", type=float, default=2e-3,
                    help="Relative IQR (q99-q1)/|median| threshold (default: 2e-3).")
    ap.add_argument("--depth-hist-bins", type=int, default=256,
                    help="Histogram bins for dominant-bin check (default: 256).")
    ap.add_argument("--depth-hist-top-frac", type=float, default=0.99,
                    help="Flag if a single histogram bin holds ≥ this fraction (default: 0.99).")
    ap.add_argument("--depth-unique-frac-tol", type=float, default=0.01,
                    help="Flag if unique value fraction ≤ this (default: 0.01).")
    ap.add_argument("--depth-time-axis", type=int, default=None,
                    help="Force time axis index for *_depth datasets (default: auto).")
    ap.add_argument("--depth-debug", action="store_true",
                    help="Print mid-frame stats for *_depth datasets that have any flat frames.")
    ap.add_argument("--depth-debug-all", action="store_true",
                    help="Print stats for every sampled frame (verbose).")

    args = ap.parse_args()

    raise SystemExit(
        scan_h5(
            path=args.path,
            root_group=args.root_group,
            demo_prefix=args.demo_prefix,
            short_threshold=args.short_threshold,
            chunk_size=args.chunk_size,
            verbose=args.verbose,
            check_depth=args.check_depth,
            depth_tol_abs=args.depth_tol_abs,
            depth_tol_rel=args.depth_tol_rel,
            depth_pixel_frac=args.depth_pixel_frac,
            depth_flag_frac=args.depth_flag_frac,
            depth_spatial_stride=args.depth_spatial_stride,
            depth_time_stride=args.depth_time_stride,
            depth_cv_tol=args.depth_cv_tol,
            depth_iqr_rel_tol=args.depth_iqr_rel_tol,
            depth_hist_bins=args.depth_hist_bins,
            depth_hist_top_frac=args.depth_hist_top_frac,
            depth_unique_frac_tol=args.depth_unique_frac_tol,
            depth_time_axis=args.depth_time_axis,
            depth_debug=args.depth_debug,
            depth_debug_all=args.depth_debug_all,
        )
    )

if __name__ == "__main__":
    main()
