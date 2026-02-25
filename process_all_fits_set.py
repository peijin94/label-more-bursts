#!/usr/bin/env python3
"""Batch-generate 0image / ivstack3 / ivmsi3 training sets from FITS files."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import numpy as np
from astropy.io import fits

from make_sample_formats import resize_2d, save_0image, save_ivstack3
from radiosoap.utils import build_label_filename, normalize_linear, normalize_log


def compute_window_starts(
    t_start: float,
    t_end: float,
    duration_s: float,
    cadence_s: float,
) -> List[float]:
    """Compute window starts using the requested cadence/duration rule."""
    if t_end <= t_start or duration_s <= 0 or cadence_s <= 0:
        return []

    # User-specified rule:
    # i = 0 .. N-1, N = (t_end - (duration-cadence) - t_start) // cadence
    n_main = int((t_end - (duration_s - cadence_s) - t_start) // cadence_s)
    starts: List[float] = []
    for i in range(max(n_main, 0)):
        s = t_start + i * cadence_s
        if s + duration_s <= t_end + 1e-6:
            starts.append(s)

    # User-specified tail rule:
    # if t_end - N*cadence > 1000, add frame [t_end-duration, t_end]
    if starts:
        tail_check = t_end - (t_start + len(starts) * cadence_s)
    else:
        tail_check = t_end - t_start

    tail_start = t_end - duration_s
    if tail_check > 1000 and tail_start >= t_start:
        if not starts or abs(tail_start - starts[-1]) > 1.0:
            starts.append(tail_start)

    return starts


def save_ivmsi3(i_arr: np.ndarray, v_arr: np.ndarray, output_npz: Path) -> None:
    """Save legacy ivmsi3 npz as (3, 300, 800): linear-I, log-I, V/I."""
    ch0 = normalize_linear(i_arr, vmin=0.0, vmax=200.0)
    ch1 = normalize_log(i_arr, vmin=0.5, vmax=200.0)
    ch2 = np.divide(
        np.asarray(v_arr, dtype=np.float32),
        np.asarray(i_arr, dtype=np.float32),
        out=np.zeros_like(v_arr, dtype=np.float32),
        where=np.isfinite(i_arr) & (np.abs(i_arr) > 1e-6),
    )
    ch2 = np.clip(ch2, -0.4, 0.4)

    msi = np.stack([resize_2d(ch0), resize_2d(ch1), resize_2d(ch2)], axis=0).astype(np.float32)
    np.savez_compressed(
        output_npz,
        msi=msi,
        channel_order=np.array(["i_linear_0_200", "i_log_0p5_200", "v_over_i_clipped_pm0p4"]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fits-dir", default="/common/lwa/spec_v2/fits", help="Directory containing FITS files.")
    parser.add_argument("--output-root", default="./output_training", help="Root output directory.")
    parser.add_argument("--pattern", default="*.fits", help="Glob pattern under fits-dir.")
    parser.add_argument("--freq-min-mhz", type=float, default=32.0, help="Minimum frequency (MHz).")
    parser.add_argument("--freq-max-mhz", type=float, default=80.0, help="Maximum frequency (MHz).")
    parser.add_argument("--duration-s", type=float, default=1920.0, help="Window duration in seconds.")
    parser.add_argument("--cadence-s", type=float, default=1800.0, help="Window cadence in seconds.")
    parser.add_argument("--max-percentile", type=float, default=99.5, help="ivstack3 panel-1 max percentile.")
    parser.add_argument("--max-files", type=int, default=0, help="Process at most this many files (0 = all).")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fits_dir = Path(args.fits_dir)
    out_root = Path(args.output_root)
    out_0image = out_root / "0image"
    out_ivstack3 = out_root / "ivstack3"
    out_ivmsi3 = out_root / "ivmsi3"
    out_0image.mkdir(parents=True, exist_ok=True)
    out_ivstack3.mkdir(parents=True, exist_ok=True)
    out_ivmsi3.mkdir(parents=True, exist_ok=True)

    fits_files = sorted(fits_dir.glob(args.pattern))
    if args.max_files > 0:
        fits_files = fits_files[: args.max_files]

    freq_min = float(min(args.freq_min_mhz, args.freq_max_mhz))
    freq_max = float(max(args.freq_min_mhz, args.freq_max_mhz))

    total_frames = 0
    for file_idx, fits_path in enumerate(fits_files, start=1):
        try:
            if fits_path.stat().st_size < 1024:
                print(f"[{file_idx}/{len(fits_files)}] skip tiny file: {fits_path.name}")
                continue

            with fits.open(fits_path, memmap=True) as hdul:
                data = hdul[0].data
                if data is None or data.ndim < 4 or data.shape[0] < 2:
                    print(f"[{file_idx}/{len(fits_files)}] skip unsupported shape: {fits_path.name}")
                    continue

                sfreq_mhz = hdul["SFREQ"].data["sfreq"].astype(np.float64) * 1e3
                f_idx = np.where((sfreq_mhz >= freq_min) & (sfreq_mhz <= freq_max))[0]
                if f_idx.size == 0:
                    print(f"[{file_idx}/{len(fits_files)}] skip no freq slice: {fits_path.name}")
                    continue

                ut = hdul["UT"].data
                abs_time_s = (ut["mjd"].astype(np.float64) - 40587.0) * 86400.0 + ut["time"].astype(np.float64) / 1000.0
                if abs_time_s.size < 2:
                    print(f"[{file_idx}/{len(fits_files)}] skip insufficient time points: {fits_path.name}")
                    continue

                t_start = float(abs_time_s[0])
                t_end = float(abs_time_s[-1])
                starts = compute_window_starts(t_start, t_end, args.duration_s, args.cadence_s)
                if not starts:
                    print(f"[{file_idx}/{len(fits_files)}] skip no valid windows: {fits_path.name}")
                    continue

                frames_written = 0
                for start_abs in starts:
                    end_abs = start_abs + args.duration_s
                    i0 = int(np.searchsorted(abs_time_s, start_abs, side="left"))
                    i1 = int(np.searchsorted(abs_time_s, end_abs, side="left"))
                    if i1 <= i0 + 1:
                        continue

                    i_arr = np.asarray(data[0, 0, f_idx, i0:i1], dtype=np.float32)
                    v_arr = np.asarray(data[1, 0, f_idx, i0:i1], dtype=np.float32)
                    f_arr = np.asarray(sfreq_mhz[f_idx], dtype=np.float32)
                    t_arr = np.asarray(abs_time_s[i0:i1] - abs_time_s[i0], dtype=np.float32)

                    if f_arr[0] > f_arr[-1]:
                        i_arr = i_arr[::-1, :]
                        v_arr = v_arr[::-1, :]
                        f_arr = f_arr[::-1]

                    dt_utc = datetime.fromtimestamp(abs_time_s[i0], tz=timezone.utc).replace(microsecond=0)
                    duration_label = int(round(args.duration_s))
                    fn_0 = build_label_filename(dt_utc, duration_label, freq_min, freq_max, "0image", "png")
                    fn_s = build_label_filename(dt_utc, duration_label, freq_min, freq_max, "ivstack3", "png")
                    fn_m = build_label_filename(dt_utc, duration_label, freq_min, freq_max, "ivmsi3", "npz")

                    p0 = out_0image / fn_0
                    ps = out_ivstack3 / fn_s
                    pm = out_ivmsi3 / fn_m
                    if (not args.overwrite) and p0.exists() and ps.exists() and pm.exists():
                        continue

                    save_0image(i_arr, p0, max_percentile=args.max_percentile)
                    save_ivstack3(i_arr, v_arr, f_arr, t_arr, ps, max_percentile=args.max_percentile)
                    save_ivmsi3(i_arr, v_arr, pm)
                    frames_written += 1

                total_frames += frames_written
                print(
                    f"[{file_idx}/{len(fits_files)}] {fits_path.name}: "
                    f"windows={len(starts)} written={frames_written}"
                )

        except Exception as exc:
            print(f"[{file_idx}/{len(fits_files)}] error {fits_path.name}: {exc}")

    print(f"Done. Files processed: {len(fits_files)} | total frames written: {total_frames}")


if __name__ == "__main__":
    main()
