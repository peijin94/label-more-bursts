#!/usr/bin/python3
"""Generate sample 0image, ivstack3, and ivmsi3 files from an LWA FITS file."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from radiosoap.utils import (
    build_label_filename,
    load_lwa_fits_iv_window,
    normalize_linear,
    normalize_log,
)


def save_0image(i_arr: np.ndarray, output_png: Path) -> None:
    """Save format-1 plain spectrum image."""
    p1 = float(np.nanpercentile(i_arr, 1.0))
    p99 = float(np.nanpercentile(i_arr, 99.9))
    if p99 <= p1:
        p99 = p1 + 1e-6
    fig = plt.figure(figsize=(6, 3), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(
        i_arr,
        origin="lower",
        aspect="auto",
        cmap="CMRmap",
        vmin=p1,
        vmax=p99,
        interpolation="nearest",
    )
    ax.axis("off")
    fig.savefig(output_png, dpi=100, bbox_inches=None, pad_inches=0)
    plt.close(fig)


def save_ivstack3(
    i_arr: np.ndarray,
    v_arr: np.ndarray,
    freq_mhz: np.ndarray,
    time_s: np.ndarray,
    output_png: Path,
    max_percentile: float = 99.5,
) -> None:
    """Save format-2.2 stacked I/V image with 3 panels."""
    pos_i = i_arr[np.isfinite(i_arr) & (i_arr > 0)]
    if pos_i.size == 0:
        i_max_pct = 1.0
    else:
        i_max_pct = float(np.nanpercentile(pos_i, max_percentile))
    i_max_pct = max(i_max_pct, 0.55)

    fig, axes = plt.subplots(3, 1, figsize=(6, 9), dpi=100)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0)

    axes[0].imshow(
        i_arr,
        origin="lower",
        aspect="auto",
        cmap="CMRmap",
        norm=LogNorm(vmin=0.5, vmax=i_max_pct),
        interpolation="nearest",
    )

    axes[1].imshow(
        i_arr,
        origin="lower",
        aspect="auto",
        cmap="CMRmap",
        norm=LogNorm(vmin=0.5, vmax=200.0),
        interpolation="nearest",
    )

    vi_ratio = np.divide(
        v_arr,
        i_arr,
        out=np.zeros_like(v_arr, dtype=np.float32),
        where=np.isfinite(i_arr) & (np.abs(i_arr) > 1e-6),
    )
    axes[2].imshow(
        vi_ratio,
        origin="lower",
        aspect="auto",
        cmap="RdBu_r",
        vmin=-0.4,
        vmax=0.4,
        interpolation="nearest",
    )
    for ax in axes:
        ax.axis("off")

    fig.savefig(output_png, dpi=100, bbox_inches=None, pad_inches=0)
    plt.close(fig)


def save_ivmsi3(i_arr: np.ndarray, v_arr: np.ndarray, freq_mhz: np.ndarray, time_s: np.ndarray, output_npz: Path) -> None:
    """Save format-3 multispectral 3-channel npz."""
    ch0 = normalize_linear(i_arr, vmin=0.0, vmax=200.0)
    ch1 = normalize_log(i_arr, vmin=0.5, vmax=200.0)
    ch2 = np.clip(np.asarray(v_arr, dtype=np.float32), -0.5, 0.5)
    msi = np.stack([ch0, ch1, ch2], axis=0).astype(np.float32)

    np.savez_compressed(
        output_npz,
        msi=msi,
        freq_mhz=np.asarray(freq_mhz, dtype=np.float32),
        time_s=np.asarray(time_s, dtype=np.float32),
        channel_order=np.array(["i_linear_0_200", "i_log_0p5_200", "pol_v_clipped_pm0p5"]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fits",
        default="/common/lwa/spec_v2/fits/20240527.fits",
        help="Source FITS file path.",
    )
    parser.add_argument("--outdir", default="./outtmp", help="Output directory.")
    parser.add_argument("--start-offset-s", type=float, default=1800.0, help="Window start offset (s) from file start.")
    parser.add_argument("--duration-s", type=float, default=1800.0, help="Window duration in seconds.")
    parser.add_argument("--freq-min-mhz", type=float, default=30.0, help="Min frequency in MHz.")
    parser.add_argument("--freq-max-mhz", type=float, default=80.0, help="Max frequency in MHz.")
    parser.add_argument(
        "--max-percentile",
        type=float,
        default=99.5,
        help="Percentile used for ivstack3 panel-1 vmax.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    win = load_lwa_fits_iv_window(
        fits_path=args.fits,
        start_offset_s=args.start_offset_s,
        duration_s=args.duration_s,
        freq_min_mhz=args.freq_min_mhz,
        freq_max_mhz=args.freq_max_mhz,
    )

    dt_utc = win["window_start_utc"]
    duration_label = int(round(args.duration_s))
    fmin_label = float(min(args.freq_min_mhz, args.freq_max_mhz))
    fmax_label = float(max(args.freq_min_mhz, args.freq_max_mhz))

    fn_0image = build_label_filename(
        dt_utc,
        duration_s=duration_label,
        start_freq_mhz=fmin_label,
        end_freq_mhz=fmax_label,
        fmt="0image",
        ext="png",
    )
    fn_ivstack3 = build_label_filename(
        dt_utc,
        duration_s=duration_label,
        start_freq_mhz=fmin_label,
        end_freq_mhz=fmax_label,
        fmt="ivstack3",
        ext="png",
    )
    fn_ivmsi3 = build_label_filename(
        dt_utc,
        duration_s=duration_label,
        start_freq_mhz=fmin_label,
        end_freq_mhz=fmax_label,
        fmt="ivmsi3",
        ext="npz",
    )

    out_0image = outdir / fn_0image
    out_ivstack3 = outdir / fn_ivstack3
    out_ivmsi3 = outdir / fn_ivmsi3

    save_0image(win["i"], out_0image)
    save_ivstack3(
        win["i"],
        win["v"],
        win["freq_mhz"],
        win["time_s"],
        out_ivstack3,
        max_percentile=args.max_percentile,
    )
    save_ivmsi3(win["i"], win["v"], win["freq_mhz"], win["time_s"], out_ivmsi3)

    print(f"Wrote: {out_0image}")
    print(f"Wrote: {out_ivstack3}")
    print(f"Wrote: {out_ivmsi3}")
    print(
        "Shapes: "
        f"I={win['i'].shape}, "
        f"V={win['v'].shape}, "
        f"ivmsi3(msi)={np.load(out_ivmsi3)['msi'].shape}"
    )


if __name__ == "__main__":
    main()
