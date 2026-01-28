#!/usr/bin/env python3
"""
Re-draw the Tc(Ja) overlay plot from saved results (no MC re-run).

Inputs:
  - JSON metadata produced by plot_phase_diagram_mc.py
  - CSV results produced by plot_phase_diagram_mc.py

This script recomputes ONLY the analytic Tc(Ja) curve at high resolution and overlays it
on the stored MC points (with error bars).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from typing import Dict, List, Tuple

try:
    from tqdm import tqdm  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    def tqdm(x, **_kwargs):  # type: ignore
        return x


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--meta",
        type=str,
        default="phase_diagram_Tc_vs_Ja_mc.json",
        help="Metadata JSON from plot_phase_diagram_mc.py",
    )
    p.add_argument(
        "--csv",
        type=str,
        default="",
        help="CSV from plot_phase_diagram_mc.py (default: same basename as --meta, but .csv)",
    )
    p.add_argument(
        "--out",
        type=str,
        default="phase_diagram_Tc_vs_Ja_mc_redraw.png",
        help="Output PNG path",
    )
    p.add_argument("--analytic-num", type=int, default=1000, help="Analytic Ja grid points")
    p.add_argument("--Tc-max", type=float, default=6.0, help="Y-axis max")
    p.add_argument(
        "--drop-top-margin",
        type=float,
        default=0.05,
        help="Drop points with Tc >= Tc_max - margin (avoid points stuck at the ceiling).",
    )
    return p.parse_args()


def load_meta(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)


def load_mc_csv(path: str) -> Tuple[List[float], List[float], List[float], List[float]]:
    xs: List[float] = []
    ys: List[float] = []
    ylo: List[float] = []
    yhi: List[float] = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            xs.append(float(row["Ja"]))
            ys.append(float(row["Tc_mc"]))
            ylo.append(float(row.get("Tc_mc_lo", "nan")))
            yhi.append(float(row.get("Tc_mc_hi", "nan")))
    return xs, ys, ylo, yhi


def main() -> None:
    args = parse_args()
    meta = load_meta(args.meta)

    csv_path = args.csv.strip()
    if not csv_path:
        base, _ = os.path.splitext(args.meta)
        csv_path = base + ".csv"

    xs, ys, ylo, yhi = load_mc_csv(csv_path)

    J = float(meta["model"]["J"])
    n = int(meta["model"]["n"])
    phis = tuple(float(x) for x in meta["model"]["phis"])
    Ja_min = float(meta["Ja_grid"]["min"])
    Ja_max = float(meta["Ja_grid"]["max"])

    from angle_modulated_triangular_ising_closedform_vs_mc import solve_Tc

    # Dense analytic curve
    ax_all: List[float] = []
    ay_all: List[float] = []
    npts = max(2, int(args.analytic_num))
    dJa = (Ja_max - Ja_min) / (npts - 1)
    for i in tqdm(range(npts), total=npts, desc="Analytic curve"):
        Ja = Ja_min + i * dJa
        try:
            Tc = solve_Tc(J, Ja, n, phis, T_lo=1e-8, T_hi=max(10.0, args.Tc_max * 2.0))
        except Exception:
            continue
        if math.isfinite(Tc):
            ax_all.append(Ja)
            ay_all.append(Tc)

    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(7.0, 4.5), dpi=160)

    # Phase region coloring (FE / PM / ST) using the analytic curve as the boundary.
    # Convention: PM above Tc(Ja/J), ordered phases below Tc(Ja/J) split at Ja/J=2 (paper case, n=2).
    if ax_all:
        x = np.asarray(ax_all, dtype=float) / max(J, 1e-300)
        y = np.asarray(ay_all, dtype=float)
        # Clamp to plotting range so fills don't extend beyond axes limits.
        y = np.clip(y, 0.0, float(args.Tc_max))

        Ja_star = 2.0  # i.e. Ja = 2J
        fe_color = "#f26d6d"  # saturated red
        pm_color = "#9b7bd4"  # saturated purple
        st_color = "#6fb1f2"  # saturated blue

        axh = plt.gca()
        axh.fill_between(x, 0.0, y, where=(x <= Ja_star), color=fe_color, alpha=0.48, linewidth=0, zorder=0)
        axh.fill_between(x, 0.0, y, where=(x >= Ja_star), color=st_color, alpha=0.48, linewidth=0, zorder=0)
        axh.fill_between(x, y, float(args.Tc_max), color=pm_color, alpha=0.38, linewidth=0, zorder=0)

    # Drop MC points outside the displayed Tc range to avoid "stuck at the top" clipping artifacts.
    xs_f = []
    ys_f = []
    ylo_f = []
    yhi_f = []
    for x, y, lo, hi in zip(xs, ys, ylo, yhi):
        if not math.isfinite(y):
            continue
        # Drop points outside visible range, and also those that would clip at the top edge.
        if y < 0.0:
            continue
        if y >= (args.Tc_max - args.drop_top_margin):
            continue
        if math.isfinite(hi) and hi >= args.Tc_max:
            continue
        xs_f.append(x / max(J, 1e-300))
        ys_f.append(y)
        ylo_f.append(lo)
        yhi_f.append(hi)

    yerr_low = [max(0.0, y - lo) if math.isfinite(lo) else 0.0 for y, lo in zip(ys_f, ylo_f)]
    yerr_hi = [max(0.0, hi - y) if math.isfinite(hi) else 0.0 for y, hi in zip(ys_f, yhi_f)]
    plt.errorbar(
        xs_f,
        ys_f,
        yerr=[yerr_low, yerr_hi],
        fmt=".",
        markersize=3.0,
        capsize=2.0,
        label="MC (\u03c7 peak, L=36)",
    )
    # Plot the analytic curve, but avoid clutter from points stuck at the top edge.
    ax_plot = []
    ay_plot = []
    for x0, y0 in zip(ax_all, ay_all):
        if y0 < (args.Tc_max - args.drop_top_margin):
            ax_plot.append(x0 / max(J, 1e-300))
            ay_plot.append(y0)
    plt.plot(ax_plot, ay_plot, "-", linewidth=1.5, label="Critical manifold curve")
    # Indicate the true analytic endpoint at Ja/J=2: Tc=0 exactly at Ja/J=2.
    # The approach is logarithmically slow, so on a coarse Tc axis it can look like the cusp
    # bottoms out above zero. Add a short vertical guide at Ja/J=2.
    if (Ja_min / max(J, 1e-300)) <= 2.0 <= (Ja_max / max(J, 1e-300)):
        plt.plot([2.0, 2.0], [0.0, 0.8], color="C1", linewidth=1.5, alpha=0.9)

    plt.xlim(Ja_min / max(J, 1e-300), Ja_max / max(J, 1e-300))
    plt.ylim(0.0, args.Tc_max)
    plt.xlabel(r"$J_a/J$")
    plt.ylabel(r"$T_c$")
    plt.title(r"Angle-modulated triangular Ising phase diagram: exact manifold + MC")
    # Phase labels (match common notation: FE / PM / ST).
    axh = plt.gca()
    label_kw = dict(fontsize=12, fontweight="bold", alpha=0.85, ha="center", va="center")
    axh.text(0.22, 0.30, "FE", transform=axh.transAxes, **label_kw)
    axh.text(0.52, 0.78, "PM", transform=axh.transAxes, **label_kw)
    axh.text(0.82, 0.50, "ST", transform=axh.transAxes, **label_kw)
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best", fontsize=9, frameon=False)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    plt.close()

    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()

