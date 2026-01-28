#!/usr/bin/env python3
"""
Plot a MC-only phase diagram Tc(Ja) for the angle-modulated triangular Ising model.

This script DOES NOT use the analytic Tc solver. It estimates Tc(Ja) from Monte Carlo scans
at fixed lattice size using the susceptibility peak (with a local parabolic refinement).

Output: PNG dot plot with axes
  Ja in [0, Ja_max]
  Tc in [0, Tc_max]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from typing import Dict, List, Tuple

try:
    from tqdm import tqdm  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    def tqdm(x, **_kwargs):  # type: ignore
        return x


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--J", type=float, default=1.0)
    p.add_argument("--n", type=int, default=2)
    p.add_argument("--phis", type=str, default="default")

    p.add_argument("--Ja-min", type=float, default=0.0)
    p.add_argument("--Ja-max", type=float, default=4.5)
    p.add_argument("--num", type=int, default=91, help="Number of Ja points.")

    p.add_argument("--Tc-max", type=float, default=6.0)

    # MC controls (kept modest; total runtime can be ~1h depending on cores).
    p.add_argument("--L", type=int, default=48)
    p.add_argument("--therm", type=int, default=2500, help="Thermalization updates per temperature.")
    p.add_argument("--meas", type=int, default=3500, help="Measurement updates per temperature.")
    p.add_argument("--thin", type=int, default=4, help="Record every `thin` updates.")
    p.add_argument("--replicas", type=int, default=8)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--nprocs", type=int, default=0, help="0=all cores")
    p.add_argument("--update", type=str, default="wolff", choices=["wolff", "metropolis"])

    # Scan strategy
    p.add_argument("--Tmin", type=float, default=0.4)
    p.add_argument("--Tmax", type=float, default=6.0)
    p.add_argument("--coarse-step", type=float, default=0.25)
    p.add_argument("--refine-window", type=float, default=0.5)
    p.add_argument("--refine-step", type=float, default=0.05)

    p.add_argument("--out", type=str, default="phase_diagram_Tc_vs_Ja_mc.png")
    p.add_argument(
        "--data-out",
        type=str,
        default="",
        help="CSV output path (default: same as --out with .csv).",
    )
    p.add_argument(
        "--meta-out",
        type=str,
        default="",
        help="JSON metadata output path (default: same as --out with .json).",
    )
    p.add_argument(
        "--analytic-num",
        type=int,
        default=901,
        help="Number of Ja points for smooth analytic overlay curve.",
    )
    return p.parse_args()


def parse_phis(spec: str) -> Tuple[float, float, float]:
    if spec.strip().lower() == "default":
        from angle_modulated_triangular_ising_closedform_vs_mc import default_phis

        return default_phis()
    parts = [s.strip() for s in spec.split(",") if s.strip()]
    if len(parts) != 3:
        raise ValueError("phis must be 'default' or 3 comma-separated radians")
    return (float(parts[0]), float(parts[1]), float(parts[2]))


def frange(a: float, b: float, step: float) -> List[float]:
    if step <= 0:
        raise ValueError("step must be > 0")
    n = int(math.floor((b - a) / step)) + 1
    return [a + i * step for i in range(n)]


def main() -> None:
    args = parse_args()
    phis = parse_phis(args.phis)

    from angle_modulated_triangular_ising_closedform_vs_mc import (
        MCConfig,
        estimate_Tc_from_susceptibility,
        run_mc_scan,
    )

    cfg = MCConfig(
        L=args.L,
        therm_sweeps=args.therm,
        meas_sweeps=args.meas,
        thin=args.thin,
        seed=args.seed,
        replicas=args.replicas,
        nprocs=args.nprocs,
        update_method=args.update,
    )

    # Ja grid
    if args.num < 2:
        Ja_vals = [float(args.Ja_min)]
    else:
        step = (args.Ja_max - args.Ja_min) / (args.num - 1)
        Ja_vals = [args.Ja_min + i * step for i in range(args.num)]

    xs: List[float] = []
    ys: List[float] = []
    ylos: List[float] = []
    yhis: List[float] = []
    skipped = 0

    # Start with a broad guess for Tc (triangular Ising ~3.64 at Ja=0).
    Tc_guess = 3.64

    t_start = time.time()
    for i, Ja in enumerate(tqdm(Ja_vals, total=len(Ja_vals), desc="Ja sweep")):
        # Coarse scan: bracket around last Tc_guess if available; otherwise full range.
        if i == 0:
            T0, T1 = args.Tmin, args.Tmax
        else:
            T0 = max(args.Tmin, Tc_guess - 1.5)
            T1 = min(args.Tmax, Tc_guess + 1.5)

        Ts_coarse = frange(T0, T1, args.coarse_step)
        scan0: Dict[float, Dict[str, float]] = run_mc_scan(args.J, Ja, args.n, phis, Ts_coarse, cfg)
        Tc0, _, _ = estimate_Tc_from_susceptibility(scan0)

        # Refine around Tc0
        r0 = max(args.Tmin, Tc0 - args.refine_window)
        r1 = min(args.Tmax, Tc0 + args.refine_window)
        Ts_ref = frange(r0, r1, args.refine_step)
        scan1: Dict[float, Dict[str, float]] = run_mc_scan(args.J, Ja, args.n, phis, Ts_ref, cfg)
        Tc1, Tc1_lo, Tc1_hi = estimate_Tc_from_susceptibility(scan1)

        if not math.isfinite(Tc1):
            skipped += 1
            continue

        # Conservative floor from temperature grid resolution: Tc is inferred from discrete Ts,
        # so there is an irreducible uncertainty of ~ΔT/2 (even with infinite statistics).
        dt_floor = 0.5 * float(args.refine_step)
        Tc1_lo = min(Tc1_lo, Tc1 - dt_floor)
        Tc1_hi = max(Tc1_hi, Tc1 + dt_floor)
        Tc1_lo = max(args.Tmin, Tc1_lo)
        Tc1_hi = min(args.Tmax, Tc1_hi)

        xs.append(Ja)
        ys.append(Tc1)
        ylos.append(Tc1_lo)
        yhis.append(Tc1_hi)
        Tc_guess = Tc1

        elapsed = time.time() - t_start
        # lightweight progress
        print(f"[{i+1:>3}/{len(Ja_vals)}] Ja={Ja:.3f}  Tc≈{Tc1:.4f}   elapsed={elapsed/60:.1f} min")

    # Save data + metadata for later overlay plotting
    base, _ext = os.path.splitext(args.out)
    data_out = args.data_out.strip() or (base + ".csv")
    meta_out = args.meta_out.strip() or (base + ".json")

    with open(data_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Ja", "Tc_mc", "Tc_mc_lo", "Tc_mc_hi"])
        for Ja, Tc, lo, hi in zip(xs, ys, ylos, yhis):
            w.writerow([f"{Ja:.10g}", f"{Tc:.10g}", f"{lo:.10g}", f"{hi:.10g}"])

    meta = {
        "model": {"J": args.J, "n": args.n, "phis": list(phis)},
        "Ja_grid": {"min": args.Ja_min, "max": args.Ja_max, "num": args.num},
        "T_scan": {
            "Tmin": args.Tmin,
            "Tmax": args.Tmax,
            "coarse_step": args.coarse_step,
            "refine_window": args.refine_window,
            "refine_step": args.refine_step,
        },
        "mc": {
            "L": args.L,
            "therm": args.therm,
            "meas": args.meas,
            "thin": args.thin,
            "replicas": args.replicas,
            "seed": args.seed,
            "nprocs": args.nprocs,
            "update": args.update,
        },
        "results": {"n_points": len(xs), "skipped": skipped},
    }
    with open(meta_out, "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)

    print(f"Saved: {data_out}")
    print(f"Saved: {meta_out}")

    # Plot (optional dependency)
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ModuleNotFoundError:
        print("matplotlib not installed; skipping PNG plot. Install with: ./env/bin/pip install matplotlib")
        return

    # Smooth analytic overlay curve across the full Ja axis range.
    # This does not affect the MC estimation; it's just for plotting/comparison.
    ax: List[float] = []
    ay: List[float] = []
    try:
        from angle_modulated_triangular_ising_closedform_vs_mc import solve_Tc

        if args.analytic_num < 2:
            Ja_dense = [args.Ja_min]
        else:
            dJa = (args.Ja_max - args.Ja_min) / (args.analytic_num - 1)
            Ja_dense = [args.Ja_min + i * dJa for i in range(args.analytic_num)]
        for Ja in Ja_dense:
            try:
                Tc = solve_Tc(args.J, Ja, args.n, phis, T_lo=1e-8, T_hi=max(10.0, args.Tc_max * 2.0))
            except Exception:
                continue
            if math.isfinite(Tc):
                ax.append(Ja)
                ay.append(Tc)
    except Exception:
        # If analytic import/solve fails for any reason, we still produce MC plot.
        ax, ay = [], []

    plt.figure(figsize=(7.0, 4.5), dpi=160)
    # Phase region coloring using the analytic curve boundary, if available.
    if ax:
        import numpy as np

        x = np.asarray(ax, dtype=float) / max(args.J, 1e-300)
        y = np.asarray(ay, dtype=float)
        y = np.clip(y, 0.0, float(args.Tc_max))

        Ja_star = 2.0  # i.e. Ja = 2J
        fe_color = "#f26d6d"  # saturated red
        pm_color = "#9b7bd4"  # saturated purple
        st_color = "#6fb1f2"  # saturated blue

        axh = plt.gca()
        axh.fill_between(x, 0.0, y, where=(x <= Ja_star), color=fe_color, alpha=0.48, linewidth=0, zorder=0)
        axh.fill_between(x, 0.0, y, where=(x >= Ja_star), color=st_color, alpha=0.48, linewidth=0, zorder=0)
        axh.fill_between(x, y, float(args.Tc_max), color=pm_color, alpha=0.38, linewidth=0, zorder=0)

    # MC points with error bars if available
    yerr_low = [max(0.0, y - lo) if math.isfinite(lo) else 0.0 for y, lo in zip(ys, ylos)]
    yerr_hi = [max(0.0, hi - y) if math.isfinite(hi) else 0.0 for y, hi in zip(ys, yhis)]
    xs_plot = [x / max(args.J, 1e-300) for x in xs]
    plt.errorbar(
        xs_plot,
        ys,
        yerr=[yerr_low, yerr_hi],
        fmt=".",
        markersize=3.0,
        capsize=2.0,
        label="MC (\u03c7 peak, L=36)",
    )
    if ax:
        # Avoid drawing parts of the curve stuck at the top axis edge.
        ax_plot = []
        ay_plot = []
        for x0, y0 in zip(ax, ay):
            if y0 < (args.Tc_max - 0.05):
                ax_plot.append(x0 / max(args.J, 1e-300))
                ay_plot.append(y0)
        plt.plot(ax_plot, ay_plot, "-", linewidth=1.5, label="Critical manifold curve")
    plt.xlim(args.Ja_min / max(args.J, 1e-300), args.Ja_max / max(args.J, 1e-300))
    plt.ylim(0.0, args.Tc_max)
    plt.xlabel(r"$J_a/J$")
    plt.ylabel(r"$T_c$")
    plt.title(r"Angle-modulated triangular Ising phase diagram: exact manifold + MC")
    # Phase labels (common shorthand): FE / PM / ST.
    axh = plt.gca()
    label_kw = dict(fontsize=12, fontweight="bold", alpha=0.85, ha="center", va="center")
    axh.text(0.22, 0.30, "FE", transform=axh.transAxes, **label_kw)
    axh.text(0.52, 0.78, "PM", transform=axh.transAxes, **label_kw)
    axh.text(0.82, 0.50, "ST", transform=axh.transAxes, **label_kw)
    if skipped:
        plt.text(
            0.02,
            0.02,
            f"Skipped {skipped} points",
            transform=plt.gca().transAxes,
            fontsize=8,
            alpha=0.8,
        )
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best", fontsize=9, frameon=False)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    plt.close()

    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()

