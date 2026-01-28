#!/usr/bin/env python3
"""
Plot an analytic phase diagram Tc(Ja) for the angle-modulated triangular Ising model.

This uses ONLY the closed-form Tc solver (no Monte Carlo).
Outputs a PNG dot plot with axes:
  Ja in [0, Ja_max]
  Tc in [0, Tc_max]
"""

from __future__ import annotations

import argparse
import math
import os
import multiprocessing as mp
from typing import List, Tuple

try:
    from tqdm import tqdm  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    def tqdm(x, **_kwargs):  # type: ignore
        return x


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--J", type=float, default=1.0, help="Base coupling J.")
    p.add_argument("--n", type=int, default=2, help="Modulation integer n.")
    p.add_argument(
        "--phis",
        type=str,
        default="default",
        help="Bond angles: 'default' or comma-separated radians (3 values).",
    )
    # Parse as strings so --mp-dps mode can preserve ultra-fine ranges near Ja=2.
    p.add_argument("--Ja-min", type=str, default="0.0", help="Minimum Ja.")
    p.add_argument("--Ja-max", type=str, default="4.5", help="Maximum Ja.")
    p.add_argument("--num", type=int, default=901, help="Number of Ja points.")
    p.add_argument("--Tc-min", type=float, default=0.0, help="Y-axis min for Tc.")
    p.add_argument("--Tc-max", type=float, default=6.0, help="Y-axis max for Tc.")
    p.add_argument(
        "--x-offset",
        type=str,
        default="0.0",
        help="Plot transform: x_plot = (Ja - x_offset) * x_scale. (String for mp-dps precision.)",
    )
    p.add_argument(
        "--x-scale",
        type=float,
        default=1.0,
        help="Plot transform: x_plot = (Ja - x_offset) * x_scale.",
    )
    p.add_argument(
        "--x-label",
        type=str,
        default=r"$J_a/J$",
        help="X-axis label (after any transform).",
    )
    p.add_argument(
        "--plain-x",
        action="store_true",
        help="Disable scientific notation/offset on x-axis tick labels.",
    )
    p.add_argument(
        "--center-line-to",
        type=float,
        default=0.0,
        help="If >0, draw a vertical guide at x=0 from Tc=0 up to this value.",
    )
    p.add_argument(
        "--shade-phases",
        action="store_true",
        help="Shade FE/PM/ST regions using Tc(Ja) as boundary (split at x=0).",
    )
    p.add_argument("--maxiter", type=int, default=80, help="Bisection iterations per Ja point (plot accuracy).")
    p.add_argument("--rtol", type=float, default=1e-9, help="Relative tolerance for Tc (plot accuracy).")
    p.add_argument(
        "--mp-dps",
        type=int,
        default=0,
        help="If >0, use mpmath with this many decimal digits of precision for the analytic solve.",
    )
    p.add_argument(
        "--nprocs",
        type=int,
        default=0,
        help="Worker processes for analytic sweep (0=all cores, 1=single-process).",
    )
    p.add_argument(
        "--out",
        type=str,
        default="phase_diagram_Tc_vs_Ja_analytic.png",
        help="Output PNG path.",
    )
    return p.parse_args()


def parse_phis(spec: str) -> Tuple[float, float, float]:
    if spec.strip().lower() == "default":
        # Import from the main script for consistency.
        from angle_modulated_triangular_ising_closedform_vs_mc import default_phis

        return default_phis()
    parts = [s.strip() for s in spec.split(",") if s.strip()]
    if len(parts) != 3:
        raise ValueError("phis must be 'default' or 3 comma-separated radians, e.g. 0,1.0472,2.0944")
    return (float(parts[0]), float(parts[1]), float(parts[2]))

def _as_float(x: str) -> float:
    return float(x.strip())


def _fast_solve_Tc_nohint(args: Tuple[float, float, int, Tuple[float, float, float], int, float, float, float]) -> Tuple[float, float, bool]:
    """
    Worker: solve Tc for a single Ja without using any cross-point hint.
    Returns (Ja, Tc, ok).
    """
    Ja, J, n, phis, maxiter, rtol, Tc_max, T_hi0 = args
    from angle_modulated_triangular_ising_closedform_vs_mc import critical_equation_T, effective_Jks

    # 1D-like decoupled limit: no finite-T transition
    Jks = effective_Jks(J, Ja, n, phis)
    n_small = sum(1 for x in Jks if abs(x) < 1e-15)
    if n_small >= 2:
        return Ja, 0.0, True

    def fT(T: float) -> float:
        return critical_equation_T(T, J, Ja, n, phis)

    T_lo = 1e-12
    T_hi = max(T_hi0, 10.0, Tc_max * 2.0)
    f_lo = fT(T_lo)
    f_hi = fT(T_hi)
    if not (math.isfinite(f_lo) and math.isfinite(f_hi)):
        return Ja, float("nan"), False

    # Expand upward to find sign change; if none exists, Tc=0
    expands = 0
    while f_lo * f_hi > 0 and expands < 80:
        T_hi *= 1.5
        f_hi = fT(T_hi)
        expands += 1
        if not math.isfinite(f_hi):
            return Ja, float("nan"), False

    if f_lo == 0.0:
        return Ja, T_lo, True
    if f_hi == 0.0:
        return Ja, T_hi, True
    if f_lo * f_hi > 0:
        return Ja, 0.0, True

    for _ in range(max(1, int(maxiter))):
        T_mid = 0.5 * (T_lo + T_hi)
        f_mid = fT(T_mid)
        if f_mid == 0.0:
            return Ja, T_mid, True
        if f_lo * f_mid < 0.0:
            T_hi, f_hi = T_mid, f_mid
        else:
            T_lo, f_lo = T_mid, f_mid
        if abs(T_hi - T_lo) / max(1e-30, T_mid) < rtol:
            break
    return Ja, 0.5 * (T_lo + T_hi), True


def _mp_solve_Tc_nohint(
    args: Tuple[
        int,  # idx
        str,  # Ja_min
        str,  # dJa
        float,  # J
        int,  # n
        Tuple[float, float, float],  # phis
        int,  # maxiter
        float,  # rtol
        float,  # Tc_max
        float,  # T_hi0
        int,  # mp_dps
        str,  # x_offset
        float,  # x_scale
    ]
) -> Tuple[float, float, bool]:
    """
    Worker: mpmath solve Tc for a single Ja without hints.
    Returns (Ja, Tc_as_float, ok).
    """
    idx, Ja_min_s, dJa_s, J_f, n, phis_f, maxiter, rtol_f, Tc_max_f, T_hi0_f, mp_dps, x_offset_s, x_scale_f = args
    try:
        import mpmath as mp  # type: ignore
    except ModuleNotFoundError:
        return float("nan"), float("nan"), False

    mp.mp.dps = int(mp_dps)
    Ja = mp.mpf(Ja_min_s) + mp.mpf(idx) * mp.mpf(dJa_s)
    J = mp.mpf(J_f)
    rtol = mp.mpf(rtol_f)
    Tc_max = mp.mpf(Tc_max_f)
    # If angles look like the default triangular directions, switch to exact mpmath pi values.
    # This avoids tiny sign-asymmetries near Ja≈2 caused by float rounding of phis.
    if (
        abs(float(phis_f[0]) - 0.0) < 1e-12
        and abs(float(phis_f[1]) - (2.0 * math.pi / 3.0)) < 1e-9
        and abs(float(phis_f[2]) - (4.0 * math.pi / 3.0)) < 1e-9
    ):
        phis = (mp.mpf("0"), 2 * mp.pi / 3, 4 * mp.pi / 3)
    else:
        phis = tuple(mp.mpf(p) for p in phis_f)

    # Treat ultra-small couplings as exactly zero in sign logic.
    zero_tol = mp.mpf("1e-40")

    def effective_Jks_mp() -> List[mp.mpf]:
        Jks0 = [J + Ja * mp.cos(n * phi) for phi in phis]
        # zero-out tiny values before sign checks
        Jks0 = [mp.mpf("0") if mp.fabs(x) < zero_tol else x for x in Jks0]
        nneg = sum(1 for x in Jks0 if x < 0)
        if nneg == 2:
            return [mp.fabs(x) if x != 0 else mp.mpf("0") for x in Jks0]
        return Jks0

    Jks = effective_Jks_mp()
    n_small = sum(1 for x in Jks if mp.fabs(x) < mp.mpf("1e-50"))
    # x-axis is dimensionless by default: x := Ja/J
    x_plot = float(((Ja / J) - mp.mpf(x_offset_s)) * mp.mpf(x_scale_f))
    if n_small >= 2:
        return x_plot, 0.0, True

    # reject frustrated patterns (1 or 3 negatives) for this closed-form equation
    nneg = sum(1 for x in Jks if x < 0)
    if nneg not in (0, 2):
        return x_plot, float("nan"), False

    def fT(T: mp.mpf) -> mp.mpf:
        if T <= 0:
            return mp.mpf("1e9")
        Ks = [Jk / T for Jk in Jks]
        term = mp.exp(-2 * (Ks[0] + Ks[1])) + mp.exp(-2 * (Ks[1] + Ks[2])) + mp.exp(-2 * (Ks[2] + Ks[0]))
        return term - 1

    T_lo = mp.mpf("1e-50")
    T_hi = mp.mpf(max(T_hi0_f, 10.0, float(Tc_max) * 2.0))
    f_lo = fT(T_lo)
    f_hi = fT(T_hi)

    expands = 0
    while f_lo * f_hi > 0 and expands < 120:
        T_hi *= mp.mpf("1.5")
        f_hi = fT(T_hi)
        expands += 1

    if f_lo == 0:
        return x_plot, float(T_lo), True
    if f_hi == 0:
        return x_plot, float(T_hi), True
    if f_lo * f_hi > 0:
        return x_plot, 0.0, True

    for _ in range(max(1, int(maxiter))):
        T_mid = (T_lo + T_hi) / 2
        f_mid = fT(T_mid)
        if f_mid == 0:
            return x_plot, float(T_mid), True
        if f_lo * f_mid < 0:
            T_hi, f_hi = T_mid, f_mid
        else:
            T_lo, f_lo = T_mid, f_mid
        if mp.fabs(T_hi - T_lo) / (T_mid if T_mid != 0 else mp.mpf("1")) < rtol:
            break
    return x_plot, float((T_lo + T_hi) / 2), True


def main() -> None:
    args = parse_args()
    phis = parse_phis(args.phis)

    # Import analytic equation (analytic only; MC is not used).
    from angle_modulated_triangular_ising_closedform_vs_mc import critical_equation_T, effective_Jks

    # If the user requested arbitrary precision, ensure the dependency exists up front.
    if args.mp_dps and args.mp_dps > 0:
        try:
            import mpmath as _mp  # type: ignore
        except ModuleNotFoundError as e:
            raise SystemExit(
                "You passed --mp-dps but 'mpmath' is not installed in this environment.\n"
                "Install it with:\n"
                "  ./env/bin/pip install -U mpmath\n"
            ) from e

    # Build Ja grid.
    # - float mode: build explicit float list
    # - mpmath mode: generate Ja in worker to avoid float rounding for ultra-fine ranges
    Ja_vals: List[float] = []
    if not (args.mp_dps and args.mp_dps > 0):
        if args.num < 2:
            Ja_vals = [_as_float(args.Ja_min)]
        else:
            step = (_as_float(args.Ja_max) - _as_float(args.Ja_min)) / (args.num - 1)
            Ja_vals = [_as_float(args.Ja_min) + i * step for i in range(args.num)]

    def fT(T: float, Ja: float) -> float:
        return critical_equation_T(T, args.J, Ja, args.n, phis)

    def fast_solve_Tc(Ja: float, Tc_hint: float | None) -> float:
        # Special case: if the model is effectively 1D (two couplings ~0), Tc=0.
        Jks = effective_Jks(args.J, Ja, args.n, phis)
        n_small = sum(1 for x in Jks if abs(x) < 1e-15)
        if n_small >= 2:
            return 0.0

        # Bracket around previous Tc if available.
        if Tc_hint is None or not math.isfinite(Tc_hint) or Tc_hint <= 0:
            T_lo, T_hi = 1e-12, max(10.0, args.Tc_max * 2.0)
        else:
            T_lo = max(1e-12, Tc_hint * 0.2)
            T_hi = max(T_lo * 1.5, Tc_hint * 5.0)

        f_lo = fT(T_lo, Ja)
        f_hi = fT(T_hi, Ja)
        if not (math.isfinite(f_lo) and math.isfinite(f_hi)):
            raise RuntimeError("Not applicable (likely frustrated sign pattern).")

        # Expand bracket if needed
        expands = 0
        while f_lo * f_hi > 0 and expands < 60:
            # Prefer expanding upward (Tc tends to move smoothly in Ja)
            T_hi *= 1.5
            f_hi = fT(T_hi, Ja)
            expands += 1

        if f_lo == 0.0:
            return T_lo
        if f_hi == 0.0:
            return T_hi
        if f_lo * f_hi > 0:
            # No finite-T root (e.g. 1D limit): treat as Tc=0
            return 0.0

        # Bisection (loose tolerances are fine for plotting, especially for 100k points)
        for _ in range(max(1, int(args.maxiter))):
            T_mid = 0.5 * (T_lo + T_hi)
            f_mid = fT(T_mid, Ja)
            if f_mid == 0.0:
                return T_mid
            if f_lo * f_mid < 0.0:
                T_hi, f_hi = T_mid, f_mid
            else:
                T_lo, f_lo = T_mid, f_mid
            if abs(T_hi - T_lo) / max(1e-30, T_mid) < args.rtol:
                break
        return 0.5 * (T_lo + T_hi)

    xs: List[float] = []
    ys: List[float] = []
    failures = 0
    nprocs = int(args.nprocs)
    if nprocs == 0:
        nprocs = os.cpu_count() or 1

    # If mp-dps is enabled, prefer multiprocessing to avoid Python-level overhead.
    if args.mp_dps and args.mp_dps > 0:
        import mpmath as mpm  # type: ignore

        mpm.mp.dps = max(50, int(args.mp_dps))
        if args.num < 2:
            dJa_s = "0"
        else:
            dJa_s = mpm.nstr(
                (mpm.mpf(args.Ja_max) - mpm.mpf(args.Ja_min)) / mpm.mpf(args.num - 1),
                n=int(args.mp_dps),
            )

        work = [
            (
                i,
                args.Ja_min,
                dJa_s,
                float(args.J),
                args.n,
                phis,
                int(args.maxiter),
                float(args.rtol),
                float(args.Tc_max),
                max(10.0, args.Tc_max * 2.0),
                int(args.mp_dps),
                args.x_offset,
                float(args.x_scale),
            )
            for i in range(max(1, int(args.num)))
        ]
        if nprocs <= 1:
            for w in tqdm(work, total=len(work), desc="Analytic sweep (mpmath)"):
                x_plot, Tc, ok = _mp_solve_Tc_nohint(w)
                if not ok or not math.isfinite(Tc):
                    failures += 1
                    continue
                xs.append(x_plot)
                ys.append(Tc)
        else:
            chunksize = max(50, len(work) // (nprocs * 8) if nprocs > 0 else 200)
            with mp.Pool(processes=nprocs) as pool:
                for x_plot, Tc, ok in tqdm(
                    pool.imap(_mp_solve_Tc_nohint, work, chunksize=chunksize),
                    total=len(work),
                    desc="Analytic sweep (mpmath)",
                ):
                    if not ok or not math.isfinite(Tc):
                        failures += 1
                        continue
                    xs.append(x_plot)
                    ys.append(Tc)

    # Single-process float path uses a rolling hint (usually faster for smooth curves).
    elif nprocs <= 1:
        Tc_hint: float | None = None
        for Ja in tqdm(Ja_vals, total=len(Ja_vals), desc="Analytic sweep"):
            try:
                Tc = fast_solve_Tc(Ja, Tc_hint)
            except Exception:
                failures += 1
                continue
            xs.append(Ja)
            ys.append(Tc)
            Tc_hint = Tc
    else:
        # Multi-process path: solve each Ja independently (better CPU utilization for huge grids).
        work = [(Ja, args.J, args.n, phis, int(args.maxiter), float(args.rtol), float(args.Tc_max), max(10.0, args.Tc_max * 2.0)) for Ja in Ja_vals]
        # Heuristic chunksize: enough work per task to amortize IPC overhead.
        chunksize = max(50, len(work) // (nprocs * 8) if nprocs > 0 else 200)
        with mp.Pool(processes=nprocs) as pool:
            for Ja, Tc, ok in tqdm(
                pool.imap(_fast_solve_Tc_nohint, work, chunksize=chunksize),
                total=len(work),
                desc="Analytic sweep",
            ):
                if not ok or not math.isfinite(Tc):
                    failures += 1
                    continue
                xs.append(Ja)
                ys.append(Tc)

    # Plot
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter  # type: ignore
    import numpy as np

    # Apply optional x-axis transform for display.
    # In mpmath mode we already stored x_plot directly to preserve ultra-fine ranges.
    x_offset = _as_float(args.x_offset)
    x_scale = float(args.x_scale)
    if args.mp_dps and args.mp_dps > 0:
        xs_plot = xs
    else:
        # x-axis is dimensionless by default: x := Ja/J
        xs_plot = [((x / max(args.J, 1e-300)) - x_offset) * x_scale for x in xs]

    plt.figure(figsize=(7.0, 4.5), dpi=160)

    # Optional phase-region shading (FE/PM/ST) using Tc curve as boundary.
    if args.shade_phases and xs_plot:
        x = np.asarray(xs_plot, dtype=float)
        y = np.asarray(ys, dtype=float)
        y = np.clip(y, float(args.Tc_min), float(args.Tc_max))

        fe_color = "#f26d6d"  # saturated red
        pm_color = "#9b7bd4"  # saturated purple
        st_color = "#6fb1f2"  # saturated blue

        axh = plt.gca()
        axh.fill_between(x, float(args.Tc_min), y, where=(x <= 0.0), color=fe_color, alpha=0.48, linewidth=0, zorder=0)
        axh.fill_between(x, float(args.Tc_min), y, where=(x >= 0.0), color=st_color, alpha=0.48, linewidth=0, zorder=0)
        axh.fill_between(x, y, float(args.Tc_max), color=pm_color, alpha=0.38, linewidth=0, zorder=0)

        label_kw = dict(fontsize=12, fontweight="bold", alpha=0.9, ha="center", va="center")
        axh.text(0.22, 0.30, "FE", transform=axh.transAxes, **label_kw)
        axh.text(0.52, 0.78, "PM", transform=axh.transAxes, **label_kw)
        axh.text(0.82, 0.30, "ST", transform=axh.transAxes, **label_kw)

    # For large num (e.g. 100k), a line is typically faster and looks cleaner.
    if args.num >= 5000:
        plt.plot(xs_plot, ys, "-", linewidth=1.0)
    else:
        plt.plot(xs_plot, ys, ".", markersize=2.0)

    # X limits in transformed coordinates; use mpmath for ultra-fine ranges if requested.
    if args.mp_dps and args.mp_dps > 0:
        import mpmath as mpm  # type: ignore

        mpm.mp.dps = max(50, int(args.mp_dps))
        Jm = mpm.mpf(str(args.J))
        xl = float(((mpm.mpf(args.Ja_min) / Jm) - mpm.mpf(args.x_offset)) * mpm.mpf(x_scale))
        xh = float(((mpm.mpf(args.Ja_max) / Jm) - mpm.mpf(args.x_offset)) * mpm.mpf(x_scale))
    else:
        xl = ((_as_float(args.Ja_min) / max(args.J, 1e-300)) - x_offset) * x_scale
        xh = ((_as_float(args.Ja_max) / max(args.J, 1e-300)) - x_offset) * x_scale
    plt.xlim(xl, xh)
    plt.ylim(args.Tc_min, args.Tc_max)
    plt.xlabel(args.x_label)
    plt.ylabel(r"$T_c$")
    plt.title(r"Analytic $T_c(J_a/J)$")
    if args.center_line_to and args.center_line_to > 0:
        plt.plot([0.0, 0.0], [0.0, float(args.center_line_to)], color="C0", linewidth=1.0, alpha=0.9)
    if args.plain_x:
        ax = plt.gca()
        ax.ticklabel_format(axis="x", style="plain", useOffset=False)
        fmt = ScalarFormatter(useOffset=False)
        fmt.set_scientific(False)
        ax.xaxis.set_major_formatter(fmt)
    if failures:
        plt.text(
            0.02,
            0.02,
            f"Skipped {failures} points (no bracket / not applicable)",
            transform=plt.gca().transAxes,
            fontsize=8,
            alpha=0.8,
        )
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    plt.close()

    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()

