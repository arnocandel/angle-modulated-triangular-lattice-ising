#!/usr/bin/env python3
"""
Closed-form critical line for an angle-modulated triangular-lattice Ising model
and a Monte Carlo check with error bars.

Model:
    H = - sum_{<i,j>} [ J + J_a cos(n * phi_{ij}) ] * s_i s_j
where s_i = ±1.

If the bond angles take only the 3 triangular directions (k=1,2,3) with fixed
phi_k, the model reduces to an anisotropic triangular Ising model with 3
couplings:
    J_k = J + J_a cos(n * phi_k)
    K_k = beta * J_k = (J_k / T)

Exact (rigorous) critical surface for ferromagnetic triangular Ising (one transition):
    p1 p2 + p2 p3 + p3 p1 = 1,    where p_k = exp(-2 K_k)
Equivalently:
    exp(-2(K1+K2)) + exp(-2(K2+K3)) + exp(-2(K3+K1)) = 1

This script:
  1) Computes T_c(J, J_a, n, phi_k) by solving the exact equation.
  2) Runs Metropolis MC on an LxL rhombus of the triangular lattice (periodic),
     estimates pseudo-critical temperature via susceptibility peak, and reports
     uncertainty via bootstrap.
  3) Compares closed-form T_c to MC estimate, and prints timing.

No external data required.
"""

from __future__ import annotations
import math, time, random, statistics
from dataclasses import dataclass
from typing import List, Tuple, Dict, Callable, Optional

import os
import multiprocessing as mp
import numpy as np

def effective_Jks(J: float, Ja: float, n: int, phis: Tuple[float, float, float]) -> List[float]:
    """
    Return the 3 directional couplings J_k = J + Ja*cos(n*phi_k).

    Convention:
    - If exactly two J_k are negative (stripe sign pattern, unfrustrated), return |J_k|
      so cluster algorithms can be used; Tc is invariant under this gauge map.
    - Otherwise return the raw J_k (may include frustrated sign patterns).
    """
    Jks = [J + Ja * math.cos(n * phi) for phi in phis]
    nneg = sum(1 for x in Jks if x < 0.0)
    if nneg == 2:
        return [abs(x) for x in Jks]
    return Jks

# ----------------------------
# Exact Tc solver
# ----------------------------

def critical_equation_T(T: float, J: float, Ja: float, n: int, phis: Tuple[float,float,float]) -> float:
    """
    Return f(T) where f(Tc)=0 for the triangular critical surface.

    Notes on sign patterns:
    - If all three couplings J_k are ferromagnetic (J_k > 0), the usual exact
      triangular-lattice Ising critical manifold applies.
    - If exactly two of the J_k are antiferromagnetic (J_k < 0), the uniform
      bond-sign pattern is *gaugeable* on the triangular rhombus (product of
      signs around each triangle is +1). A sublattice spin flip maps the model
      to ferromagnetic couplings |J_k|, and the same critical manifold applies
      to the transition into the corresponding striped order in the original
      variables.
    - If exactly one (or all three) couplings are negative, the model is
      frustrated and this closed-form Tc equation is not applicable.
    """
    if T <= 0:
        return 1e9
    Jks_raw = [J + Ja * math.cos(n * phi) for phi in phis]
    nneg = sum(1 for x in Jks_raw if x < 0.0)
    if nneg == 2:
        Jks = [abs(x) for x in Jks_raw]
    elif nneg != 0:
        # Frustrated sign pattern: do not pretend there is a Tc from this formula.
        return float("nan")
    else:
        Jks = Jks_raw
    Ks  = [Jk / T for Jk in Jks]  # beta=1/T (kB=1)
    # f(T) = sum exp(-2(Ki+Kj)) - 1
    term = math.exp(-2*(Ks[0]+Ks[1])) + math.exp(-2*(Ks[1]+Ks[2])) + math.exp(-2*(Ks[2]+Ks[0]))
    return term - 1.0

def solve_Tc(J: float, Ja: float, n: int, phis: Tuple[float,float,float], T_lo: float=1e-6, T_hi: float=50.0) -> float:
    """
    Solve for Tc using bisection on the exact critical equation.
    Assumes (at least) one root bracketed by a sign change f(T_lo)*f(T_hi)<0.
    (For the ferromagnetic triangular Ising critical line used here, typically
    f(T)->-1 as T->0+ and f(T)->+2 as T->+inf, so Tc is usually bracketable.)
    """
    def f(T: float) -> float:
        return critical_equation_T(T, J, Ja, n, phis)

    f_lo = f(T_lo)
    f_hi = f(T_hi)
    if not (math.isfinite(f_lo) and math.isfinite(f_hi)):
        raise RuntimeError(
            "Closed-form Tc equation not applicable for this parameter set (likely frustrated bond signs)."
        )

    # If not bracketed, do an adaptive geometric scan while expanding T_hi.
    # This avoids assuming which side is positive/negative.
    def geomspace(a: float, b: float, npts: int) -> List[float]:
        if npts < 2:
            return [a]
        if a <= 0 or b <= 0:
            raise ValueError("geomspace requires positive endpoints")
        r = (b / a) ** (1.0 / (npts - 1))
        out = [a]
        x = a
        for _ in range(npts - 2):
            x *= r
            out.append(x)
        out.append(b)
        return out

    if f_lo == 0.0:
        return T_lo
    if f_hi == 0.0:
        return T_hi

    bracketed = (f_lo * f_hi) < 0.0
    if not bracketed:
        max_expands = 80
        scan_points = 200
        for _ in range(max_expands):
            Ts = geomspace(T_lo, T_hi, scan_points)
            fs = [f(t) for t in Ts]
            found = False
            for i in range(len(Ts) - 1):
                if fs[i] == 0.0:
                    return Ts[i]
                if fs[i] * fs[i + 1] < 0.0:
                    T_lo, f_lo = Ts[i], fs[i]
                    T_hi, f_hi = Ts[i + 1], fs[i + 1]
                    found = True
                    bracketed = True
                    break
            if found:
                break
            T_hi *= 1.5
            f_hi = f(T_hi)

    if not bracketed:
        raise RuntimeError(
            "Could not bracket Tc (no sign change found over the searched T range). "
            "This can happen if some effective couplings are antiferromagnetic (frustrated), "
            "if the chosen closed-form equation is not applicable for this parameter regime, "
            "or if multiple transitions/roots exist."
        )

    # bisection
    for _ in range(200):
        T_mid = 0.5*(T_lo+T_hi)
        f_mid = f(T_mid)
        if f_mid == 0.0:
            return T_mid
        # Keep the sub-interval that preserves the sign change.
        if f_lo * f_mid < 0.0:
            T_hi, f_hi = T_mid, f_mid
        else:
            T_lo, f_lo = T_mid, f_mid
        if abs(T_hi-T_lo) / max(1e-12, T_mid) < 1e-12:
            break
    return 0.5*(T_lo+T_hi)

# ----------------------------
# Triangular lattice MC
# ----------------------------

@dataclass(frozen=True)
class NeighborTable:
    """6-neighbor lookup tables for an LxL periodic triangular rhombus."""
    px: np.ndarray   # +x  (uses K0)
    mx: np.ndarray   # -x  (uses K0)
    py: np.ndarray   # +y  (uses K1)
    my: np.ndarray   # -y  (uses K1)
    pxy: np.ndarray  # +(x+y) (uses K2)
    mxy: np.ndarray  # -(x+y) (uses K2)


def build_neighbor_table(L: int) -> NeighborTable:
    """Precompute the 6 neighbor indices for each site."""
    N = L * L

    def idx(x: int, y: int) -> int:
        return (x % L) + L * (y % L)

    px = np.empty(N, dtype=np.int32)
    mx = np.empty(N, dtype=np.int32)
    py = np.empty(N, dtype=np.int32)
    my = np.empty(N, dtype=np.int32)
    pxy = np.empty(N, dtype=np.int32)
    mxy = np.empty(N, dtype=np.int32)

    for i in range(N):
        x = i % L
        y = i // L
        px[i] = idx(x + 1, y)
        mx[i] = idx(x - 1, y)
        py[i] = idx(x, y + 1)
        my[i] = idx(x, y - 1)
        pxy[i] = idx(x + 1, y + 1)
        mxy[i] = idx(x - 1, y - 1)

    return NeighborTable(px=px, mx=mx, py=py, my=my, pxy=pxy, mxy=mxy)


def metropolis_sweep(spins: np.ndarray, neigh: NeighborTable, Ks: Tuple[float, float, float], rng: random.Random) -> None:
    """
    One sweep = N attempted flips.
    Uses the 6 precomputed neighbor indices:
       ±x uses K0, ±y uses K1, ±(x+y) uses K2
    """
    N = spins.size
    for _ in range(N):
        i = rng.randrange(N)

        si = spins[i]
        # local field from neighbors with anisotropic couplings
        h = (
            Ks[0] * (spins[neigh.px[i]] + spins[neigh.mx[i]])
            + Ks[1] * (spins[neigh.py[i]] + spins[neigh.my[i]])
            + Ks[2] * (spins[neigh.pxy[i]] + spins[neigh.mxy[i]])
        )
        dE = 2.0 * si * h  # energy change for flip
        if dE <= 0.0 or rng.random() < math.exp(-dE):
            spins[i] = -si


def wolff_update(
    spins: np.ndarray,
    neigh: NeighborTable,
    Ks: Tuple[float, float, float],
    rng: random.Random,
    in_cluster: np.ndarray,
    stack: List[int],
    cluster_sites: List[int],
) -> int:
    """
    One Wolff cluster flip for ferromagnetic couplings (Ks >= 0).
    Returns cluster size.
    """
    if Ks[0] < 0 or Ks[1] < 0 or Ks[2] < 0:
        raise RuntimeError("Wolff update requires ferromagnetic couplings (all Ks >= 0).")

    # bond activation probabilities per direction
    p0 = 1.0 - math.exp(-2.0 * Ks[0])
    p1 = 1.0 - math.exp(-2.0 * Ks[1])
    p2 = 1.0 - math.exp(-2.0 * Ks[2])

    N = spins.size
    i0 = rng.randrange(N)
    s0 = spins[i0]

    stack.clear()
    cluster_sites.clear()
    stack.append(i0)
    cluster_sites.append(i0)
    in_cluster[i0] = True

    while stack:
        i = stack.pop()

        # ±x
        j = int(neigh.px[i])
        if (not in_cluster[j]) and spins[j] == s0 and rng.random() < p0:
            in_cluster[j] = True
            stack.append(j)
            cluster_sites.append(j)
        j = int(neigh.mx[i])
        if (not in_cluster[j]) and spins[j] == s0 and rng.random() < p0:
            in_cluster[j] = True
            stack.append(j)
            cluster_sites.append(j)

        # ±y
        j = int(neigh.py[i])
        if (not in_cluster[j]) and spins[j] == s0 and rng.random() < p1:
            in_cluster[j] = True
            stack.append(j)
            cluster_sites.append(j)
        j = int(neigh.my[i])
        if (not in_cluster[j]) and spins[j] == s0 and rng.random() < p1:
            in_cluster[j] = True
            stack.append(j)
            cluster_sites.append(j)

        # ±(x+y)
        j = int(neigh.pxy[i])
        if (not in_cluster[j]) and spins[j] == s0 and rng.random() < p2:
            in_cluster[j] = True
            stack.append(j)
            cluster_sites.append(j)
        j = int(neigh.mxy[i])
        if (not in_cluster[j]) and spins[j] == s0 and rng.random() < p2:
            in_cluster[j] = True
            stack.append(j)
            cluster_sites.append(j)

    # flip and clear marks
    for i in cluster_sites:
        spins[i] = -spins[i]
    for i in cluster_sites:
        in_cluster[i] = False

    return len(cluster_sites)

def measure(spins: np.ndarray, L:int) -> Tuple[float,float]:
    """Return (m, m2) magnetization and magnetization^2 per site."""
    m = float(spins.sum()) / (L*L)
    return m, m*m

def susceptibility(m_samples: List[float], T: float, N: int) -> float:
    """Chi = N/T * ( <m^2> - <|m|>^2 ). Use |m| to reduce sign flips."""
    m_abs = [abs(m) for m in m_samples]
    m2 = [m*m for m in m_samples]
    return (N / T) * (statistics.mean(m2) - statistics.mean(m_abs)**2)

def bootstrap_ci(values: List[float], nboot: int=500, alpha: float=0.05, rng: Optional[random.Random]=None) -> Tuple[float,float,float]:
    """Return (mean, lo, hi) bootstrap percentile CI."""
    if rng is None:
        rng = random.Random(123)
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    means = []
    for _ in range(nboot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(statistics.mean(sample))
    means.sort()
    lo = means[int((alpha/2)*nboot)]
    hi = means[int((1-alpha/2)*nboot)-1]
    return statistics.mean(values), lo, hi


def hierarchical_bootstrap_ci(
    values_by_replica: List[List[float]],
    nboot: int = 500,
    alpha: float = 0.05,
    rng: Optional[random.Random] = None,
) -> Tuple[float, float, float]:
    """
    Hierarchical bootstrap for correlated time-series split into independent replicas.

    We resample replicas with replacement, and within each selected replica we
    resample its block-level estimates with replacement. This is typically more
    conservative than pooling all blocks, and helps avoid underestimating errors
    when block size is imperfect.
    """
    if rng is None:
        rng = random.Random(123)

    reps = [v for v in values_by_replica if len(v) > 0]
    if not reps:
        return float("nan"), float("nan"), float("nan")

    # Point estimate: mean over replica means (equal weight per replica).
    rep_means = [statistics.mean(v) for v in reps]
    point = statistics.mean(rep_means)

    means = []
    R = len(reps)
    for _ in range(nboot):
        chosen = [reps[rng.randrange(R)] for _ in range(R)]
        flat = []
        for blocks in chosen:
            m = len(blocks)
            flat.extend(blocks[rng.randrange(m)] for _ in range(m))
        means.append(statistics.mean(flat))
    means.sort()
    lo = means[int((alpha / 2) * nboot)]
    hi = means[int((1 - alpha / 2) * nboot) - 1]
    return point, lo, hi

@dataclass
class MCConfig:
    L: int = 36
    therm_sweeps: int = 2000
    meas_sweeps: int = 4000
    thin: int = 5
    seed: int = 0
    # Update algorithm: "metropolis" or "wolff" (cluster).
    update_method: str = "wolff"
    # Independent replicas per temperature (improves error bars; scales runtime ~linearly).
    replicas: int = 8
    # Number of worker processes for the MC temperature scan.
    # - 0: use all available CPU cores
    # - 1: run single-process (keeps "annealing" across temperatures by reusing spins)
    nprocs: int = 0
    # If True and running single-process, reuse spins between temperatures (annealing).
    # Only used when replicas==1 and update_method=="metropolis".
    anneal: bool = True


def _stable_temp_seed(base_seed: int, T: float) -> int:
    """Deterministic per-temperature seed (stable across runs / processes)."""
    # Use microkelvin-ish integerization so 2.2958 and 2.2958000 map identically.
    t_int = int(round(T * 1_000_000))
    # Mix bits (LCG-ish) into 32-bit space for Random().
    return (base_seed * 1_000_003 + t_int * 91_382_323) & 0xFFFFFFFF


def _mc_single_temperature(args: Tuple[float, float, float, int, Tuple[float, float, float], MCConfig]) -> Tuple[float, Dict[str, float]]:
    """Worker: run MC at a single temperature T and return its measurements."""
    T, J, Ja, n, phis, cfg = args
    L = cfg.L
    N = L * L
    neigh = build_neighbor_table(L)

    Jks = effective_Jks(J, Ja, n, phis)
    Ks = tuple(Jk / T for Jk in Jks)

    block = 10  # block size in (thinned) measurements
    chi_vals_by_rep: List[List[float]] = []
    u4_vals_by_rep: List[List[float]] = []

    nrep = max(1, int(cfg.replicas))
    for r in range(nrep):
        rng = random.Random((_stable_temp_seed(cfg.seed, T) + 1_664_525 * r) & 0xFFFFFFFF)

        # Independent hot start (parallel-friendly).
        spins = np.array([1 if rng.random() < 0.5 else -1 for _ in range(N)], dtype=np.int8)

        # thermalize
        if cfg.update_method == "wolff":
            in_cluster = np.zeros(N, dtype=np.bool_)
            stack: List[int] = []
            cluster_sites: List[int] = []
            for _ in range(cfg.therm_sweeps):
                wolff_update(spins, neigh, Ks, rng, in_cluster, stack, cluster_sites)
        else:
            for _ in range(cfg.therm_sweeps):
                metropolis_sweep(spins, neigh, Ks, rng)

        # measure (store thinned magnetization time series)
        m_samples: List[float] = []
        if cfg.update_method == "wolff":
            in_cluster = np.zeros(N, dtype=np.bool_)
            stack = []
            cluster_sites = []
            for sweep in range(cfg.meas_sweeps):
                wolff_update(spins, neigh, Ks, rng, in_cluster, stack, cluster_sites)
                if sweep % cfg.thin == 0:
                    m = float(spins.sum()) / N
                    m_samples.append(m)
        else:
            for sweep in range(cfg.meas_sweeps):
                metropolis_sweep(spins, neigh, Ks, rng)
                if sweep % cfg.thin == 0:
                    m = float(spins.sum()) / N
                    m_samples.append(m)

        # block estimates (reduces autocorrelation impact)
        chi_vals: List[float] = []
        u4_vals: List[float] = []
        for b in range(0, len(m_samples) - block + 1, block):
            ms = m_samples[b : b + block]
            chi_vals.append(susceptibility(ms, T, N))
            m2 = statistics.mean([m * m for m in ms])
            m4 = statistics.mean([m ** 4 for m in ms])
            if m2 > 0:
                u4_vals.append(1.0 - m4 / (3.0 * m2 * m2))
        chi_vals_by_rep.append(chi_vals)
        u4_vals_by_rep.append(u4_vals)

    # Use hierarchical bootstrap when we have multiple independent replicas.
    if nrep > 1:
        mean_chi, lo, hi = hierarchical_bootstrap_ci(chi_vals_by_rep, nboot=400, rng=random.Random(12345))
        mean_u4, u4_lo, u4_hi = hierarchical_bootstrap_ci(u4_vals_by_rep, nboot=400, rng=random.Random(54321))
        n_blocks = sum(len(v) for v in chi_vals_by_rep)
    else:
        mean_chi, lo, hi = bootstrap_ci(chi_vals_by_rep[0], nboot=400, rng=random.Random(12345))
        mean_u4, u4_lo, u4_hi = bootstrap_ci(u4_vals_by_rep[0], nboot=400, rng=random.Random(54321))
        n_blocks = len(chi_vals_by_rep[0])
    return T, {
        "chi": mean_chi,
        "chi_lo": lo,
        "chi_hi": hi,
        "U4": mean_u4,
        "U4_lo": u4_lo,
        "U4_hi": u4_hi,
        "n_blocks": n_blocks,
        "replicas": nrep,
    }


def _mc_single_temperature_replica(
    args: Tuple[float, float, float, int, Tuple[float, float, float], MCConfig, int]
) -> Tuple[float, int, List[float], List[float]]:
    """Worker: run ONE replica at temperature T; return block-level chi and U4 values."""
    T, J, Ja, n, phis, cfg, r = args
    L = cfg.L
    N = L * L
    neigh = build_neighbor_table(L)

    Jks = effective_Jks(J, Ja, n, phis)
    Ks = tuple(Jk / T for Jk in Jks)

    rng = random.Random((_stable_temp_seed(cfg.seed, T) + 1_664_525 * int(r)) & 0xFFFFFFFF)
    spins = np.array([1 if rng.random() < 0.5 else -1 for _ in range(N)], dtype=np.int8)

    block = 10
    chi_vals: List[float] = []
    u4_vals: List[float] = []

    # thermalize
    if cfg.update_method == "wolff":
        in_cluster = np.zeros(N, dtype=np.bool_)
        stack: List[int] = []
        cluster_sites: List[int] = []
        for _ in range(cfg.therm_sweeps):
            wolff_update(spins, neigh, Ks, rng, in_cluster, stack, cluster_sites)
    else:
        for _ in range(cfg.therm_sweeps):
            metropolis_sweep(spins, neigh, Ks, rng)

    # measure (store thinned magnetization time series)
    m_samples: List[float] = []
    if cfg.update_method == "wolff":
        in_cluster = np.zeros(N, dtype=np.bool_)
        stack = []
        cluster_sites = []
        for sweep in range(cfg.meas_sweeps):
            wolff_update(spins, neigh, Ks, rng, in_cluster, stack, cluster_sites)
            if sweep % cfg.thin == 0:
                m_samples.append(float(spins.sum()) / N)
    else:
        for sweep in range(cfg.meas_sweeps):
            metropolis_sweep(spins, neigh, Ks, rng)
            if sweep % cfg.thin == 0:
                m_samples.append(float(spins.sum()) / N)

    for b in range(0, len(m_samples) - block + 1, block):
        ms = m_samples[b : b + block]
        chi_vals.append(susceptibility(ms, T, N))
        m2 = statistics.mean([m * m for m in ms])
        m4 = statistics.mean([m ** 4 for m in ms])
        if m2 > 0:
            u4_vals.append(1.0 - m4 / (3.0 * m2 * m2))

    return T, int(r), chi_vals, u4_vals


def run_mc_scan(J: float, Ja: float, n: int, phis: Tuple[float,float,float],
                Ts: List[float], cfg: MCConfig) -> Dict[float, Dict[str,float]]:
    """
    Scan temperatures and return dict with susceptibility and CIs.
    """
    out: Dict[float, Dict[str,float]] = {}
    # Decide process count.
    nprocs = cfg.nprocs
    if nprocs == 0:
        nprocs = os.cpu_count() or 1

    # Single-process annealed scan (kept for backwards-compatibility).
    # Only enabled for the classic metropolis + single replica path.
    if (
        nprocs <= 1
        and cfg.anneal
        and cfg.update_method == "metropolis"
        and int(cfg.replicas) == 1
    ):
        rng = random.Random(cfg.seed)
        L = cfg.L
        N = L * L
        neigh = build_neighbor_table(L)
        spins = np.array([1 if rng.random() < 0.5 else -1 for _ in range(N)], dtype=np.int8)
        block = 10

        for T in Ts:
            Jks = effective_Jks(J, Ja, n, phis)
            Ks = tuple(Jk / T for Jk in Jks)

            for _ in range(cfg.therm_sweeps):
                metropolis_sweep(spins, neigh, Ks, rng)

            m_samples: List[float] = []
            for sweep in range(cfg.meas_sweeps):
                metropolis_sweep(spins, neigh, Ks, rng)
                if sweep % cfg.thin == 0:
                    m_samples.append(float(spins.sum()) / N)

            chi_vals = []
            u4_vals = []
            for b in range(0, len(m_samples) - block + 1, block):
                ms = m_samples[b : b + block]
                chi_vals.append(susceptibility(ms, T, N))
                m2 = statistics.mean([m * m for m in ms])
                m4 = statistics.mean([m ** 4 for m in ms])
                if m2 > 0:
                    u4_vals.append(1.0 - m4 / (3.0 * m2 * m2))

            mean_chi, lo, hi = bootstrap_ci(chi_vals, nboot=400, rng=rng)
            mean_u4, u4_lo, u4_hi = bootstrap_ci(u4_vals, nboot=400, rng=rng)
            out[T] = {
                "chi": mean_chi,
                "chi_lo": lo,
                "chi_hi": hi,
                "U4": mean_u4,
                "U4_lo": u4_lo,
                "U4_hi": u4_hi,
                "n_blocks": len(chi_vals),
                "replicas": 1,
            }
        return out

    # Parallel path: each temperature is simulated independently (parallel-friendly).
    # NOTE: this changes the correlation between neighboring temperatures vs the annealed scan,
    # but massively improves throughput for long scans.
    nrep = max(1, int(cfg.replicas))
    if nrep <= 1:
        work = [(T, J, Ja, n, phis, cfg) for T in Ts]
        if nprocs <= 1:
            for w in work:
                T, res = _mc_single_temperature(w)
                out[T] = res
            return out

        with mp.Pool(processes=nprocs) as pool:
            for T, res in pool.imap_unordered(_mc_single_temperature, work, chunksize=1):
                out[T] = res
        return out

    # Replica-parallel path: schedule one task per (T, replica) to better saturate many-core nodes.
    # This is especially helpful when the temperature grid is small.
    work_rep = [(T, J, Ja, n, phis, cfg, r) for T in Ts for r in range(nrep)]
    chi_by_T_rep: Dict[float, List[List[float]]] = {T: [[] for _ in range(nrep)] for T in Ts}
    u4_by_T_rep: Dict[float, List[List[float]]] = {T: [[] for _ in range(nrep)] for T in Ts}

    def consume(item: Tuple[float, int, List[float], List[float]]) -> None:
        T, r, chi_vals, u4_vals = item
        chi_by_T_rep[T][r] = chi_vals
        u4_by_T_rep[T][r] = u4_vals

    if nprocs <= 1:
        for w in work_rep:
            consume(_mc_single_temperature_replica(w))
    else:
        with mp.Pool(processes=nprocs) as pool:
            for item in pool.imap_unordered(_mc_single_temperature_replica, work_rep, chunksize=1):
                consume(item)

    # Aggregate bootstrap CIs per temperature
    for T in Ts:
        chi_vals_by_rep = chi_by_T_rep[T]
        u4_vals_by_rep = u4_by_T_rep[T]
        mean_chi, lo, hi = hierarchical_bootstrap_ci(chi_vals_by_rep, nboot=400, rng=random.Random(12345))
        mean_u4, u4_lo, u4_hi = hierarchical_bootstrap_ci(u4_vals_by_rep, nboot=400, rng=random.Random(54321))
        out[T] = {
            "chi": mean_chi,
            "chi_lo": lo,
            "chi_hi": hi,
            "U4": mean_u4,
            "U4_lo": u4_lo,
            "U4_hi": u4_hi,
            "n_blocks": sum(len(v) for v in chi_vals_by_rep),
            "replicas": nrep,
        }
    return out

def estimate_Tc_from_susceptibility(scan: Dict[float, Dict[str,float]]) -> Tuple[float,float,float]:
    """
    Take argmax of chi(T) as pseudo-Tc. Uncertainty: treat chi CIs and do
    a crude parabolic fit around the top 3 points with Monte Carlo resampling.
    """
    Ts = sorted(scan.keys())
    chis = [scan[T]["chi"] for T in Ts]
    i0 = int(np.argmax(chis))
    # fall back if at boundary
    if i0 == 0 or i0 == len(Ts)-1:
        return Ts[i0], Ts[i0], Ts[i0]

    # Parabolic fit to (T_{i-1},T_i,T_{i+1})
    def parabola_vertex(Tm, ym, T0, y0, Tp, yp):
        # fit y = aT^2 + bT + c
        A = np.array([[Tm*Tm, Tm, 1],[T0*T0, T0, 1],[Tp*Tp, Tp, 1]], float)
        y = np.array([ym,y0,yp], float)
        a,b,c = np.linalg.solve(A,y)
        if a >= 0:  # not concave
            return T0
        return -b/(2*a)

    # Resample within CI bands.
    # Important: a quadratic through noisy points can place its vertex far outside
    # the local 3-point bracket. Clamp to [Tm, Tp] to keep a physically meaningful
    # local peak estimate.
    rng = random.Random(999)
    verts = []
    for _ in range(800):
        def draw(T):
            lo,hi = scan[T]["chi_lo"], scan[T]["chi_hi"]
            return lo + (hi-lo)*rng.random()
        Tm,T0,Tp = Ts[i0-1], Ts[i0], Ts[i0+1]
        v = parabola_vertex(Tm, draw(Tm), T0, draw(T0), Tp, draw(Tp))
        if v < Tm:
            v = Tm
        elif v > Tp:
            v = Tp
        verts.append(v)
    verts.sort()
    Tc_hat = statistics.mean(verts)
    lo = verts[int(0.025*len(verts))]
    hi = verts[int(0.975*len(verts))-1]
    return Tc_hat, lo, hi

def default_phis() -> Tuple[float,float,float]:
    """Three bond angles in a triangular lattice (radians)."""
    return (0.0, 2.0*math.pi/3.0, 4.0*math.pi/3.0)

def main():
    # Parameters (edit these)
    J  = 1.0
    Ja = 1.8
    # NOTE: The paper-style "bond-phonon / case-II" setup corresponds to n=2.
    n  = 2
    phis = default_phis()

    print("Parameters:")
    print(f"  J={J}, Ja={Ja}, n={n}, phis={phis}")

    # Closed-form Tc
    t0 = time.time()
    Tc_exact = solve_Tc(J, Ja, n, phis)
    t1 = time.time()
    print(f"\nClosed-form Tc = {Tc_exact:.8f}  (solve time {t1-t0:.3f}s)")

    # Monte Carlo scan around Tc
    cfg = MCConfig(L=36, therm_sweeps=1500, meas_sweeps=2500, thin=5, seed=1)
    dT = 0.15
    Ts = [Tc_exact + dT*(k-6) for k in range(13)]
    Ts = [T for T in Ts if T>0.05]
    print(f"\nMC scan on L={cfg.L} with {len(Ts)} temperatures:")
    print("  Ts =", [round(T,4) for T in Ts])

    t2 = time.time()
    scan = run_mc_scan(J, Ja, n, phis, Ts, cfg)
    t3 = time.time()
    Tc_mc, Tc_lo, Tc_hi = estimate_Tc_from_susceptibility(scan)

    print(f"\nMC pseudo-Tc (susceptibility peak) = {Tc_mc:.6f}  [95% CI {Tc_lo:.6f}, {Tc_hi:.6f}]")
    print(f"MC wall time: {t3-t2:.1f}s")

    # Print table
    print("\nT, chi, CI, U4:")
    for T in sorted(scan.keys()):
        r = scan[T]
        u4 = r.get("U4", float("nan"))
        u4_lo = r.get("U4_lo", float("nan"))
        u4_hi = r.get("U4_hi", float("nan"))
        reps = r.get("replicas", 1)
        print(
            f"  {T:7.4f}  {r['chi']:12.6f}   [{r['chi_lo']:12.6f}, {r['chi_hi']:12.6f}]"
            f"   U4={u4: .6f} [{u4_lo: .6f}, {u4_hi: .6f}]"
            f"   blocks={r['n_blocks']} reps={reps}"
        )

    # Compare
    err = Tc_mc - Tc_exact
    print(f"\nDifference (MC - exact): {err:+.6f}")

if __name__ == "__main__":
    main()
