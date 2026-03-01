"""
beale_robustness.py — Parameter sweep + Δ-distribution fingerprint.

1. Sweeps forward-step probability p_sim from 0.1 to 0.9.
   For each p_sim, generates N_TRIALS simulated sorted-list ciphers,
   measures Lag-1 autocorrelation, and compares to empirical B1/B2/B3.
   Shows which p_sim region best fits each cipher.

2. Computes distribution of differences Δ_t = Index_t − Index_{t-1}:
   - Skewness, kurtosis
   - Fraction of Δ in [1, 10] (small positive spike)
   - Fraction of Δ negative
   This fingerprint separates sorted-list from prose-like generation.
"""
from __future__ import annotations

import math
import random
import statistics
from pathlib import Path

DATA_DIR = Path("/home/phil/.gemini/antigravity/scratch/beale-engine/data")
random.seed(7)

N_TRIALS = 300
STEP_MAX = 50


# ─── Loaders ─────────────────────────────────────────────────────────────────

def load_cipher(filename: str) -> list[int]:
    """Load space-delimited cipher file."""
    return [int(x) for x in (DATA_DIR / filename).read_text().split()]


# ─── Metrics ─────────────────────────────────────────────────────────────────

def lag1_autocorr(seq: list[int]) -> float:
    """Pearson autocorrelation at lag 1."""
    n = len(seq)
    mean = sum(seq) / n
    var = sum((x - mean) ** 2 for x in seq) / n
    if var == 0:
        return 0.0
    cov = sum((seq[i] - mean) * (seq[i + 1] - mean) for i in range(n - 1)) / (n - 1)
    return cov / var


def delta_fingerprint(seq: list[int]) -> dict:
    """Compute Δ_t = seq[t] - seq[t-1] distribution fingerprint."""
    deltas = [seq[i] - seq[i - 1] for i in range(1, len(seq))]
    n = len(deltas)
    mean_d = sum(deltas) / n
    std_d = statistics.stdev(deltas) if n > 1 else 1.0

    # Skewness and kurtosis
    if std_d == 0:
        skew = kurtosis = 0.0
    else:
        skew = sum(((d - mean_d) / std_d) ** 3 for d in deltas) / n
        kurtosis = sum(((d - mean_d) / std_d) ** 4 for d in deltas) / n - 3.0

    small_positive = sum(1 for d in deltas if 1 <= d <= 10) / n
    frac_negative = sum(1 for d in deltas if d < 0) / n
    frac_large_pos = sum(1 for d in deltas if d > 100) / n
    median_d = sorted(deltas)[n // 2]

    return {
        "mean_delta": round(mean_d, 1),
        "median_delta": median_d,
        "std_delta": round(std_d, 1),
        "skewness": round(skew, 3),
        "excess_kurtosis": round(kurtosis, 3),
        "frac_small_pos [1–10]": round(small_positive, 3),
        "frac_negative": round(frac_negative, 3),
        "frac_large_pos [>100]": round(frac_large_pos, 3),
    }


# ─── Simulator ───────────────────────────────────────────────────────────────

def simulate_sorted_list(n: int, max_idx: int, p_forward: float) -> list[int]:
    """Sorted-list cipher with variable forward probability."""
    cursor = random.randint(1, max_idx // 3)
    seq = []
    for _ in range(n):
        if random.random() < p_forward:
            step = random.randint(1, STEP_MAX)
            cursor = min(cursor + step, max_idx)
        else:
            cursor = random.randint(1, max_idx)
        seq.append(cursor)
    return seq


# ─── Parameter sweep ─────────────────────────────────────────────────────────

def parameter_sweep(
    target_lag1: float,
    n: int,
    max_idx: int,
    p_values: list[float],
) -> list[tuple[float, float, float, float]]:
    """
    For each p in p_values, simulate N_TRIALS sorted-list ciphers and
    record mean/std of lag-1 autocorrelation.
    Returns list of (p, mean_lag1, std_lag1, distance_to_target).
    """
    results = []
    for p_fwd in p_values:
        lag1s = [lag1_autocorr(simulate_sorted_list(n, max_idx, p_fwd)) for _ in range(N_TRIALS)]
        mean_l = sum(lag1s) / N_TRIALS
        std_l = statistics.stdev(lag1s)
        dist = abs(mean_l - target_lag1)
        results.append((p_fwd, round(mean_l, 4), round(std_l, 4), round(dist, 4)))
    return results


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    b1 = load_cipher("b1.txt")
    b2 = load_cipher("b2.txt")
    b3 = load_cipher("b3.txt")

    p_range = [round(p / 10, 1) for p in range(1, 10)]  # 0.1 to 0.9

    # ── 1. Parameter sweep ───────────────────────────────────────────────────
    print("=" * 70)
    print("1. PARAMETER SWEEP: Which p_forward best fits each cipher?")
    print("   (comparing simulated Lag-1 to empirical)")
    print("=" * 70)

    for cipher_name, seq in [("B1", b1), ("B2", b2), ("B3", b3)]:
        emp_lag1 = lag1_autocorr(seq)
        n, max_idx = len(seq), max(seq)
        sweep = parameter_sweep(emp_lag1, n, max_idx, p_range)

        best_p, best_mean, best_std, best_dist = min(sweep, key=lambda x: x[3])

        print(f"\n  {cipher_name} empirical Lag-1 = {emp_lag1:+.4f}")
        print(f"  {'p_fwd':>6} | {'sim Lag-1':>10} | {'std':>7} | {'|diff|':>8}")
        print(f"  {'-'*40}")
        for p_fwd, mean_l, std_l, dist in sweep:
            marker = " ◄ best" if abs(p_fwd - best_p) < 0.001 else ""
            print(f"  {p_fwd:6.1f} | {mean_l:>10.4f} | {std_l:>7.4f} | {dist:>8.4f}{marker}")
        print(f"  → Best fit: p_forward = {best_p} (simulated lag-1 = {best_mean:+.4f})")

    # ── 2. Robustness region for B3 ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("2. ROBUSTNESS CHECK: What p-range produces B3-like autocorrelation?")
    print("   (B3 empirical Lag-1 = +0.5985)")
    print("=" * 70)
    b3_lag1 = lag1_autocorr(b3)
    n3, max3 = len(b3), max(b3)
    TOLERANCE = 0.15  # Accept if sim mean within ±0.15 of empirical
    compatible = []
    for p_fwd, mean_l, std_l, dist in parameter_sweep(b3_lag1, n3, max3, p_range):
        if dist <= TOLERANCE:
            compatible.append(p_fwd)
    print(f"  p-values within ±{TOLERANCE} of B3 lag-1:")
    if compatible:
        print(f"  → Compatible range: {min(compatible):.1f} – {max(compatible):.1f}")
        print(f"  → Width: {max(compatible)-min(compatible):.1f}  (robustness band)")
    else:
        print("  → None within tolerance")

    b2_lag1 = lag1_autocorr(b2)
    n2, max2 = len(b2), max(b2)
    compatible2 = []
    for p_fwd, mean_l, std_l, dist in parameter_sweep(b2_lag1, n2, max2, p_range):
        if dist <= TOLERANCE:
            compatible2.append(p_fwd)
    print(f"\n  B2 compatible p-range: {min(compatible2):.1f} – {max(compatible2):.1f}" if compatible2 else "  B2: none compatible")

    # ── 3. Δ-distribution fingerprint ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("3. Δ-DISTRIBUTION FINGERPRINT (Index_t − Index_{t-1})")
    print("   Sorted traversal → heavy positive skew, spike at Δ=[1–10]")
    print("   Prose cipher     → broad symmetric, low skew")
    print("=" * 70)
    header = f"  {'Metric':<28} | {'B1':>10} | {'B2 (solved)':>12} | {'B3':>10}"
    print(f"\n{header}")
    print(f"  {'-'*68}")

    fp = {
        "B1": delta_fingerprint(b1),
        "B2": delta_fingerprint(b2),
        "B3": delta_fingerprint(b3),
    }
    for metric in [
        "mean_delta", "median_delta", "std_delta",
        "skewness", "excess_kurtosis",
        "frac_small_pos [1–10]", "frac_negative", "frac_large_pos [>100]",
    ]:
        v1 = fp["B1"][metric]
        v2 = fp["B2"][metric]
        v3 = fp["B3"][metric]
        print(f"  {metric:<28} | {str(v1):>10} | {str(v2):>12} | {str(v3):>10}")

    # Simulated references for comparison
    print(f"\n  {'Metric':<28} | {'PROSE sim':>12} | {'SORTED sim':>12}")
    print(f"  {'-'*58}")
    # Single-trial reference (N=1000 samples)
    prose_ref = [random.randint(1, 1005) for _ in range(763)]
    sorted_ref = simulate_sorted_list(763, 1005, 0.7)
    fp_prose = delta_fingerprint(prose_ref)
    fp_sorted = delta_fingerprint(sorted_ref)
    for metric in ["skewness", "excess_kurtosis", "frac_small_pos [1–10]", "frac_negative"]:
        vp = fp_prose[metric]
        vs = fp_sorted[metric]
        print(f"  {metric:<28} | {str(vp):>12} | {str(vs):>12}")
