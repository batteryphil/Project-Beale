"""
beale_profiler.py — Phase 1 analysis of Beale Ciphers.
Applies the Levitas information-theory methodology to B1, B2, B3.
"""
from __future__ import annotations

import collections
import math
import os
from pathlib import Path

DATA_DIR = Path("/home/phil/.gemini/antigravity/scratch/beale-engine/data")


def load_cipher(filename: str) -> list[int]:
    """Load a space-delimited cipher file into a list of integers."""
    path = DATA_DIR / filename
    with open(path, "r") as f:
        return [int(x) for x in f.read().split()]


def entropy(counts: dict) -> float:
    """Compute Shannon entropy in bits."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((v / total) * math.log2(v / total) for v in counts.values() if v > 0)


def branching_entropy(seq: list[int]) -> float:
    """Compute bigram branching entropy: H(next | current)."""
    pair_counts: dict = collections.defaultdict(collections.Counter)
    for a, b in zip(seq, seq[1:]):
        pair_counts[a][b] += 1
    total = sum(sum(c.values()) for c in pair_counts.values())
    if total == 0:
        return 0.0
    weighted = sum(
        sum(c.values()) * entropy(c)
        for c in pair_counts.values()
    )
    return weighted / total


def determinism_at_95(seq: list[int]) -> float:
    """Fraction of positions where the top-1 successor accounts for >= 95%."""
    pair_counts: dict = collections.defaultdict(collections.Counter)
    for a, b in zip(seq, seq[1:]):
        pair_counts[a][b] += 1
    det_count = 0
    total = 0
    for counter in pair_counts.values():
        total_here = sum(counter.values())
        max_prob = max(counter.values()) / total_here
        if max_prob >= 0.95:
            det_count += total_here
        total += total_here
    return det_count / total if total > 0 else 0.0


def benford_deviation(seq: list[int]) -> float:
    """Measure how closely first digits follow Benford's Law."""
    first_digits = [int(str(n)[0]) for n in seq if n > 0]
    observed = collections.Counter(first_digits)
    total = len(first_digits)
    benford_expected = {d: math.log10(1 + 1 / d) for d in range(1, 10)}
    mad = sum(
        abs(observed.get(d, 0) / total - benford_expected[d])
        for d in range(1, 10)
    ) / 9
    return mad


def distinct_ratio(seq: list[int]) -> float:
    """Ratio of distinct values to total values (vocabulary richness)."""
    return len(set(seq)) / len(seq)


def index_range(seq: list[int]) -> tuple[int, int]:
    """Min and max values in the sequence."""
    return min(seq), max(seq)


def periodicity_scan(seq: list[int], max_period: int = 40) -> list[tuple[int, float]]:
    """
    Scan for periodicity by computing autocorrelation at integer lags.
    Returns sorted (lag, correlation) pairs.
    """
    n = len(seq)
    mean = sum(seq) / n
    variance = sum((x - mean) ** 2 for x in seq) / n
    if variance == 0:
        return []
    results = []
    for lag in range(1, min(max_period + 1, n // 2)):
        cov = sum((seq[i] - mean) * (seq[i + lag] - mean) for i in range(n - lag)) / (n - lag)
        corr = cov / variance
        results.append((lag, round(corr, 4)))
    return sorted(results, key=lambda x: -abs(x[1]))


def run_profile(name: str, seq: list[int]) -> dict:
    """Run all profiling metrics on a cipher sequence."""
    results = {
        "name": name,
        "length": len(seq),
        "distinct": len(set(seq)),
        "distinct_ratio": round(distinct_ratio(seq), 4),
        "min_val": min(seq),
        "max_val": max(seq),
        "mean_val": round(sum(seq) / len(seq), 2),
        "unigram_entropy": round(entropy(collections.Counter(seq)), 4),
        "branching_entropy": round(branching_entropy(seq), 4),
        "det_at_95": round(determinism_at_95(seq), 4),
        "benford_mad": round(benford_deviation(seq), 6),
    }
    return results


def print_profile(p: dict) -> None:
    """Pretty-print a profile dictionary."""
    print(f"\n{'=' * 60}")
    print(f"  CIPHER: {p['name']}")
    print(f"{'=' * 60}")
    print(f"  Length (numbers):      {p['length']}")
    print(f"  Distinct values:       {p['distinct']} ({p['distinct_ratio']:.1%})")
    print(f"  Index range:           [{p['min_val']}, {p['max_val']}]")
    print(f"  Mean value:            {p['mean_val']:.2f}")
    print(f"  Unigram Entropy (H):   {p['unigram_entropy']:.4f} bits")
    print(f"  Branching Entropy:     {p['branching_entropy']:.4f} bits")
    print(f"  Determinism @ 95%:     {p['det_at_95']:.2%}")
    print(f"  Benford MAD:           {p['benford_mad']:.6f}")


if __name__ == "__main__":
    ciphers = {
        "B1 (Location)": load_cipher("b1.txt"),
        "B2 (Contents, SOLVED)": load_cipher("b2.txt"),
        "B3 (Names)": load_cipher("b3.txt"),
    }

    print("\nPROJECT BEALE: Phase 1 — Statistical Profiling")

    profiles = {}
    for name, seq in ciphers.items():
        p = run_profile(name, seq)
        profiles[name] = p
        print_profile(p)

    print("\n\n" + "=" * 60)
    print("COMPARATIVE SUMMARY")
    print(f"{'Metric':<28} | {'B1':>10} | {'B2 (solved)':>12} | {'B3':>10}")
    print("-" * 68)
    metrics = [
        ("Unigram Entropy", "unigram_entropy"),
        ("Branching Entropy", "branching_entropy"),
        ("Determinism @ 95%", "det_at_95"),
        ("Distinct Ratio", "distinct_ratio"),
        ("Benford MAD", "benford_mad"),
        ("Max Index", "max_val"),
    ]
    for label, key in metrics:
        b1v = profiles["B1 (Location)"][key]
        b2v = profiles["B2 (Contents, SOLVED)"][key]
        b3v = profiles["B3 (Names)"][key]
        print(f"{label:<28} | {b1v:>10} | {b2v:>12} | {b3v:>10}")

    print("\n\nPERIODICITY SCAN (top 5 lags by autocorrelation):")
    for name, seq in ciphers.items():
        top = periodicity_scan(seq, max_period=60)[:5]
        print(f"\n  {name}:")
        for lag, corr in top:
            print(f"    Lag {lag:3d}: r = {corr:+.4f}")
