"""
beale_b1b3_analysis.py — Deep structural analysis of unsolved Beale Ciphers B1 and B3.
Key findings so far:
 - B2 verified against 1310-word DoI transcription.
 - B1 max index = 2906 → uses a DIFFERENT, longer key document.
 - B3 Lag-1 autocorrelation = +0.60 → anomalous structure.
"""
from __future__ import annotations
import collections
import math
import re
from pathlib import Path

DATA_DIR = Path("/home/phil/.gemini/antigravity/scratch/beale-engine/data")


def load_cipher(filename: str) -> list[int]:
    """Load space-delimited cipher file."""
    return [int(x) for x in (DATA_DIR / filename).read_text().split()]


def entropy(counts: dict) -> float:
    """Shannon entropy in bits."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((v / total) * math.log2(v / total) for v in counts.values() if v > 0)


def autocorrelation(seq: list[int], lag: int) -> float:
    """Pearson autocorrelation at a given lag."""
    n = len(seq)
    mean = sum(seq) / n
    var = sum((x - mean) ** 2 for x in seq) / n
    if var == 0:
        return 0.0
    cov = sum((seq[i] - mean) * (seq[i + lag] - mean) for i in range(n - lag)) / (n - lag)
    return cov / var


def consecutive_difference_analysis(seq: list[int]) -> dict:
    """Analyze |seq[i+1] - seq[i]|; high autocorr means small diffs."""
    diffs = [abs(seq[i + 1] - seq[i]) for i in range(len(seq) - 1)]
    return {
        "mean_diff": round(sum(diffs) / len(diffs), 2),
        "zero_diffs": diffs.count(0),
        "diff_entropy": round(entropy(collections.Counter(diffs)), 4),
        "max_diff": max(diffs),
        "small_diff_frac": round(sum(1 for d in diffs if d <= 5) / len(diffs), 4),
    }


def modular_frequency_analysis(seq: list[int], modulus: int) -> dict:
    """Check if seq values cluster in certain residue classes mod M."""
    residues = collections.Counter(n % modulus for n in seq)
    max_residue = max(residues, key=residues.get)
    uniformity = entropy(residues) / math.log2(modulus) if modulus > 1 else 1.0
    return {
        "modulus": modulus,
        "dominant_residue": max_residue,
        "dominant_count": residues[max_residue],
        "uniformity": round(uniformity, 4),
    }


def doi_partial_decode(seq: list[int], word_list: list[str]) -> tuple[str, int, int]:
    """Decode using DoI; mark out-of-range as '?'."""
    decoded = []
    oor = 0
    for n in seq:
        idx = n - 1
        if 0 <= idx < len(word_list):
            decoded.append(word_list[idx][0].upper())
        else:
            decoded.append("?")
            oor += 1
    return "".join(decoded), oor, len(seq)


if __name__ == "__main__":
    b1 = load_cipher("b1.txt")
    b2 = load_cipher("b2.txt")
    b3 = load_cipher("b3.txt")

    # Load verified DoI word list
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "beale_doi_wordlist",
        "/home/phil/.gemini/antigravity/scratch/beale-engine/beale_doi_wordlist.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    doi = list(mod.BEALE_DOI)
    print(f"DoI word count: {len(doi)}")

    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("1. B3 CONSECUTIVE DIFFERENCE ANALYSIS")
    print("   (High Lag-1 autocorr implies small diffs between numbers)")
    for name, seq in [("B1", b1), ("B2", b2), ("B3", b3)]:
        d = consecutive_difference_analysis(seq)
        print(f"  {name}: mean_diff={d['mean_diff']}, "
              f"zero={d['zero_diffs']}, "
              f"small_frac={d['small_diff_frac']:.1%}, "
              f"diff_entropy={d['diff_entropy']:.3f}")

    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("2. AUTOCORRELATION PROFILE (lags 1–10)")
    for name, seq in [("B1", b1), ("B2", b2), ("B3", b3)]:
        corrs = [autocorrelation(seq, lag) for lag in range(1, 11)]
        corr_str = "  ".join(f"{c:+.3f}" for c in corrs)
        print(f"  {name}: {corr_str}")

    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("3. MODULAR RESIDUE UNIFORMITY (B3 — cycles test)")
    for m in [2, 3, 4, 5, 7, 10, 26]:
        r = modular_frequency_analysis(b3, m)
        print(f"  B3 mod {m:2d}: dominant_residue={r['dominant_residue']}, "
              f"uniformity={r['uniformity']:.4f}")

    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("4. B1 PARTIAL DOI DECODE (numbers ≤ 1310 only)")
    b1_doi = [n for n in b1 if n <= len(doi)]
    b1_oor = [n for n in b1 if n > len(doi)]
    print(f"  B1 numbers within DoI range: {len(b1_doi)}/{len(b1)} ({len(b1_doi)/len(b1):.1%})")
    print(f"  B1 out-of-range values: {len(b1_oor)} items, max={max(b1_oor)}")

    # Decode only in-range positions
    decoded_b1 = ""
    for n in b1:
        if n <= len(doi):
            decoded_b1 += doi[n - 1][0].upper()
        else:
            decoded_b1 += "_"
    print(f"  B1 partial decode (first 120 chars):")
    print(f"  {decoded_b1[:120]}")
    print(f"  {decoded_b1[120:240]}")

    print("\n  English-like substrings in B1 partial decode:")
    import re
    sections = re.findall(r"[A-Z]{4,}", decoded_b1.replace("_", " "))
    if not words:
        # Just show readable sections
        sections = re.findall(r"[A-Z]{4,}", decoded_b1)
        if sections:
            print(f"  Long runs: {sections[:10]}")
        else:
            print("  (No long runs found — OOR positions break continuity)")

    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("5. B3 VALUE DISTRIBUTION — Is it a Name List?")
    # B3 purports to list ~30 names with next-of-kin
    # If so, we expect clustering of numbers by sub-groups
    b3_sorted = sorted(b3)
    quartiles = [
        b3_sorted[len(b3) // 4],
        b3_sorted[len(b3) // 2],
        b3_sorted[3 * len(b3) // 4],
    ]
    print(f"  B3 quartiles: Q1={quartiles[0]}, Q2={quartiles[1]}, Q3={quartiles[2]}")
    print(f"  B3 values > 500: {sum(1 for n in b3 if n > 500)}")
    print(f"  B3 values > 300: {sum(1 for n in b3 if n > 300)}")
    print(f"  B3 values ≤ 100: {sum(1 for n in b3 if n <= 100)}")
