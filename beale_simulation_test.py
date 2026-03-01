"""
beale_simulation_test.py — Decisive structural test for Beale B1, B2, B3.

Generates three synthetic cipher types:
  1. TRUE_PROSE   — genuine book cipher using a random english prose text
  2. SORTED_LIST  — cipher where indices come from scanning a sorted word list
  3. RANDOM_NOISE — pure random integers in the observed index range

Compares Lag autocorrelation spectrum, mean diff, monotonic run distribution
against the empirical B1, B2, B3 data.

If B3 matches SORTED_LIST rather than TRUE_PROSE → strong structural evidence
of fabrication or anomalous generation method.
"""
from __future__ import annotations

import collections
import math
import random
import re
from pathlib import Path

DATA_DIR = Path("/home/phil/.gemini/antigravity/scratch/beale-engine/data")
random.seed(42)  # Reproducible


# ─── Helper functions ────────────────────────────────────────────────────────

def load_cipher(filename: str) -> list[int]:
    """Load space-delimited cipher file."""
    return [int(x) for x in (DATA_DIR / filename).read_text().split()]


def autocorr_profile(seq: list[int], lags: int = 10) -> list[float]:
    """Compute Pearson autocorrelation for lags 1..lags."""
    n = len(seq)
    mean = sum(seq) / n
    var = sum((x - mean) ** 2 for x in seq) / n
    if var == 0:
        return [0.0] * lags
    result = []
    for lag in range(1, lags + 1):
        cov = sum((seq[i] - mean) * (seq[i + lag] - mean) for i in range(n - lag)) / (n - lag)
        result.append(round(cov / var, 4))
    return result


def mean_consecutive_diff(seq: list[int]) -> float:
    """Mean |seq[i+1] - seq[i]|."""
    diffs = [abs(seq[i + 1] - seq[i]) for i in range(len(seq) - 1)]
    return sum(diffs) / len(diffs)


def monotonic_run_stats(seq: list[int]) -> dict:
    """Stats on monotonically increasing runs."""
    runs = []
    run_len = 1
    for i in range(1, len(seq)):
        if seq[i] > seq[i - 1]:
            run_len += 1
        else:
            if run_len > 1:
                runs.append(run_len)
            run_len = 1
    if run_len > 1:
        runs.append(run_len)
    if not runs:
        return {"mean_run": 0, "max_run": 0, "runs_gt5": 0}
    return {
        "mean_run": round(sum(runs) / len(runs), 2),
        "max_run": max(runs),
        "runs_gt5": sum(1 for r in runs if r > 5),
    }


def fraction_positive_diffs(seq: list[int]) -> float:
    """Fraction of consecutive differences that are positive."""
    diffs = [seq[i + 1] - seq[i] for i in range(len(seq) - 1)]
    return sum(1 for d in diffs if d > 0) / len(diffs)


def profile(name: str, seq: list[int]) -> dict:
    """Full structural profile of a sequence."""
    return {
        "name": name,
        "n": len(seq),
        "autocorr": autocorr_profile(seq),
        "mean_abs_diff": round(mean_consecutive_diff(seq), 1),
        "frac_pos_diff": round(fraction_positive_diffs(seq), 3),
        "run_stats": monotonic_run_stats(seq),
    }


# ─── Simulators ──────────────────────────────────────────────────────────────

def _extract_words(text: str) -> list[str]:
    """Clean word list from text."""
    return re.findall(r"[a-z']+", text.lower())


def simulate_prose_cipher(word_list: list[str], n: int, max_idx: int) -> list[int]:
    """
    Genuine book cipher on prose:
    - Message words are sampled from a target distribution.
    - For each message letter, pick a RANDOM word starting with that letter.
    - Index assignment is random among all matching words → low autocorr.
    """
    # Build letter → list of word indices
    by_letter: dict = collections.defaultdict(list)
    for i, w in enumerate(word_list[:max_idx], start=1):
        by_letter[w[0]].append(i)

    # Sample random target letters (uniform over a-z)
    letters = [chr(ord('a') + random.randint(0, 25)) for _ in range(n)]
    indices = []
    for letter in letters:
        candidates = by_letter.get(letter, [])
        if candidates:
            indices.append(random.choice(candidates))
        else:
            indices.append(random.randint(1, max_idx))
    return indices


def simulate_sorted_list_cipher(n: int, max_idx: int) -> list[int]:
    """
    Sorted-list cipher simulation:
    - Imagine a list of ~30 names, each with multiple sub-entries.
    - Names are scanned top-to-bottom, left-to-right within a printed page.
    - Within each 'entry block', indices increase monotonically in small steps.
    - Jumps occur when moving to a distant page.
    """
    indices = []
    cursor = random.randint(1, max_idx // 4)

    for i in range(n):
        # ~70% chance: small forward step (same page/column)
        if random.random() < 0.70:
            step = random.randint(1, 30)
            cursor = min(cursor + step, max_idx)
        else:
            # Jump to a random new location (different entry)
            cursor = random.randint(1, max_idx)
        indices.append(cursor)
    return indices


def simulate_random_cipher(n: int, max_idx: int) -> list[int]:
    """
    Pure random integers in [1, max_idx]: baseline null model.
    """
    return [random.randint(1, max_idx) for _ in range(n)]


# ─── Main ────────────────────────────────────────────────────────────────────

def print_profile(p: dict) -> None:
    """Pretty-print a profile."""
    ac = p["autocorr"]
    ac_str = "  ".join(f"{c:+.3f}" for c in ac[:6])
    print(f"\n  [{p['name']}]  n={p['n']}")
    print(f"    AutoCorr (lag 1–6):  {ac_str}")
    print(f"    Mean |diff|:         {p['mean_abs_diff']}")
    print(f"    Frac pos diffs:      {p['frac_pos_diff']:.1%}")
    rs = p["run_stats"]
    print(f"    Monotonic runs:      mean={rs['mean_run']}, max={rs['max_run']}, >=5: {rs['runs_gt5']}")


if __name__ == "__main__":
    # Load real ciphers
    b1 = load_cipher("b1.txt")
    b2 = load_cipher("b2.txt")
    b3 = load_cipher("b3.txt")

    # Load the DoI as our prose text
    doi_text = (DATA_DIR / "declaration_of_independence.txt").read_text()
    doi_words = _extract_words(doi_text)

    # Parameters for simulation anchored to each cipher's stats
    configs = {
        "B1": (len(b1), max(b1)),
        "B2": (len(b2), max(b2)),
        "B3": (len(b3), max(b3)),
    }

    # Run N_TRIALS simulations and average the autocorr at lag 1
    N_TRIALS = 200

    print("=" * 65)
    print("PROJECT BEALE: Simulation-Based Structural Test")
    print("=" * 65)

    print("\n--- EMPIRICAL CIPHERS ---")
    for name, seq in [("B1 (empirical)", b1), ("B2 (empirical)", b2), ("B3 (empirical)", b3)]:
        print_profile(profile(name, seq))

    print("\n--- SIMULATIONS (averaged over", N_TRIALS, "trials) ---")
    for cipher_name, (n, max_idx) in configs.items():
        # For each simulation type, average key metrics
        for sim_name, sim_fn in [
            ("PROSE", lambda n=n, m=max_idx: simulate_prose_cipher(doi_words, n, m)),
            ("SORTED_LIST", lambda n=n, m=max_idx: simulate_sorted_list_cipher(n, m)),
            ("RANDOM", lambda n=n, m=max_idx: simulate_random_cipher(n, m)),
        ]:
            lag1_acc = []
            mean_diff_acc = []
            pos_diff_acc = []
            max_run_acc = []
            for _ in range(N_TRIALS):
                sim = sim_fn()
                lag1_acc.append(autocorr_profile(sim, 1)[0])
                mean_diff_acc.append(mean_consecutive_diff(sim))
                pos_diff_acc.append(fraction_positive_diffs(sim))
                max_run_acc.append(monotonic_run_stats(sim)["max_run"])

            avg_lag1 = sum(lag1_acc) / N_TRIALS
            avg_diff = sum(mean_diff_acc) / N_TRIALS
            avg_pos = sum(pos_diff_acc) / N_TRIALS
            avg_run = sum(max_run_acc) / N_TRIALS
            print(f"\n  [{cipher_name} | {sim_name}]  n={n}, max_idx={max_idx}")
            print(f"    Avg Lag-1 autocorr:  {avg_lag1:+.4f}")
            print(f"    Avg mean |diff|:     {avg_diff:.1f}")
            print(f"    Avg frac pos diffs:  {avg_pos:.1%}")
            print(f"    Avg max run:         {avg_run:.1f}")

    print("\n" + "=" * 65)
    print("VERDICT TABLE: Which model best matches each cipher?")
    print("(Higher |autocorr| = sorted-list; near 0 = prose/random)")
    print("=" * 65)
    print(f"  {'Cipher':<10} Empirical Lag-1   PROSE sim   SORTED sim   RANDOM sim")
    print(f"  {'-'*62}")
    for cipher_name, seq in [("B1", b1), ("B2", b2), ("B3", b3)]:
        emp = autocorr_profile(seq, 1)[0]
        n, max_idx = len(seq), max(seq)
        p_trials = [autocorr_profile(simulate_prose_cipher(doi_words, n, max_idx), 1)[0] for _ in range(50)]
        s_trials = [autocorr_profile(simulate_sorted_list_cipher(n, max_idx), 1)[0] for _ in range(50)]
        r_trials = [autocorr_profile(simulate_random_cipher(n, max_idx), 1)[0] for _ in range(50)]
        mp = sum(p_trials) / 50
        ms = sum(s_trials) / 50
        mr = sum(r_trials) / 50
        # Which is closest to empirical?
        diffs = {"PROSE": abs(emp - mp), "SORTED": abs(emp - ms), "RANDOM": abs(emp - mr)}
        best = min(diffs, key=diffs.get)
        print(f"  {cipher_name:<10} {emp:+.4f}          {mp:+.4f}     {ms:+.4f}      {mr:+.4f}   → {best}")
