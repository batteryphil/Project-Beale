"""
beale_decode_attempt.py — Decryption attempts for B1 and B3 using structural insights.

Hypotheses from structural analysis:
  B1: Alphabetical runs in partial DoI decode → try SORTED DoI as key
  B3: Sequential scanning pattern → try standard DoI; also try B2-style key variations

This may fail — the Beale ciphers have resisted 140 years of attempts.
But these are structurally motivated, not random.
"""
from __future__ import annotations

import re
import math
from pathlib import Path
from collections import Counter

DATA_DIR = Path("/home/phil/.gemini/antigravity/scratch/beale-engine/data")


def load_cipher(filename: str) -> list[int]:
    """Load space-delimited cipher file."""
    return [int(x) for x in (DATA_DIR / filename).read_text().split()]


def score_english(text: str) -> float:
    """
    Score a string for English-likeness using common word frequency.
    Higher is more English-like.
    """
    common = set("THEANDOFTOAINS".split() +
                 list("THEANDOFTOAINISATWITHFORINHEBYHISATFROMTHEYAREBUT"))
    text = text.upper()
    # Common bigrams in English
    bigrams = {"TH","HE","IN","ER","AN","RE","ON","EN","AT","OU","ED","ND","TO","EA"}
    # Count matching bigrams
    score = 0.0
    for i in range(len(text) - 1):
        if text[i:i+2] in bigrams:
            score += 1.0
    # Penalise Q, X, Z runs
    for c in "QXZ":
        score -= text.count(c) * 0.5
    return score / max(len(text), 1)


def decode(cipher: list[int], word_list: list[str]) -> tuple[str, int]:
    """Decode cipher using word_list. Returns (decoded_string, out_of_range_count)."""
    decoded = []
    oor = 0
    for n in cipher:
        idx = n - 1
        if 0 <= idx < len(word_list):
            decoded.append(word_list[idx][0].upper())
        else:
            decoded.append("_")
            oor += 1
    return "".join(decoded), oor


def find_words(text: str, min_len: int = 4) -> list[str]:
    """Find English-ish letter runs of minimum length."""
    return re.findall(f"[A-Z]{{{min_len},}}", text.replace("_", " "))


if __name__ == "__main__":
    b1 = load_cipher("b1.txt")
    b2 = load_cipher("b2.txt")
    b3 = load_cipher("b3.txt")

    # Load the verified DoI word list
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "beale_doi_wordlist",
        "/home/phil/.gemini/antigravity/scratch/beale-engine/beale_doi_wordlist.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    doi = list(mod.BEALE_DOI)

    # Verify B2 still works
    b2_decoded, _ = decode(b2, doi)
    b2_clean = "".join(c for c in b2_decoded if c.isalpha())
    assert "IHAVEDEPOSITED" in b2_clean, "B2 verification failed!"
    print("✓ B2 VERIFIED\n")

    # ── B3: Try standard DoI ──────────────────────────────────────────────────
    print("=" * 65)
    print("B3 DECODE ATTEMPT 1: Standard DoI (same as B2)")
    print("=" * 65)
    b3_doi, b3_oor = decode(b3, doi)
    b3_clean = "".join(c for c in b3_doi if c.isalpha())
    print(f"OOR: {b3_oor}/605\n")
    print("Raw (first 200):")
    print(b3_doi[:200])
    print("\nClean (first 200):")
    print(b3_clean[:200])
    print(f"\nEnglish-likeness score: {score_english(b3_clean):.5f}")
    runs = find_words(b3_clean, 5)
    print(f"Long runs (≥5): {runs[:15]}")

    # ── B1: Try standard DoI ──────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("B1 DECODE ATTEMPT 1: Standard DoI (known to produce alpha runs)")
    print("=" * 65)
    b1_doi, b1_oor = decode(b1, doi)
    b1_clean = "".join(c for c in b1_doi if c.isalpha())
    print(f"OOR: {b1_oor}/520 (known: 11 numbers exceed DoI)\n")
    print("Clean (first 200):")
    print(b1_clean[:200])
    print(f"\nEnglish-likeness score: {score_english(b1_clean):.5f}")
    runs = find_words(b1_clean, 5)
    print(f"Long runs (≥5): {runs[:10]}")

    # ── B1: Try SORTED DoI ───────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("B1 DECODE ATTEMPT 2: ALPHABETICALLY SORTED DoI")
    print("  Hypothesis: B1 used an alphabetised wordlist (explains alpha runs)")
    print("=" * 65)
    doi_sorted = sorted(doi)
    b1_sorted, b1s_oor = decode(b1, doi_sorted)
    b1s_clean = "".join(c for c in b1_sorted if c.isalpha())
    print(f"OOR: {b1s_oor}/520\n")
    print("Clean (first 200):")
    print(b1s_clean[:200])
    print(f"\nEnglish-likeness score: {score_english(b1s_clean):.5f}")
    runs = find_words(b1s_clean, 5)
    print(f"Long runs (≥5): {runs[:10]}")

    # ── B1: Try FREQUENCY-SORTED DoI ─────────────────────────────────────────
    print("\n" + "=" * 65)
    print("B1 DECODE ATTEMPT 3: FREQUENCY-SORTED DoI (most common first)")
    print("  Hypothesis: encoded using a word frequency table")
    print("=" * 65)
    word_counts = Counter(doi)
    # Sort by frequency descending, then alpha for ties
    doi_freq = sorted(doi, key=lambda w: (-word_counts[w], w))
    doi_freq_deduped = []
    seen = set()
    for w in doi_freq:
        if w not in seen:
            doi_freq_deduped.append(w)
            seen.add(w)
    b1_freq, b1f_oor = decode(b1, doi_freq_deduped)
    b1f_clean = "".join(c for c in b1_freq if c.isalpha())
    print(f"OOR: {b1f_oor}/520\n")
    print("Clean (first 200):")
    print(b1f_clean[:200])
    print(f"\nEnglish-likeness score: {score_english(b1f_clean):.5f}")
    runs = find_words(b1f_clean, 5)
    print(f"Long runs (≥5): {runs[:10]}")

    # ── B3: Try SORTED DoI ───────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("B3 DECODE ATTEMPT 2: SORTED DoI")
    print("  Hypothesis: names encoded from alphabetical register")
    print("=" * 65)
    b3_sorted, b3s_oor = decode(b3, doi_sorted)
    b3s_clean = "".join(c for c in b3_sorted if c.isalpha())
    print(f"OOR: {b3s_oor}/605\n")
    print("Clean (first 200):")
    print(b3s_clean[:200])
    print(f"\nEnglish-likeness score: {score_english(b3s_clean):.5f}")
    runs = find_words(b3s_clean, 5)
    print(f"Long runs (≥5): {runs[:10]}")

    # ── Comparative summary ──────────────────────────────────────────────────
    print("\n\n" + "=" * 65)
    print("COMPARATIVE ENGLISH SCORE SUMMARY")
    print("  (B2 reference provided for calibration)")
    print("=" * 65)
    b2_score = score_english(b2_clean)
    scores = [
        ("B2 reference (SOLVED)", b2_score),
        ("B1 + standard DoI", score_english(b1_clean)),
        ("B1 + sorted DoI", score_english(b1s_clean)),
        ("B1 + freq-sorted DoI", score_english(b1f_clean)),
        ("B3 + standard DoI", score_english(b3_clean)),
        ("B3 + sorted DoI", score_english(b3s_clean)),
    ]
    for label, score in sorted(scores, key=lambda x: -x[1]):
        bar = "█" * int(score * 200)
        pct = score / b2_score * 100
        print(f"  {label:<30} {score:.5f}  ({pct:.0f}% of B2)  {bar}")
