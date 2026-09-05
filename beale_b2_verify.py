"""
beale_b2_verify.py — Verify B2 decodes correctly against the Declaration of Independence.
This is the ground-truth calibration step before tackling unsolved B1 and B3.
"""
from __future__ import annotations

import re
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DOI_PATH = DATA_DIR / "declaration_of_independence.txt"
DOI_URL = "https://www.gutenberg.org/cache/epub/1/pg1.txt"


def fetch_doi() -> str:
    """Fetch and cache the Declaration of Independence text."""
    if DOI_PATH.exists():
        return DOI_PATH.read_text(encoding="utf-8", errors="replace")
    print(f"Fetching DoI from {DOI_URL}...")
    with urllib.request.urlopen(DOI_URL, timeout=10) as r:
        text = r.read().decode("utf-8", errors="replace")
    DOI_PATH.write_text(text, encoding="utf-8")
    return text



def build_word_list(text: str) -> list[str]:
    """Extract words from text, stripping all punctuation, lowercased."""
    # Extract only the Declaration content (after the *** line)
    start = text.find("IN CONGRESS, July 4, 1776")
    if start == -1:
        start = 0
    content = text[start:]
    words = re.findall(r"[a-zA-Z']+", content)
    return [w.lower() for w in words]


def decode_cipher(cipher_nums: list[int], word_list: list[str]) -> str:
    """
    Book cipher decoding: each number N → first letter of word N (1-indexed).
    Returns decoded string.
    """
    decoded = []
    errors = 0
    for n in cipher_nums:
        idx = n - 1  # 1-indexed
        if 0 <= idx < len(word_list):
            decoded.append(word_list[idx][0].upper())
        else:
            decoded.append(f"[{n}?]")
            errors += 1
    return "".join(decoded), errors


def load_cipher(filename: str) -> list[int]:
    """Load space-delimited cipher file."""
    path = DATA_DIR / filename
    return [int(x) for x in path.read_text().split()]


if __name__ == "__main__":
    from beale_doi_wordlist import BEALE_DOI

    print("=================================================================")
    print("BEALE CIPHER 2 GROUND-TRUTH VERIFICATION")
    print("=================================================================")
    
    words = list(BEALE_DOI)
    print(f"Beale-variant DoI word list length: {len(words)}")
    print(f"First 20 words: {words[:20]}")

    b2 = load_cipher("b2.txt")
    b2_decoded, errors = decode_cipher(b2, words)

    print(f"\nB2 LENGTH: {len(b2)} numbers")
    print(f"OUT-OF-RANGE ERRORS: {errors}")
    print(f"\nFirst 120 chars:\n{b2_decoded[:120]}")

    known_fragment = "IHAVEDEPOSITED"
    if known_fragment in b2_decoded:
        print(f"\n✓ VERIFIED: Ground-truth plaintext '{known_fragment}' successfully recovered.")
    else:
        print(f"\n✗ WARNING: '{known_fragment}' NOT found.")


    # Now also test B1 max index against this word list length
    b1 = load_cipher("b1.txt")
    b3 = load_cipher("b3.txt")
    print(f"\nB1 max index: {max(b1)} vs DoI length: {len(words)}")
    print(f"B3 max index: {max(b3)} vs DoI length: {len(words)}")
    b1_oor = sum(1 for n in b1 if n > len(words))
    b3_oor = sum(1 for n in b3 if n > len(words))
    print(f"B1 out-of-range for DoI: {b1_oor}/{len(b1)} ({b1_oor/len(b1):.1%})")
    print(f"B3 out-of-range for DoI: {b3_oor}/{len(b3)} ({b3_oor/len(b3):.1%})")
