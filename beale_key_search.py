"""
beale_key_search.py — Automated key-document search for Beale Cipher B1.

B1 constraints from structural analysis:
  - Max index = 2906 → key document must have ≥2906 words
  - 97.9% of B1 fits DoI range → some overlap with early American docs
  - Alphabetical run signature → key may be index/glossary or have special structure
  - Best-fit p_forward = 0.3 → mild sequential tendency (not strongly sorted)

Tests period-appropriate documents (1776–1830):
  1. Jefferson's Notes on the State of Virginia
  2. The Federalist Papers
  3. Common Sense (Paine)
  4. Articles of Confederation
  5. US Constitution + Bill of Rights (extended text)
  6. Virginia Statute for Religious Freedom
  7. Patrick Henry's Liberty or Death speech + other 1770s texts
  8. American Spelling Book (Noah Webster 1783)
  9. Thomas Jefferson's First Inaugural Address
  10. Various combinations and sliding windows
"""
from __future__ import annotations

import re
import math
import urllib.request
from pathlib import Path
from collections import Counter

DATA_DIR = Path("/home/phil/.gemini/antigravity/scratch/beale-engine/data")
KEY_DIR = DATA_DIR / "key_candidates"
KEY_DIR.mkdir(exist_ok=True)


def load_cipher(filename: str) -> list[int]:
    """Load space-delimited cipher file."""
    return [int(x) for x in (DATA_DIR / filename).read_text().split()]


def fetch_text(url: str, name: str) -> str:
    """Download or load cached text."""
    cache = KEY_DIR / f"{name}.txt"
    if cache.exists():
        return cache.read_text(encoding="utf-8", errors="ignore")
    try:
        print(f"  Fetching: {name}...")
        with urllib.request.urlopen(url, timeout=15) as r:
            text = r.read().decode("utf-8", errors="ignore")
        cache.write_text(text, encoding="utf-8")
        return text
    except Exception as e:
        print(f"  FAILED: {name}: {e}")
        return ""


def extract_words(text: str) -> list[str]:
    """Extract lowercase words from text."""
    return re.findall(r"[a-z']+", text.lower())


def bigram_score(text: str) -> float:
    """Score text by English bigram frequency (higher = more English-like)."""
    common_bigrams = {
        "TH", "HE", "IN", "ER", "AN", "RE", "ON", "EN", "AT", "OU",
        "ED", "ND", "TO", "EA", "TI", "NG", "OR", "IS", "IT", "AL",
        "AS", "WA", "VE", "HA", "OF", "BE", "BY", "MA", "ST", "ME",
        "RI", "WH", "NO", "SE", "AR", "CO", "LE", "DE", "EE", "NT",
    }
    t = "".join(c for c in text.upper() if c.isalpha())
    if len(t) < 2:
        return 0.0
    hits = sum(1 for i in range(len(t) - 1) if t[i:i+2] in common_bigrams)
    return hits / (len(t) - 1)


def trigram_score(text: str) -> float:
    """Score by common English trigrams."""
    common_3 = {
        "THE", "AND", "ING", "ION", "ENT", "FOR", "TIO", "ERE",
        "HER", "ATE", "HAT", "THA", "EST", "ALL", "ARE", "BUT",
    }
    t = "".join(c for c in text.upper() if c.isalpha())
    if len(t) < 3:
        return 0.0
    hits = sum(1 for i in range(len(t) - 2) if t[i:i+3] in common_3)
    return hits / (len(t) - 2)


def decode(cipher: list[int], word_list: list[str]) -> tuple[str, int]:
    """Decode and return (decoded_string, out_of_range_count)."""
    result = []
    oor = 0
    for n in cipher:
        idx = n - 1
        if 0 <= idx < len(word_list):
            result.append(word_list[idx][0].upper())
        else:
            result.append("_")
            oor += 1
    return "".join(result), oor


def find_words_in_decode(text: str, min_len: int = 4) -> list[str]:
    """Find English-looking sequences."""
    return re.findall(f"[A-Z]{{{min_len},}}", text.replace("_", " "))


def test_document(name: str, words: list[str], ciphers: dict) -> dict:
    """Score a word list against all ciphers."""
    results = {}
    for cipher_name, cipher in ciphers.items():
        decoded, oor = decode(cipher, words)
        bg = bigram_score(decoded)
        tg = trigram_score(decoded)
        oor_pct = oor / len(cipher)
        combined = bg * 0.6 + tg * 0.4
        results[cipher_name] = {
            "bigram": round(bg, 5),
            "trigram": round(tg, 5),
            "combined": round(combined, 5),
            "oor_pct": round(oor_pct, 3),
            "decoded_head": "".join(c for c in decoded if c.isalpha())[:60],
            "long_words": find_words_in_decode(decoded, 5)[:5],
        }
    return results


def sliding_window_search(
    name: str,
    words: list[str],
    cipher: list[int],
    window_size: int = 1000,
    stride: int = 100,
) -> tuple[int, float, str]:
    """
    Slide a window over the word list, testing each offset.
    Returns (best_offset, best_score, decode_head).
    """
    max_idx = max(cipher)
    best_offset = 0
    best_score = -1.0
    best_decoded = ""
    n_windows = max(1, (len(words) - window_size) // stride)

    for i in range(0, len(words) - window_size, stride):
        window = words[i:i + window_size]
        # Adjust cipher indices: if n > window_size, skip — reduces OOR
        decoded = []
        oor = 0
        for n in cipher:
            idx = n - 1
            if 0 <= idx < len(window):
                decoded.append(window[idx][0].upper())
            else:
                decoded.append("_")
                oor += 1
        decoded_str = "".join(decoded)
        score = bigram_score(decoded_str)
        if score > best_score:
            best_score = score
            best_offset = i
            best_decoded = "".join(c for c in decoded_str if c.isalpha())[:80]

    return best_offset, round(best_score, 5), best_decoded


if __name__ == "__main__":
    b1 = load_cipher("b1.txt")
    b2 = load_cipher("b2.txt")
    b3 = load_cipher("b3.txt")

    # B2 baseline
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "beale_doi_wordlist",
        "/home/phil/.gemini/antigravity/scratch/beale-engine/beale_doi_wordlist.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    doi = list(mod.BEALE_DOI)

    b2_decoded, _ = decode(b2, doi)
    b2_bg = bigram_score(b2_decoded)
    b2_tg = trigram_score(b2_decoded)
    b2_combined = b2_bg * 0.6 + b2_tg * 0.4
    print(f"B2 REFERENCE (DoI): bigram={b2_bg:.5f}, trigram={b2_tg:.5f}, "
          f"combined={b2_combined:.5f}\n")

    # ── Document catalog ─────────────────────────────────────────────────────
    docs = [
        ("DoI_standard",
         "https://www.gutenberg.org/cache/epub/1/pg1.txt",
         DATA_DIR / "declaration_of_independence.txt"),
        ("Federalist_Papers",
         "https://www.gutenberg.org/files/1404/1404-0.txt",
         None),
        ("Common_Sense_Paine",
         "https://www.gutenberg.org/files/147/147-0.txt",
         None),
        ("US_Constitution_1819",
         "https://www.gutenberg.org/files/5/5-0.txt",
         None),
        ("Jefferson_Writings",
         "https://www.gutenberg.org/files/16780/16780-0.txt",
         None),
        ("Virginia_1820_Laws",
         "https://www.gutenberg.org/files/2091/2091.txt",
         None),
        ("American_Crisis_Paine",
         "https://www.gutenberg.org/files/3741/3741-0.txt",
         None),
        ("George_Washington_Writings",
         "https://www.gutenberg.org/files/675/675-0.txt",
         None),
        ("Madison_Journal_Convention",
         "https://www.gutenberg.org/files/28043/28043-0.txt",
         None),
        ("Noah_Webster_Dissertations",
         "https://www.gutenberg.org/files/43773/43773-0.txt",
         None),
    ]

    ciphers = {"B1": b1, "B3": b3}
    all_results = []

    print("=" * 70)
    print("KEY DOCUMENT SEARCH (scoring each against B1 and B3)")
    print(f"  B2 combined reference: {b2_combined:.5f}")
    print("=" * 70)

    for doc_name, url, local_path in docs:
        if local_path and local_path.exists():
            text = local_path.read_text(encoding="utf-8", errors="ignore")
        else:
            text = fetch_text(url, doc_name)

        if not text:
            continue

        words = extract_words(text)
        if len(words) < 100:
            continue

        results = test_document(doc_name, words, ciphers)
        b1r = results["B1"]
        b3r = results["B3"]

        print(f"\n  [{doc_name}]  words={len(words)}")
        print(f"    B1: bigram={b1r['bigram']:.5f}  "
              f"trigram={b1r['trigram']:.5f}  "
              f"combined={b1r['combined']:.5f}  "
              f"oor={b1r['oor_pct']:.1%}")
        print(f"    B1 head: {b1r['decoded_head']}")
        print(f"    B3: bigram={b3r['bigram']:.5f}  "
              f"trigram={b3r['trigram']:.5f}  "
              f"combined={b3r['combined']:.5f}  "
              f"oor={b3r['oor_pct']:.1%}")
        print(f"    B3 head: {b3r['decoded_head']}")

        all_results.append((doc_name, len(words), b1r["combined"], b3r["combined"]))

    # ── Ranked summary ───────────────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("RANKED SUMMARY (B1 combined score)")
    print(f"  B2 reference: {b2_combined:.5f} (100%)")
    print("=" * 70)
    for doc_name, n_words, b1_score, b3_score in sorted(
        all_results, key=lambda x: -x[2]
    ):
        pct = b1_score / b2_combined * 100
        bar = "█" * int(pct / 3)
        print(f"  {doc_name:<30} n={n_words:6d}  "
              f"B1={b1_score:.5f} ({pct:5.1f}%)  "
              f"B3={b3_score:.5f}  {bar}")

    # ── Sliding window search on top candidates ───────────────────────────────
    print("\n\n" + "=" * 70)
    print("SLIDING WINDOW SEARCH: Best offset within top 3 documents")
    print("  (Finds if B1 decodes better starting partway through the text)")
    print("=" * 70)
    top3 = sorted(all_results, key=lambda x: -x[2])[:3]
    for doc_name, n_words, _, _ in top3:
        cache = KEY_DIR / f"{doc_name}.txt"
        if not cache.exists():
            continue
        text = cache.read_text(encoding="utf-8", errors="ignore")
        words = extract_words(text)
        if len(words) < 3000:
            print(f"  {doc_name}: too short for window search ({len(words)} words)")
            continue
        offset, score, head = sliding_window_search(doc_name, words, b1)
        pct = score / b2_bg * 100
        print(f"\n  {doc_name}: best offset={offset}, score={score:.5f} ({pct:.0f}% of B2)")
        print(f"  Decoded: {head}")
