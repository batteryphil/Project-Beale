"""
beale_allout_solve.py — Full cryptanalytic assault on Beale B1 and B3.

Techniques applied:
  1. Index of Coincidence → reveals if decoded text has substitution structure
  2. Frequency fitting → substitute letters to match English distribution
  3. Simulated annealing (hill climbing) → solve as monoalphabetic substitution
  4. Crib dragging → search for expected words (IHAVEDEP for B1 location;
                      names like JOHN WILLIAM THOMAS for B3 names)
  5. Alternate DoI letter positions → try 1st, 2nd, 3rd letter of each word
  6. Raw number analysis → try Vigenère modular decoding on raw indices
  7. B3 segment analysis → treat as 30 separate name-encodings (each ~20 chars)
"""
from __future__ import annotations

import math
import random
import string
import re
import importlib.util
from pathlib import Path
from collections import Counter

DATA_DIR = Path("/home/phil/.gemini/antigravity/scratch/beale-engine/data")
random.seed(271828)

# English letter frequencies (%)
ENGLISH_FREQ = {
    'E': 12.70, 'T': 9.06, 'A': 8.17, 'O': 7.51, 'I': 6.97, 'N': 6.75,
    'S': 6.33, 'H': 6.09, 'R': 5.99, 'D': 4.25, 'L': 4.03, 'C': 2.78,
    'U': 2.76, 'M': 2.41, 'W': 2.36, 'F': 2.23, 'G': 2.02, 'Y': 1.97,
    'P': 1.93, 'B': 1.49, 'V': 0.98, 'K': 0.77, 'J': 0.15, 'X': 0.15,
    'Q': 0.10, 'Z': 0.07,
}

COMMON_BIGRAMS = {
    "TH", "HE", "IN", "ER", "AN", "RE", "ON", "EN", "AT", "OU", "ED",
    "ND", "TO", "EA", "TI", "NG", "OR", "IS", "IT", "AL", "AS", "WA",
    "VE", "HA", "OF", "BE", "BY", "MA", "ST", "ME", "RI", "WH", "NO",
    "SE", "AR", "CO", "LE", "DE", "NT", "ES", "TE", "LY",
}


def load_cipher(filename: str) -> list[int]:
    """Load space-delimited cipher file."""
    return [int(x) for x in (DATA_DIR / filename).read_text().split()]


def get_doi() -> list[str]:
    """Load verified DoI word list."""
    spec = importlib.util.spec_from_file_location(
        "beale_doi_wordlist",
        "/home/phil/.gemini/antigravity/scratch/beale-engine/beale_doi_wordlist.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return list(mod.BEALE_DOI)


def decode_letters(cipher: list[int], words: list[str], pos: int = 0) -> str:
    """Decode cipher: take character at position `pos` of each word."""
    result = []
    for n in cipher:
        idx = n - 1
        if 0 <= idx < len(words):
            word = words[idx]
            if pos < len(word):
                result.append(word[pos].upper())
            else:
                result.append(word[-1].upper())
        else:
            result.append("_")
    return "".join(result)


def index_of_coincidence(text: str) -> float:
    """IC = Σ f_i*(f_i-1) / (N*(N-1)). English ≈ 0.065, random ≈ 0.038."""
    t = "".join(c for c in text.upper() if c.isalpha())
    n = len(t)
    if n < 2:
        return 0.0
    freq = Counter(t)
    return sum(v * (v - 1) for v in freq.values()) / (n * (n - 1))


def bigram_fitness(text: str) -> float:
    """Score by bigram frequency."""
    t = "".join(c for c in text.upper() if c.isalpha())
    if len(t) < 2:
        return 0.0
    return sum(1 for i in range(len(t) - 1) if t[i:i+2] in COMMON_BIGRAMS) / (len(t) - 1)


def apply_key(text: str, key: dict) -> str:
    """Apply substitution key dict {from → to}."""
    return "".join(key.get(c, c) for c in text)


def random_key() -> dict:
    """Generate random monoalphabetic substitution key."""
    alpha = list(string.ascii_uppercase)
    shuffled = alpha[:]
    random.shuffle(shuffled)
    return dict(zip(alpha, shuffled))


def swap_key(key: dict) -> dict:
    """Randomly swap two key entries."""
    new_key = dict(key)
    letters = list(string.ascii_uppercase)
    a, b = random.sample(letters, 2)
    # Find what a and b map to and swap
    inv = {v: k for k, v in new_key.items()}
    a_val = new_key[a]
    b_val = new_key[b]
    new_key[a] = b_val
    new_key[b] = a_val
    return new_key


def simulated_annealing(ciphertext: str, n_iter: int = 50000) -> tuple[str, dict, float]:
    """
    Hill-climbing / simulated annealing to find best substitution key.
    Works on the decoded first-letter string; treats it as monoalphabetic.
    Returns (best_decrypted, best_key, best_score).
    """
    text = "".join(c for c in ciphertext.upper() if c.isalpha())
    key = random_key()
    decrypted = apply_key(text, key)
    score = bigram_fitness(decrypted)
    best_key = dict(key)
    best_score = score
    best_decrypted = decrypted

    T = 10.0
    cooling = 0.9999

    for i in range(n_iter):
        new_key = swap_key(key)
        new_dec = apply_key(text, new_key)
        new_score = bigram_fitness(new_dec)

        delta = new_score - score
        if delta > 0 or random.random() < math.exp(delta / max(T, 1e-10)):
            key = new_key
            decrypted = new_dec
            score = new_score
            if score > best_score:
                best_score = score
                best_key = dict(key)
                best_decrypted = decrypted

        T *= cooling

    return best_decrypted, best_key, round(best_score, 5)


def crib_drag(plaintext: str, crib: str) -> list[int]:
    """Find positions where crib could match (as first letters check)."""
    positions = []
    for i in range(len(plaintext) - len(crib) + 1):
        window = plaintext[i:i+len(crib)]
        if not any(c == "_" for c in window):
            positions.append(i)
    return positions


if __name__ == "__main__":
    b1 = load_cipher("b1.txt")
    b2 = load_cipher("b2.txt")
    b3 = load_cipher("b3.txt")
    doi = get_doi()

    b2_dec = decode_letters(b2, doi)
    b2_ic = index_of_coincidence(b2_dec)
    b2_bg = bigram_fitness(b2_dec)
    print("=" * 65)
    print("B2 REFERENCE CALIBRATION")
    print("=" * 65)
    print(f"  IC: {b2_ic:.5f}  (English ≈ 0.065, random ≈ 0.038)")
    print(f"  Bigram fitness: {b2_bg:.5f}")
    print(f"  First 60: {b2_dec[:60]}")

    # ── 1. INDEX OF COINCIDENCE ──────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("1. INDEX OF COINCIDENCE")
    print("   (If IC > 0.060 → monoalphabetic structure exists → solvable)")
    print("=" * 65)
    for name, cipher in [("B1", b1), ("B2", b2), ("B3", b3)]:
        for pos in range(3):
            dec = decode_letters(cipher, doi, pos=pos)
            ic = index_of_coincidence(dec)
            flag = " ← HIGH!" if ic > 0.060 else ""
            print(f"  {name} (letter pos {pos}): IC={ic:.5f}{flag}  "
                  f"bg={bigram_fitness(dec):.5f}")

    # ── 2. FREQUENCY FITTING ─────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("2. FREQUENCY-BASED LETTER MAPPING")
    print("   (Map most-frequent decoded letter → E, T, A... in order)")
    print("=" * 65)
    for name, cipher in [("B1", b1), ("B3", b3)]:
        dec = decode_letters(cipher, doi)
        text = "".join(c for c in dec.upper() if c.isalpha())
        freq = Counter(text).most_common()
        eng_order = sorted(ENGLISH_FREQ.keys(), key=lambda k: -ENGLISH_FREQ[k])
        key = {freq[i][0]: eng_order[i] for i in range(min(len(freq), len(eng_order)))}
        substituted = "".join(key.get(c, c) for c in text)
        bg = bigram_fitness(substituted)
        print(f"  {name}: bigram after freq-sub={bg:.5f}")
        print(f"  Key mapping (top 8): {dict(list(key.items())[:8])}")
        print(f"  Result (60): {substituted[:60]}")

    # ── 3. SIMULATED ANNEALING (× 5 restarts) ────────────────────────────────
    print("\n" + "=" * 65)
    print("3. SIMULATED ANNEALING SUBSTITUTION SOLVER")
    print("   (5 independent restarts × 50k iterations each)")
    print("=" * 65)
    for name, cipher in [("B1", b1), ("B3", b3)]:
        dec = decode_letters(cipher, doi)
        best_sol = ""
        best_sc = 0.0
        best_k: dict = {}
        for restart in range(5):
            sol, k, sc = simulated_annealing(dec, n_iter=50000)
            if sc > best_sc:
                best_sc = sc
                best_sol = sol
                best_k = k
        pct = best_sc / b2_bg * 100
        print(f"\n  {name}: best bigram={best_sc:.5f} ({pct:.0f}% of B2)")
        print(f"  Result (80): {best_sol[:80]}")
        # Look for any words
        words_found = re.findall(r"[A-Z]{4,}", best_sol)
        if words_found:
            print(f"  Words (≥4): {words_found[:10]}")

    # ── 4. CRIB DRAGGING ────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("4. CRIB DRAGGING")
    print("   (B1 expected: IHAVEDEP, BEDFORD, GOLD, SILVER, BURIED)")
    print("   (B3 expected: JOHN, WILLIAM, THOMAS, ROBERT, JAMES)")
    print("=" * 65)
    b1_dec = decode_letters(b1, doi)
    b3_dec = decode_letters(b3, doi)

    print(f"\n  B1 decoded: {b1_dec}")
    b1_cribs = ["IHAVE", "GOLD", "SILVER", "BURIED", "BEDFOR", "COUNTY",
                "VAULT", "TONS", "POUNDS", "FOUR", "MILES", "STATE"]
    for crib in b1_cribs:
        pos = b1_dec.find(crib)
        if pos >= 0:
            context = b1_dec[max(0, pos-5):pos+len(crib)+5]
            print(f"  ✓ B1 crib '{crib}' at pos {pos}: ...{context}...")
    # Also check case-insensitive common sequences
    for crib in ["THE", "AND", "FOR", "ARE", "HAS", "ALL", "WAS"]:
        positions = [i for i in range(len(b1_dec)-len(crib)+1)
                     if b1_dec[i:i+len(crib)] == crib]
        if positions:
            print(f"  B1 '{crib}' at positions: {positions[:8]}")

    print(f"\n  B3 decoded (names cipher):")
    # B3 is supposed to list ~30 people's names, so look for name-initial sequences
    name_initials = [
        "JO", "WI", "TH", "RO", "JA", "CH", "GE", "AN", "ED", "HE",
        "SA", "DA", "ST", "NA", "MA", "HA", "FR", "RI", "BE", "CA"
    ]
    # Split B3 into ~30 chunks of 20
    chunk = len(b3_dec) // 30
    print(f"  B3 in 30 chunks of {chunk} (looking for name-like 2-letter starts):")
    named_chunks = []
    for i in range(30):
        c = b3_dec[i * chunk:(i + 1) * chunk]
        first2 = c[:2] if len(c) >= 2 else ""
        match = "→ possible name" if first2 in name_initials else ""
        if match:
            named_chunks.append((i + 1, c[:8], first2))
        print(f"    Entry {i+1:2d}: {c[:12]}  {match}")

    # ── 5. ALTERNATE LETTER POSITIONS ────────────────────────────────────────
    print("\n" + "=" * 65)
    print("5. ALTERNATE WORD POSITIONS IN DOI")
    print("   (What if Beale used 2nd or 3rd letter of each word?)")
    print("=" * 65)
    for name, cipher in [("B1", b1), ("B3", b3)]:
        for pos in range(1, 4):
            dec = decode_letters(cipher, doi, pos=pos)
            ic = index_of_coincidence(dec)
            bg = bigram_fitness(dec)
            text = "".join(c for c in dec if c.isalpha() or c == "_")
            print(f"  {name} pos={pos}: IC={ic:.4f}  bg={bg:.4f}  "
                  f"head={text[:40]}")

    # ── 6. VIGENÈRE PERIOD DETECTION ────────────────────────────────────────
    print("\n" + "=" * 65)
    print("6. VIGENÈRE / PERIODIC KEY DETECTION")
    print("   (Kasiski: look for repeating index subsequences in raw cipher)")
    print("=" * 65)
    for name, cipher in [("B1", b1), ("B3", b3)]:
        # Convert to mod-26 for Vigenère analysis
        dec = [n % 26 for n in cipher]
        # Compute IC for each period guess
        ics = {}
        for period in range(1, 15):
            streams = [[] for _ in range(period)]
            for i, v in enumerate(dec):
                streams[i % period].append(v)
            avg_ic = 0.0
            for stream in streams:
                n = len(stream)
                freq_ic = Counter(stream)
                if n > 1:
                    ic_val = sum(f * (f - 1) for f in freq_ic.values()) / (n * (n - 1))
                    avg_ic += ic_val
            ics[period] = round(avg_ic / period, 5)
        best_period = max(ics, key=ics.get)
        print(f"  {name} best period={best_period}, IC={ics[best_period]:.5f}")
        print(f"  ICs: {ics}")
