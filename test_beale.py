"""
test_beale.py — Automated Unit Tests for Project-Beale.
Validates ground-truth decoding, data integrity, metric calculations, and portability.
"""
from __future__ import annotations

import unittest
from pathlib import Path
from beale_doi_wordlist import BEALE_DOI
from beale_b2_verify import decode_cipher, load_cipher
from beale_b1b3_analysis import autocorrelation
from beale_profiler import entropy, branching_entropy

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

class TestProjectBeale(unittest.TestCase):

    def test_data_integrity(self):
        """Ensure all required cipher data files exist and are populated."""
        for filename, expected_count in [("b1.txt", 520), ("b2.txt", 763), ("b3.txt", 605)]:
            cipher = load_cipher(filename)
            self.assertEqual(len(cipher), expected_count, f"{filename} length mismatch")
            self.assertTrue(all(isinstance(x, int) for x in cipher))

    def test_b2_ground_truth_decode(self):
        """B2 must decode with 0 errors and recover the known plaintext fragment."""
        b2 = load_cipher("b2.txt")
        decoded, errors = decode_cipher(b2, list(BEALE_DOI))
        self.assertEqual(errors, 0, "B2 should have 0 out-of-range errors against Beale DOI")
        self.assertIn("IHAVEDEPOSITEDINTHECOUNTFOFBEDFORD", decoded)

    def test_cipher_max_indices(self):
        """Validate key document index boundaries."""
        b1 = load_cipher("b1.txt")
        b2 = load_cipher("b2.txt")
        b3 = load_cipher("b3.txt")

        self.assertEqual(max(b1), 2906, "B1 max index should be 2906")
        self.assertEqual(max(b2), 1005, "B2 max index should be 1005")
        self.assertEqual(max(b3), 975, "B3 max index should be 975")

    def test_statistical_metrics(self):
        """Verify metric calculation functions produce expected baseline ranges."""
        b2 = load_cipher("b2.txt")
        b3 = load_cipher("b3.txt")

        # B3 has an anomalous, known high Lag-1 autocorrelation (>0.55)
        b3_lag1 = autocorrelation(b3, 1)
        self.assertGreater(b3_lag1, 0.55, "B3 lag-1 autocorrelation should be > 0.55")

        # B2 has low autocorrelation (<0.15)
        b2_lag1 = autocorrelation(b2, 1)
        self.assertLess(abs(b2_lag1), 0.15, "B2 lag-1 autocorrelation should be near zero")

    def test_no_hardcoded_paths(self):
        """Ensure no Python files contain hardcoded machine paths."""
        py_files = [f for f in BASE_DIR.glob("*.py") if f.name != "test_beale.py"]
        forbidden_terms = ["beale" + "-" + "engine", "/home" + "/phil"]
        for f in py_files:
            content = f.read_text(encoding="utf-8")
            for term in forbidden_terms:
                self.assertNotIn(term, content, f"Hardcoded '{term}' found in {f.name}")

if __name__ == "__main__":
    unittest.main(verbosity=2)
