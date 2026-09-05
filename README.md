# Project Beale: Statistical Cryptanalysis of the Unsolved Beale Ciphers

**Project Beale** is a computational cryptanalysis and statistical physics investigation into the 140-year-old mystery of the **Beale Ciphers**.

Through information-theoretic profiling, sequential-scan generative modeling, adversarial hypothesis testing, and period-appropriate key search, this project formalizes the **Structural Asymmetry Result**: mathematical proof that Ciphers 1 and 3 are statistically incompatible with the generative book-cipher process that produced the authenticated Cipher 2.

---

## Historical Context

In 1885, an anonymous pamphlet published in Lynchburg, Virginia, revealed three numerical substitution ciphers purportedly created in the 1820s by Thomas J. Beale. The ciphers describe a massive cache of gold, silver, and jewels buried in Bedford County, Virginia:

| Cipher | Length | Max Index | Purported Content | Status |
| :--- | :--- | :--- | :--- | :--- |
| **B1** | 520 numbers | 2,906 | Exact geographic location of the vault | **UNSOLVED** |
| **B2** | 763 numbers | 1,005 | Contents and value of the treasure | **SOLVED** (Key: Declaration of Independence) |
| **B3** | 605 numbers | 975 | Names, residences, and next of kin of the 30 party members | **UNSOLVED** |

While Cipher 2 was deciphered in the 19th century using the first letters of words in the United States Declaration of Independence, Ciphers 1 and 3 have resisted all cryptanalytic attempts for over a century.

---

## The Core Finding: The Structural Asymmetry Result

A fundamental open question in historical cryptanalysis is whether Ciphers 1 and 3 represent genuine, solvable book ciphers whose key texts remain unidentified, or if they are artificial statistical artifacts (hoaxes).

Project Beale approaches this question by treating Cipher 2 as an empirical ground-truth baseline for human-generated book-cipher encoding and subjecting all three ciphers to rigorous statistical comparison:

```
                  ┌────────────────────────────────────────────────────────┐
                  │                 THE EMPIRICAL CONTRAST                 │
                  ├──────────────────────┬─────────────────────────────────┤
                  │  Cipher 2 (SOLVED)   │  Cipher 3 (ANOMALY)             │
                  ├──────────────────────┼─────────────────────────────────┤
                  │ Lag-1 Autocorr: +0.04│ Lag-1 Autocorr: +0.5985         │
                  │ Excess Kurtosis: 2.50│ Excess Kurtosis: 10.09 (4x)     │
                  │ Generative p:   ~0.10│ Generative p:   ~0.60           │
                  │ Distinct Ratio: 23.6%│ Distinct Ratio: 42.3%           │
                  └──────────────────────┴─────────────────────────────────┘
```

### Key Statistical Verdicts:

1. **Autocorrelation Discontinuity**:
   Cipher 2 displays a near-zero lag-1 autocorrelation ($r = +0.045$), consistent with natural English text enciphered by randomly searching forward and backward through a reference text. In contrast, Cipher 3 displays an extreme autocorrelation ($r = +0.5985$, with $r = +0.427$ at lag 2), indicating strong local drift and small consecutive step sizes ($\mu_{\Delta} = 89.02$ vs $\mu_{\Delta} = 172.36$ for B2).

2. **Permutation Test Significance**:
   Non-parametric permutation testing over 1,000 resamples:
   * **B1**: $p < 0.001$ (significantly deviates from random shuffling)
   * **B3**: $p < 0.001$ (extreme departure from stationary null distribution)
   * **B2**: $p = 0.096$ (consistent with human prose enciphering)

3. **Adversarial Single-Process Rejection**:
   Evaluating a sequential-scan Markov model across 19 parameterizations of $p \in [0.05, 0.95]$ proved that **no single value of $p$ simultaneously satisfies the empirical confidence intervals for B1, B2, and B3**. The single-process hypothesis is formally rejected.

4. **Corrected Likelihood Ratio Test (LRT)**:
   Per-observation $\Delta\text{AIC}$ comparison isolates sample-size effects:
   * **B2 $\Delta\text{AIC}/\text{obs}$**: $0.6231$ (confirmed prose cipher)
   * **B3 $\Delta\text{AIC}/\text{obs}$**: $1.3161$ ($2.11\times$ the rate of B2)
   The excess structural regularity in B3 is more than double that of the known authentic prose cipher.

---

## Architecture & Analysis Pipeline

The framework is organized into modular phases:

```text
Project-Beale/
├── beale_b2_verify.py        # Phase 0: Ground-truth calibration (DOI verification)
├── beale_profiler.py         # Phase 1: Information-theoretic profiling (Entropy, Benford)
├── beale_b1b3_analysis.py    # Phase 2: Structural autocorrelation & difference dynamics
├── beale_simulation_test.py  # Phase 3: Sequential-scan generative simulator
├── beale_robustness.py       # Phase 4: Parameter sweep & stability envelope
├── beale_mixture_model.py    # Phase 5: Two-component mixture model estimation
├── beale_lrt_controls.py     # Phase 6: Likelihood Ratio Test against synthetics
├── beale_adversarial.py      # Phase 7: Adversarial single-process hypothesis teardown
├── beale_key_search.py       # Candidate key screening across 12 historical texts
├── beale_allout_solve.py     # Cryptanalytic solver: Simulated Annealing, IoC, Crib Dragging
├── beale_decode_attempt.py   # Diagnostic decode passes under alternate wordlists
├── run_analysis.py           # Unified CLI Master Runner
└── test_beale.py             # Automated unit test suite
```

---

## Quick Start & Usage

### Prerequisites
Project Beale is built entirely on the Python Standard Library (`math`, `statistics`, `collections`, `pathlib`, `re`, `urllib`). No external C extensions or heavy frameworks are required. Python 3.10+ is recommended.

```bash
git clone https://github.com/batteryphil/Project-Beale.git
cd Project-Beale
```

### Running the Unified CLI

Execute the master runner (`run_analysis.py`):

```bash
# 1. Verify ground-truth decode of Cipher 2
python run_analysis.py --verify

# 2. Run Phase 1 statistical profiling (Entropy, Benford MAD, Autocorrelations)
python run_analysis.py --profile

# 3. Run structural & consecutive difference analysis
python run_analysis.py --structural

# 4. Execute the Adversarial Single-Process Teardown
python run_analysis.py --adversarial

# 5. Run the complete core pipeline
python run_analysis.py --all
```

### Running the Test Suite

Validate data integrity, metric calculations, and portability:

```bash
python test_beale.py
```

---

## Candidate Key Document Search

`beale_key_search.py` evaluates 12 historical candidate texts from the 1776–1830 era against Cipher 1's numerical bounds:

* *The Federalist Papers* (Hamilton, Madison, Jay)
* *Common Sense* (Thomas Paine)
* *Notes on the State of Virginia* (Thomas Jefferson)
* *Writings & Addresses of George Washington*
* *Speeches of Patrick Henry*
* *The American Spelling Book* (Noah Webster)
* *The American Almanac* (1830)
* *American Geography* (Jedidiah Morse)
* *US Constitution & Bill of Rights* (Extended)

All candidates are cached locally under `data/key_candidates/`. Index of Coincidence (IoC) and n-gram scoring confirm that none of the tested candidate documents produce English plaintext from B1 without unobserved auxiliary transpositions.

---

## License & Attribution

Open-source cryptanalytic research developed as part of the Antigravity Project. Released under the MIT License.
