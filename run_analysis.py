"""
run_analysis.py — Unified Master CLI for Project-Beale.
Coordinates multi-phase statistical cryptanalysis of the Beale Ciphers.
"""
from __future__ import annotations

import argparse
import sys
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

def run_script(script_name: str, desc: str) -> int:
    script_path = BASE_DIR / script_name
    print("\n" + "=" * 70)
    print(f"  RUNNING: {desc} ({script_name})")
    print("=" * 70)
    result = subprocess.run([sys.executable, str(script_path)], cwd=str(BASE_DIR))
    return result.returncode

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Project Beale: Statistical Cryptanalysis of the Beale Ciphers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_analysis.py --verify        # Verify B2 decode against DOI
  python run_analysis.py --profile       # Run Information-Theoretic Profiler
  python run_analysis.py --adversarial   # Run Single-Process Teardown
  python run_analysis.py --all           # Run Core Verification & Statistical Pipeline
        """
    )
    parser.add_argument("--verify", action="store_true", help="Run B2 ground-truth verification")
    parser.add_argument("--profile", action="store_true", help="Run Phase 1 Information-Theoretic Profiler")
    parser.add_argument("--structural", action="store_true", help="Run B1/B3 structural & autocorrelation analysis")
    parser.add_argument("--simulation", action="store_true", help="Run sequential-scan simulation tests")
    parser.add_argument("--robustness", action="store_true", help="Run parameter sweep and robustness analysis")
    parser.add_argument("--mixture", action="store_true", help="Run mixture model estimation")
    parser.add_argument("--lrt", action="store_true", help="Run Likelihood Ratio Tests (LRT)")
    parser.add_argument("--adversarial", action="store_true", help="Run Phase 7 adversarial teardown")
    parser.add_argument("--solve", action="store_true", help="Run all-out solve attempts (annealing, IoC, cribs)")
    parser.add_argument("--key-search", action="store_true", help="Run key document candidate search")
    parser.add_argument("--all", action="store_true", help="Run standard core pipeline (verify, profile, structural, adversarial)")

    args = parser.parse_args()

    # If no flags passed, default to --verify and print help
    if not any(vars(args).values()):
        parser.print_help()
        print("\n[Defaulting to --verify for quick smoke-test]:")
        return run_script("beale_b2_verify.py", "B2 Ground-Truth Verification")

    exit_code = 0
    if args.all:
        scripts = [
            ("beale_b2_verify.py", "Phase 0: B2 Ground-Truth Calibration"),
            ("beale_profiler.py", "Phase 1: Information-Theoretic Profiling"),
            ("beale_b1b3_analysis.py", "Phase 2: B1 & B3 Structural & Autocorrelation Analysis"),
            ("beale_adversarial.py", "Phase 7: Adversarial Single-Process Teardown"),
        ]
        for script, desc in scripts:
            code = run_script(script, desc)
            if code != 0:
                exit_code = code
        return exit_code

    if args.verify:
        exit_code |= run_script("beale_b2_verify.py", "B2 Ground-Truth Verification")
    if args.profile:
        exit_code |= run_script("beale_profiler.py", "Information-Theoretic Profiling")
    if args.structural:
        exit_code |= run_script("beale_b1b3_analysis.py", "Structural & Autocorrelation Analysis")
    if args.simulation:
        exit_code |= run_script("beale_simulation_test.py", "Sequential-Scan Simulation Test")
    if args.robustness:
        exit_code |= run_script("beale_robustness.py", "Parameter Sweep & Robustness Analysis")
    if args.mixture:
        exit_code |= run_script("beale_mixture_model.py", "Mixture Model Estimation")
    if args.lrt:
        exit_code |= run_script("beale_lrt_controls.py", "Likelihood Ratio Test Controls")
    if args.adversarial:
        exit_code |= run_script("beale_adversarial.py", "Adversarial Single-Process Teardown")
    if args.solve:
        exit_code |= run_script("beale_allout_solve.py", "All-Out Solve Attempts")
    if args.key_search:
        exit_code |= run_script("beale_key_search.py", "Key Document Candidate Search")

    return exit_code

if __name__ == "__main__":
    sys.exit(main())
