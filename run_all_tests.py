"""
run_all_tests.py
================
Master Test Runner for Tactical A2G Network Simulation.
Propagates PYTHONPATH to all child processes.

Usage:
  python run_all_tests.py --all   (Runs the entire multi-tier battery: Phases 1, 2 & 3)
  python run_all_tests.py --fast  (Runs 26 Tier-1 release gate tests across M1, M2 & M3)
  python run_all_tests.py --deep  (Runs full numerical convergence & stress sweeps)
"""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent


def run_command(
    desc: str,
    script_relative_path: str,
    python_bin: str,
    expected_token: Optional[str] = None,
) -> bool:
    full_path = str(REPO_ROOT / script_relative_path)
    print(f"\n[RUNNING] {desc} ({script_relative_path})...")
    t0 = time.perf_counter()

    env = os.environ.copy()
    py_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    venv_paths = [
        str(REPO_ROOT),
        str(REPO_ROOT / ".venv" / "lib64" / py_ver / "site-packages"),
        str(REPO_ROOT / ".venv" / "lib" / py_ver / "site-packages"),
    ]
    env["PYTHONPATH"] = os.pathsep.join([p for p in venv_paths if Path(p).exists()])

    res = subprocess.run(
        [python_bin, full_path],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    dt = time.perf_counter() - t0

    print(res.stdout, end="")
    if res.stderr:
        print(res.stderr, file=sys.stderr, end="")

    if res.returncode != 0:
        print(f"[FAILURE] {desc} exited with code {res.returncode}.")
        return False

    if expected_token and expected_token not in res.stdout:
        print(f"[CRITICAL FAILURE] {desc} exited code 0 but missing expected token: '{expected_token}'")
        return False

    print(f"[SUCCESS] {desc} completed in {dt:.2f}s.")
    return True

def main():
    parser = argparse.ArgumentParser(description="Tactical A2G Simulator Master Test Runner")
    parser.add_argument("--fast", action="store_true", help="Run Tier 1 release gates only (Phases 1-3)")
    parser.add_argument("--deep", action="store_true", help="Run Tier 2 deep numerical & stress sweeps")
    parser.add_argument("--all", action="store_true", help="Run full multi-tier battery")
    parser.add_argument("--mode", choices=["fast", "deep", "all"], default=None,
                        help="Alternative syntax: --mode {fast, deep, all}")
    args = parser.parse_args()

    if args.fast:
        mode = "fast"
    elif args.deep:
        mode = "deep"
    elif args.all or args.mode == "all" or args.mode is None:
        mode = "all"
    else:
        mode = args.mode

    python_bin = sys.executable
    success = True

    if mode in ["fast", "all"]:
        print("\n==================================================================")
        print("   TIER 1: FORMAL RELEASE ACCEPTANCE GATES (PHASES 1, 2 & 3)     ")
        print("==================================================================")
        success &= run_command(
            "Phase 1 Invariants (10 Tests)",
            "tests/test_phase1_verification.py",
            python_bin,
            "ALL 10 PHASE 1 PHYSICAL & NETWORKING TESTS PASSED ASSERTIONS.",
        )
        success &= run_command(
            "Phase 2 Gymnasium Loop (6 Tests)",
            "tests/test_phase2_verification.py",
            python_bin,
            "ALL 6 PHASE 2 GYMNASIUM ENVIRONMENT TESTS PASSED ASSERTIONS.",
        )
        success &= run_command("Phase 3 Baseline Schedulers (10 Tests)", "tests/test_phase3_verification.py", python_bin, "PHASE 3 SUMMARY: 10/10")

    if mode in ["deep", "all"]:
        print("\n==================================================================")
        print("   TIER 2: DEEP NUMERICAL CONVERGENCE & STRESS SUITE              ")
        print("==================================================================")
        success &= run_command("ITU-R P.526 Diffraction Properties & Reciprocity", "tests/phase1/test_diffraction_properties.py", python_bin)
        success &= run_command("DEM Ray-Casting Convergence Study", "tests/phase1/test_terrain_convergence.py", python_bin)
        success &= run_command("Randomized Queue Mass Conservation (1000 Ops)", "tests/phase1/test_queues_randomized_stress.py", python_bin)
        success &= run_command("10-Orbit Kinematics Multi-Period Stability", "tests/phase1/test_kinematics_stability.py", python_bin)
        success &= run_command("Full-Orbit Lookahead Culling Equivalence", "tests/phase2/test_lookahead_orbit_equivalence.py", python_bin)
        success &= run_command("Reward Monotonicity & Fairness Audit", "tests/phase2/test_rewards_and_fairness.py", python_bin)

    print("\n==================================================================")
    if success:
        print("   ALL REQUESTED DEFENSE SIMULATION TESTS PASSED RIGOROUSLY.      ")
    else:
        print("   TEST FAILURES DETECTED! REVIEW EXECUTION TRACE.                ")
    print("==================================================================\n")
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()