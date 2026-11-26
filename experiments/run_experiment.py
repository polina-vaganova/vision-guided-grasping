#!/usr/bin/env python3
"""
run_experiment.py — orchestrate N pick&place scenarios on the host.

For each scenario:
  1. Restart container (./run_ros2.sh) — clean Gazebo state.
  2. Wait for sim to come up.
  3. Wait for one new row in scenarios.csv (= scenario completed).
  4. If timeout, mark scenario as failed and continue.

Both scenarios.csv and metrics.csv accumulate across runs.

Usage:
  python3 experiments/run_experiment.py --scenarios 40 --mode yolo
  python3 experiments/run_experiment.py --scenarios 40 --mode baseline
"""
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_SH = os.path.join(ROOT, "run_ros2.sh")
SCENARIOS_CSV = os.path.join(ROOT, "experiments", "scenarios.csv")
PARAMS_YAML = os.path.join(
    ROOT, "ros2_ws", "src", "pick_place_bringup", "config", "pick_place_params.yaml"
)


def count_scenario_rows() -> int:
    if not os.path.exists(SCENARIOS_CSV):
        return 0
    with open(SCENARIOS_CSV) as f:
        return max(0, sum(1 for _ in f) - 1)  # minus header


def set_use_yolo(use_yolo: bool) -> None:
    """Patch pick_place_params.yaml use_yolo value in place."""
    with open(PARAMS_YAML) as f:
        text = f.read()
    new_val = "true" if use_yolo else "false"
    import re
    text = re.sub(
        r"(use_yolo:\s*)(true|false)",
        r"\1" + new_val,
        text,
    )
    with open(PARAMS_YAML, "w") as f:
        f.write(text)
    print(f"  → patched use_yolo: {new_val}")


def restart_container() -> None:
    print(f"  → ./run_ros2.sh")
    subprocess.run(["bash", RUN_SH], check=True, capture_output=True)


def wait_for_new_row(start_count: int, timeout: float = 600.0) -> bool:
    """Wait until scenarios.csv has more rows than start_count, or timeout."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        if count_scenario_rows() > start_count:
            return True
        time.sleep(2.0)
    return False


def kill_container() -> None:
    subprocess.run(["docker", "rm", "-f", "ros2-gazebo"],
                   capture_output=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", type=int, default=40,
                    help="number of scenarios to run")
    ap.add_argument("--mode", choices=["yolo", "baseline"], default="yolo")
    ap.add_argument("--scenario-timeout", type=float, default=600.0,
                    help="max seconds per scenario before giving up")
    args = ap.parse_args()

    print(f"=== Experiment runner ===")
    print(f"Mode: {args.mode}")
    print(f"Scenarios: {args.scenarios}")
    print(f"Scenario timeout: {args.scenario_timeout:.0f}s")
    print()

    set_use_yolo(args.mode == "yolo")

    completed = 0
    failed = 0
    t_start_total = time.time()

    for i in range(1, args.scenarios + 1):
        print(f"\n--- Scenario {i}/{args.scenarios} ---")
        before = count_scenario_rows()

        try:
            restart_container()
        except subprocess.CalledProcessError as e:
            print(f"  ✗ container start failed: {e}")
            failed += 1
            continue

        # Wait for sim to come up + scenario to complete
        if wait_for_new_row(before, timeout=args.scenario_timeout):
            completed += 1
            elapsed = time.time() - t_start_total
            avg = elapsed / i
            remain = avg * (args.scenarios - i)
            print(f"  ✓ done (total rows now: {count_scenario_rows()})")
            print(f"  Progress: {completed}/{i}  avg {avg:.0f}s/scenario  "
                  f"~{remain/60:.1f}min remaining")
        else:
            failed += 1
            print(f"  ✗ scenario timeout — scrapping")

        kill_container()

    total = time.time() - t_start_total
    print(f"\n=== DONE ===")
    print(f"Completed: {completed}/{args.scenarios}")
    print(f"Failed:    {failed}")
    print(f"Total time: {total/60:.1f} min")
    print(f"\nNext: python3 experiments/analyze_metrics.py")


if __name__ == "__main__":
    main()
