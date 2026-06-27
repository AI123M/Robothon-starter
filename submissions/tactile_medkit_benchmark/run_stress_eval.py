import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from submissions.tactile_medkit_benchmark.simulation import run_stress_eval
from submissions.tactile_medkit_benchmark.task_model import OUTPUTS_DIR


def main() -> int:
    parser = argparse.ArgumentParser(description="Run multi-seed stress evaluation.")
    parser.add_argument("--seeds", type=int, default=128)
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR)
    args = parser.parse_args()

    summary = run_stress_eval(seeds=args.seeds, output_dir=args.output_dir)
    print(f"RUNS={summary['runs']}")
    print(f"SUCCESS_RATE={summary['success_rate']:.3f}")
    print(f"WORST_SLIP_MM={summary['worst_slip_mm']}")
    print(f"WORST_PLACEMENT_ERROR_MM={summary['worst_placement_error_mm']}")
    return 0 if summary["success_rate"] >= 0.875 else 1


if __name__ == "__main__":
    raise SystemExit(main())
