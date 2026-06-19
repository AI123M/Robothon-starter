import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from submissions.tactile_medkit_benchmark.simulation import run_benchmark
from submissions.tactile_medkit_benchmark.task_model import OUTPUTS_DIR


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Tactile MedKit benchmark.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR)
    parser.add_argument("--no-video", action="store_true", help="Skip MP4 rendering and write JSON evidence only.")
    args = parser.parse_args()

    result = run_benchmark(seed=args.seed, output_dir=args.output_dir, render_video=not args.no_video)
    metrics = result["metrics"]
    print(f"SUCCESS={metrics['success']}")
    print(f"CAP_ROTATION_DEG={metrics['cap_rotation_deg']}")
    print(f"MAX_SLIP_MM={metrics['max_slip_mm']}")
    print(f"OUTPUT_DIR={result['output_dir']}")
    print(f"VIDEO={result['video']}")
    return 0 if metrics["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
