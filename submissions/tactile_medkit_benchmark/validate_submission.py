import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from submissions.tactile_medkit_benchmark.task_model import (
    OUTPUTS_DIR,
    PACKAGE_DIR,
    PROJECT_NAME,
    REGISTRATION_UUID,
)

REQUIRED_ROOT_FILES = [
    "registration.json",
    "README.md",
    "JUDGE_BRIEF.md",
    "rubric_scorecard.json",
    "submission_manifest.json",
    "scene.xml",
    "run_demo.py",
    "run_stress_eval.py",
]

REQUIRED_OUTPUT_FILES = [
    "summary.json",
    "trajectory.json",
    "contact_timeline.json",
    "evidence_package.json",
    "stress_eval.json",
    "video_status.json",
    "final_report.txt",
]


def _read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _check_file(path: Path, errors: List[str]) -> None:
    if not path.exists():
        errors.append(f"missing file: {path}")
    elif path.is_file() and path.stat().st_size == 0:
        errors.append(f"empty file: {path}")


def validate_submission(output_dir: Optional[Path] = None, require_video: bool = True) -> Dict:
    output_dir = Path(output_dir or OUTPUTS_DIR)
    errors: List[str] = []

    for filename in REQUIRED_ROOT_FILES:
        _check_file(PACKAGE_DIR / filename, errors)
    for filename in REQUIRED_OUTPUT_FILES:
        _check_file(output_dir / filename, errors)
    if require_video:
        _check_file(output_dir / "demo.mp4", errors)

    registration = {}
    manifest = {}
    summary = {}
    stress = {}
    if (PACKAGE_DIR / "registration.json").exists():
        registration = _read_json(PACKAGE_DIR / "registration.json")
    if (PACKAGE_DIR / "submission_manifest.json").exists():
        manifest = _read_json(PACKAGE_DIR / "submission_manifest.json")
    if (output_dir / "summary.json").exists():
        summary = _read_json(output_dir / "summary.json")
    if (output_dir / "stress_eval.json").exists():
        stress = _read_json(output_dir / "stress_eval.json")

    uuid = registration.get("uuid")
    project_name = registration.get("project_name")
    if uuid != REGISTRATION_UUID:
        errors.append(f"registration uuid mismatch: {uuid!r}")
    if project_name != PROJECT_NAME:
        errors.append(f"project name mismatch: {project_name!r}")
    if manifest.get("registration_uuid") != REGISTRATION_UUID:
        errors.append("manifest registration_uuid does not match registration.json")
    if manifest.get("project_name") != PROJECT_NAME:
        errors.append("manifest project_name does not match registration.json")

    targets = manifest.get("score_targets", {})
    if summary:
        if summary.get("cap_rotation_deg", 0) < targets.get("cap_rotation_deg", 220.0):
            errors.append("cap rotation below target")
        if summary.get("max_slip_mm", 999) > targets.get("max_slip_mm", 0.5):
            errors.append("slip exceeds target")
        if summary.get("max_placement_error_mm", 999) > targets.get("max_placement_error_mm", 10.0):
            errors.append("placement error exceeds target")
        if not summary.get("success"):
            errors.append("summary success flag is false")

    stress_summary = stress.get("summary", {}) if stress else {}
    if stress_summary:
        if stress_summary.get("success_rate", 0.0) < targets.get("stress_success_rate", 0.875):
            errors.append("stress success rate below target")

    return {
        "valid": not errors,
        "errors": errors,
        "uuid": uuid,
        "project_name": project_name,
        "metrics": summary,
        "stress": stress_summary,
        "output_dir": str(output_dir),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Tactile MedKit submission artifacts.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR)
    parser.add_argument("--no-video", action="store_true", help="Do not require demo.mp4.")
    args = parser.parse_args()

    report = validate_submission(output_dir=args.output_dir, require_video=not args.no_video)
    if report["valid"]:
        print("VALID")
        print(f"UUID={report['uuid']}")
        print(f"PROJECT={report['project_name']}")
        return 0
    print("INVALID")
    for error in report["errors"]:
        print(f"- {error}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
