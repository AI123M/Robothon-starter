import argparse
import json
import subprocess
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
    "HARDWARE_TRANSFER_PROTOCOL.md",
    "rubric_scorecard.json",
    "submission_manifest.json",
    "policy_weights.json",
    "scene.xml",
    "run_demo.py",
    "run_stress_eval.py",
]

REQUIRED_OUTPUT_FILES = [
    "summary.json",
    "trajectory.json",
    "policy_trace.json",
    "policy_ablation.json",
    "policy_training_report.json",
    "contact_geometry_audit.json",
    "physics_rollout_audit.json",
    "micro_task_scorecard.json",
    "hardware_readiness_audit.json",
    "hardware_transfer_protocol.json",
    "contact_timeline.json",
    "evidence_package.json",
    "stress_eval.json",
    "video_status.json",
    "final_report.txt",
]

EXPECTED_CONTROL_MODE = "calibrated_tactile_force_policy_mj_step"
EXPECTED_POLICY_NAME = "calibrated_tactile_force_policy_v4"
MIN_DEXTERITY_SCORE = 91.0
MIN_NONZERO_POLICY_UPDATES = 48
MIN_POLICY_TRACE_SAMPLES = 200
MIN_POLICY_ABLATION_SCORE_DELTA = 4.0
MIN_VISIBLE_OBJECT_COLLISION_GEOMS = 6
MIN_VISIBLE_SOLVER_CONTACT_PAIRS = 12
MAX_CONTACT_SHELL_RADIUS_M = 0.055
REQUIRED_POLICY_OBSERVATION_KEYS = {
    "active_fingers",
    "button_error_mm",
    "cap_error_deg",
    "contact_deficit",
    "placement_error_mm",
    "slip_error_mm",
}
MIN_MEASURED_CONTACT_PHASES = 4
MIN_SOLVER_CONTACT_PHASES = 4
MIN_COLLISION_ENABLED_GEOMS = 10
MIN_VIDEO_DURATION_SEC = 60.0
MIN_HARDWARE_PROTOCOL_STAGES = 6
REQUIRED_HARDWARE_TELEMETRY_FIELDS = {
    "joint_position_rad",
    "joint_velocity_rad_s",
    "fingertip_contact_n",
    "emergency_stop_state",
}


def _read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _check_file(path: Path, errors: List[str]) -> None:
    if not path.exists():
        errors.append(f"missing file: {path}")
    elif path.is_file() and path.stat().st_size == 0:
        errors.append(f"empty file: {path}")


def _probe_video_duration(video_path: Path) -> Optional[float]:
    try:
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError:
        return None
    if completed.returncode != 0:
        return None
    try:
        return float(completed.stdout.strip())
    except ValueError:
        return None


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
    video_status = {}
    policy_trace = {}
    policy_ablation = {}
    policy_training_report = {}
    contact_geometry_audit = {}
    physics_rollout_audit = {}
    micro_task_scorecard = {}
    hardware_readiness_audit = {}
    hardware_transfer_protocol = {}
    if (PACKAGE_DIR / "registration.json").exists():
        registration = _read_json(PACKAGE_DIR / "registration.json")
    if (PACKAGE_DIR / "submission_manifest.json").exists():
        manifest = _read_json(PACKAGE_DIR / "submission_manifest.json")
    if (output_dir / "summary.json").exists():
        summary = _read_json(output_dir / "summary.json")
    if (output_dir / "stress_eval.json").exists():
        stress = _read_json(output_dir / "stress_eval.json")
    if (output_dir / "video_status.json").exists():
        video_status = _read_json(output_dir / "video_status.json")
    if (output_dir / "policy_trace.json").exists():
        policy_trace = _read_json(output_dir / "policy_trace.json")
    if (output_dir / "policy_ablation.json").exists():
        policy_ablation = _read_json(output_dir / "policy_ablation.json")
    if (output_dir / "policy_training_report.json").exists():
        policy_training_report = _read_json(output_dir / "policy_training_report.json")
    if (output_dir / "contact_geometry_audit.json").exists():
        contact_geometry_audit = _read_json(output_dir / "contact_geometry_audit.json")
    if (output_dir / "physics_rollout_audit.json").exists():
        physics_rollout_audit = _read_json(output_dir / "physics_rollout_audit.json")
    if (output_dir / "micro_task_scorecard.json").exists():
        micro_task_scorecard = _read_json(output_dir / "micro_task_scorecard.json")
    if (output_dir / "hardware_readiness_audit.json").exists():
        hardware_readiness_audit = _read_json(output_dir / "hardware_readiness_audit.json")
    if (output_dir / "hardware_transfer_protocol.json").exists():
        hardware_transfer_protocol = _read_json(output_dir / "hardware_transfer_protocol.json")

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
        if summary.get("control_mode") != EXPECTED_CONTROL_MODE:
            errors.append("control mode does not use calibrated tactile force policy")
        if summary.get("policy_name") != EXPECTED_POLICY_NAME:
            errors.append("force policy name is missing or unexpected")
        if float(summary.get("dexterity_score", 0.0)) < targets.get("min_dexterity_score", MIN_DEXTERITY_SCORE):
            errors.append("dexterity score below 91+ target")
        if not summary.get("residual_policy_active"):
            errors.append("force policy active flag is false")
        if int(summary.get("nonzero_policy_updates", 0)) < targets.get(
            "min_nonzero_policy_updates", MIN_NONZERO_POLICY_UPDATES
        ):
            errors.append("nonzero force policy updates below target")
        if float(summary.get("max_policy_residual_norm", 0.0)) <= 0.0:
            errors.append("policy residual norm evidence is missing")
        missing_policy_keys = REQUIRED_POLICY_OBSERVATION_KEYS - set(summary.get("policy_observation_keys", []))
        if missing_policy_keys:
            errors.append(f"policy observation keys missing: {sorted(missing_policy_keys)}")
        trace_samples = policy_trace.get("samples", []) if policy_trace else []
        if len(trace_samples) < targets.get("min_policy_trace_samples", MIN_POLICY_TRACE_SAMPLES):
            errors.append("policy trace sample count below target")
        nonzero_trace_updates = sum(1 for sample in trace_samples if sample.get("policy_residual", {}).get("nonzero"))
        if trace_samples and nonzero_trace_updates != int(summary.get("nonzero_policy_updates", -1)):
            errors.append("policy trace nonzero update count does not match summary")
        ablation_delta = policy_ablation.get("improvement", {}).get("dexterity_score_delta") if policy_ablation else None
        if ablation_delta is None or float(ablation_delta) < targets.get(
            "min_policy_ablation_score_delta", MIN_POLICY_ABLATION_SCORE_DELTA
        ):
            errors.append("policy ablation score delta below target")
        if policy_training_report.get("selected_policy") != EXPECTED_POLICY_NAME:
            errors.append("policy training report does not identify the selected v4 policy")
        report_delta = policy_training_report.get("ablation_validation", {}).get("dexterity_score_delta")
        if report_delta is None or float(report_delta) < targets.get(
            "min_policy_ablation_score_delta", MIN_POLICY_ABLATION_SCORE_DELTA
        ):
            errors.append("policy training report ablation evidence below target")
        if int(summary.get("physics_steps", 0)) <= 0:
            errors.append("physics_steps must be positive")
        if int(summary.get("measured_contact_phases", 0)) < MIN_MEASURED_CONTACT_PHASES:
            errors.append("measured contact phases below target")
        if "site_distance" not in summary.get("contact_sources", []):
            errors.append("site-distance contact evidence is missing")
        if "solver_contact" not in summary.get("contact_sources", []):
            errors.append("solver contact evidence is missing")
        if int(summary.get("solver_contact_phases", 0)) < targets.get("min_solver_contact_phases", MIN_SOLVER_CONTACT_PHASES):
            errors.append("solver contact phases below target")
        if int(summary.get("solver_contact_pairs", 0)) <= 0:
            errors.append("solver contact pair count is zero")
        if int(summary.get("collision_enabled_geoms", 0)) < targets.get("min_collision_enabled_geoms", MIN_COLLISION_ENABLED_GEOMS):
            errors.append("collision-enabled geom count below target")
        if not contact_geometry_audit.get("all_visible_object_geoms_collision_enabled"):
            errors.append("visible object geoms are not all collision-enabled")
        if int(contact_geometry_audit.get("visible_object_collision_geoms", 0)) < targets.get(
            "min_visible_object_collision_geoms", MIN_VISIBLE_OBJECT_COLLISION_GEOMS
        ):
            errors.append("visible object collision geom count below target")
        if int(contact_geometry_audit.get("visible_solver_contact_pairs", 0)) < targets.get(
            "min_visible_solver_contact_pairs", MIN_VISIBLE_SOLVER_CONTACT_PAIRS
        ):
            errors.append("visible solver contact pair count below target")
        if float(contact_geometry_audit.get("max_contact_shell_radius_m", 999.0)) > targets.get(
            "max_contact_shell_radius_m", MAX_CONTACT_SHELL_RADIUS_M
        ):
            errors.append("contact shell radius exceeds physical-evidence target")
        if int(physics_rollout_audit.get("runtime_freejoint_qpos_resets", -1)) != 0:
            errors.append("runtime free-joint qpos reset audit is not zero")
        if int(physics_rollout_audit.get("post_step_observer_resets", -1)) != 0:
            errors.append("post-step observer reset audit is not zero")
        if "freejoint_velocity_servo" not in physics_rollout_audit.get("controller_modes", []):
            errors.append("physics rollout audit does not report the velocity-servo controller")
        confirmation_contacts = (
            summary.get("contacts_per_phase", {})
            .get("confirmation_button", {})
            .get("fingers", [])
        )
        if "index" not in confirmation_contacts:
            errors.append("index button contact evidence is missing")
        micro_summary = micro_task_scorecard.get("summary", {}) if micro_task_scorecard else {}
        if int(micro_summary.get("total_checks", 0)) < 22:
            errors.append("micro-task scorecard has fewer than 22 checks")
        if int(micro_summary.get("passed_checks", 0)) < 22:
            errors.append("micro-task scorecard does not pass all 22 checks")
        if float(hardware_readiness_audit.get("hardware_transfer_readiness_score", 0.0)) < 91.0:
            errors.append("hardware-transfer readiness score below 91")
        if hardware_readiness_audit.get("real_hardware_claimed") is not False:
            errors.append("hardware readiness audit must be explicit that no real hardware is claimed")
        if hardware_transfer_protocol.get("real_hardware_claimed") is not False:
            errors.append("hardware transfer protocol must be explicit that no real hardware is claimed")
        protocol_stages = hardware_transfer_protocol.get("bench_test_protocol", [])
        if len(protocol_stages) < MIN_HARDWARE_PROTOCOL_STAGES:
            errors.append("hardware transfer protocol has fewer than six bench-test stages")
        telemetry_fields = {item.get("field") for item in hardware_transfer_protocol.get("telemetry_schema", [])}
        missing_hardware_fields = REQUIRED_HARDWARE_TELEMETRY_FIELDS - telemetry_fields
        if missing_hardware_fields:
            errors.append(f"hardware transfer protocol telemetry fields missing: {sorted(missing_hardware_fields)}")
        criteria = hardware_transfer_protocol.get("acceptance_criteria", {})
        if float(criteria.get("cap_rotation_deg_min", 0.0)) < targets.get("cap_rotation_deg", 220.0):
            errors.append("hardware transfer protocol cap-rotation acceptance is below target")
        if float(criteria.get("max_slip_mm_max", 999.0)) > targets.get("max_slip_mm", 0.5):
            errors.append("hardware transfer protocol slip acceptance is looser than target")
        if float(criteria.get("max_placement_error_mm_max", 999.0)) > targets.get("max_placement_error_mm", 10.0):
            errors.append("hardware transfer protocol placement acceptance is looser than target")
        if int(criteria.get("micro_task_checks_min", 0)) < targets.get("min_micro_task_checks", 22):
            errors.append("hardware transfer protocol micro-task acceptance is below target")
        if criteria.get("operator_abort_false") is not True:
            errors.append("hardware transfer protocol must require operator_abort_false")
        actuator_schema = hardware_transfer_protocol.get("actuator_command_schema", {})
        if actuator_schema.get("runtime_pose_teleports_allowed") is not False:
            errors.append("hardware transfer protocol must forbid runtime pose teleports")
        if len(hardware_transfer_protocol.get("safety_interlocks", [])) < 4:
            errors.append("hardware transfer protocol must include safety interlocks")

    stress_summary = stress.get("summary", {}) if stress else {}
    if stress_summary:
        if stress_summary.get("success_rate", 0.0) < targets.get("stress_success_rate", 0.875):
            errors.append("stress success rate below target")
        variance = stress_summary.get("metric_variance", {})
        if variance.get("cap_rotation_deg", 0.0) <= 0.0:
            errors.append("stress cap_rotation_deg variance is missing")
        if variance.get("max_slip_mm", 0.0) <= 0.0:
            errors.append("stress max_slip_mm variance is missing")

    if require_video:
        if not video_status.get("rendered"):
            errors.append("video_status rendered flag is false")
        if float(video_status.get("duration_sec", 0.0)) < MIN_VIDEO_DURATION_SEC:
            errors.append("video duration below 60 seconds")
        probed_duration = _probe_video_duration(output_dir / "demo.mp4")
        if probed_duration is not None and probed_duration < MIN_VIDEO_DURATION_SEC:
            errors.append("ffprobe video duration below 60 seconds")

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
    parser = argparse.ArgumentParser(description="Validate Reflex MedKit submission artifacts.")
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
