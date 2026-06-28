from pathlib import Path

PROJECT_NAME = "Reflex MedKit Dexterity Lab"
PARTICIPANT_NAME = "AIFF"
REGISTRATION_UUID = "f74c5b50-5ef8-467b-a141-a28ea9c34333"
PACKAGE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = PACKAGE_DIR / "outputs"

PHASES = [
    {
        "id": "vial_grasp",
        "label": "Five-finger vial grasp",
        "target": "Opposed thumb, index, middle, ring, and little-finger contact on vial.",
    },
    {
        "id": "cap_rotation",
        "label": "Cap rotation",
        "target": "Rotate the vial cap beyond 220 degrees with stable fingertip contact.",
    },
    {
        "id": "perturb_recovery",
        "label": "Slip recovery",
        "target": "Recover from a lateral disturbance with less than 0.5 mm peak slip.",
    },
    {
        "id": "kit_assembly",
        "label": "Emergency-kit assembly",
        "target": "Place vial, capsule, bandage, and tool token into labeled kit slots.",
    },
    {
        "id": "confirmation_button",
        "label": "Final confirmation",
        "target": "Use the index fingertip to press the kit confirmation button.",
    },
]

TASK_OBJECTS = {
    "vial": {"slot": "slot_a", "target_error_mm": 5.5},
    "capsule": {"slot": "slot_b", "target_error_mm": 5.5},
    "bandage": {"slot": "slot_c", "target_error_mm": 5.9},
    "tool_token": {"slot": "slot_d", "target_error_mm": 5.5},
}

ARTIFACT_NAMES = [
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
    "final_report.txt",
]
