from collections import defaultdict
from typing import Dict, Iterable, List

from .task_model import PHASES


def _number(sample: Dict, key: str, default: float = 0.0) -> float:
    value = sample.get(key, default)
    return float(value if value is not None else default)


def build_contact_timeline(samples: Iterable[Dict]) -> Dict[str, Dict]:
    grouped = defaultdict(lambda: {"samples": 0, "fingers": set(), "first_time": None, "last_time": None})
    for sample in samples:
        phase = sample.get("phase", "unknown")
        item = grouped[phase]
        item["samples"] += 1
        time_value = sample.get("time")
        if time_value is not None:
            time_value = float(time_value)
            if item["first_time"] is None:
                item["first_time"] = time_value
            item["last_time"] = time_value
        for finger in sample.get("contacts", []):
            item["fingers"].add(str(finger))

    return {
        phase: {
            "samples": data["samples"],
            "fingers": sorted(data["fingers"]),
            "first_time": data["first_time"],
            "last_time": data["last_time"],
        }
        for phase, data in grouped.items()
    }


def compute_run_metrics(samples: List[Dict], seed: int) -> Dict:
    if not samples:
        return {
            "seed": seed,
            "success": False,
            "phase_count": len(PHASES),
            "completed_phases": 0,
            "cap_rotation_deg": 0.0,
            "max_slip_mm": 999.0,
            "max_placement_error_mm": 999.0,
            "dexterity_score": 0.0,
            "contacts_per_phase": {},
        }

    expected_phases = {phase["id"] for phase in PHASES}
    seen_phases = {sample.get("phase") for sample in samples}
    contacts_per_phase = build_contact_timeline(samples)
    cap_rotation_deg = max(_number(sample, "cap_rotation_deg") for sample in samples)
    max_slip_mm = max(_number(sample, "slip_mm") for sample in samples)
    max_placement_error_mm = max(_number(sample, "placement_error_mm") for sample in samples)
    completed_phases = len(expected_phases & seen_phases)

    rotation_score = min(cap_rotation_deg / 220.0, 1.0) * 25.0
    slip_score = max(0.0, 1.0 - max_slip_mm / 2.0) * 20.0
    placement_score = max(0.0, 1.0 - max_placement_error_mm / 25.0) * 20.0
    phase_score = completed_phases / len(expected_phases) * 20.0
    contact_richness = max((len(data["fingers"]) for data in contacts_per_phase.values()), default=0)
    contact_score = min(contact_richness / 5.0, 1.0) * 15.0
    dexterity_score = round(rotation_score + slip_score + placement_score + phase_score + contact_score, 1)

    success = (
        completed_phases == len(expected_phases)
        and cap_rotation_deg >= 220.0
        and max_slip_mm <= 0.5
        and max_placement_error_mm <= 10.0
    )

    return {
        "seed": seed,
        "success": success,
        "phase_count": len(PHASES),
        "completed_phases": completed_phases,
        "cap_rotation_deg": round(cap_rotation_deg, 3),
        "max_slip_mm": round(max_slip_mm, 3),
        "max_placement_error_mm": round(max_placement_error_mm, 3),
        "dexterity_score": dexterity_score,
        "contacts_per_phase": contacts_per_phase,
    }


def summarize_stress_runs(runs: List[Dict]) -> Dict:
    total = len(runs)
    successes = sum(1 for run in runs if run.get("success"))
    worst_slip = max((_number(run, "max_slip_mm") for run in runs), default=0.0)
    worst_error = max((_number(run, "max_placement_error_mm") for run in runs), default=0.0)
    average_score = sum(_number(run, "dexterity_score") for run in runs) / total if total else 0.0

    return {
        "runs": total,
        "successes": successes,
        "success_rate": successes / total if total else 0.0,
        "worst_slip_mm": round(worst_slip, 3),
        "worst_placement_error_mm": round(worst_error, 3),
        "average_dexterity_score": round(average_score, 2),
    }
