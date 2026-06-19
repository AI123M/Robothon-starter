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
            "control_mode": "unknown",
            "physics_steps": 0,
            "measured_contact_phases": 0,
            "contact_sources": [],
            "solver_contact_phases": 0,
            "solver_contact_pairs": 0,
            "max_solver_contact_distance_mm": None,
            "collision_enabled_geoms": 0,
            "residual_policy_active": False,
            "policy_name": "",
            "policy_updates": 0,
            "nonzero_policy_updates": 0,
            "policy_update_rate": 0.0,
            "max_policy_residual_norm": 0.0,
            "policy_observation_keys": [],
        }

    expected_phases = {phase["id"] for phase in PHASES}
    seen_phases = {sample.get("phase") for sample in samples}
    contacts_per_phase = build_contact_timeline(samples)
    cap_rotation_deg = max(_number(sample, "cap_rotation_deg") for sample in samples)
    max_slip_mm = max(_number(sample, "slip_mm") for sample in samples)
    max_placement_error_mm = max(_number(sample, "placement_error_mm") for sample in samples)
    completed_phases = len(expected_phases & seen_phases)
    measured_contact_phases = sum(1 for data in contacts_per_phase.values() if data["fingers"])
    contact_sources = sorted(
        {
            source
            for sample in samples
            for source in sample.get("contact_sources", [])
        }
    )
    physics_steps = sum(int(sample.get("physics_steps", 0)) for sample in samples)
    control_modes = sorted({sample.get("control_mode", "unknown") for sample in samples})
    solver_contact_phases = {
        sample.get("phase")
        for sample in samples
        if any(pair.get("source") == "solver_contact" for pair in sample.get("contact_pairs", []))
    }
    solver_distances = [
        float(pair["distance_mm"])
        for sample in samples
        for pair in sample.get("contact_pairs", [])
        if pair.get("source") == "solver_contact" and pair.get("distance_mm") is not None
    ]
    solver_contact_pairs = len(solver_distances)
    collision_enabled_geoms = max((int(sample.get("collision_enabled_geoms", 0)) for sample in samples), default=0)
    policy_samples = [sample for sample in samples if sample.get("policy_residual")]
    policy_updates = len(policy_samples)
    nonzero_policy_updates = sum(1 for sample in policy_samples if sample.get("policy_residual", {}).get("nonzero"))
    policy_residual_norms = [
        _number(sample.get("policy_residual", {}), "residual_norm")
        for sample in policy_samples
    ]
    policy_names = sorted({sample.get("policy_name", "") for sample in samples if sample.get("policy_name")})
    observation_keys = sorted(
        {
            key
            for sample in samples
            for key in sample.get("policy_observation", {}).get("observation_keys", [])
        }
    )

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
        "control_mode": control_modes[0] if len(control_modes) == 1 else ",".join(control_modes),
        "physics_steps": physics_steps,
        "measured_contact_phases": measured_contact_phases,
        "contact_sources": contact_sources,
        "solver_contact_phases": len(solver_contact_phases),
        "solver_contact_pairs": solver_contact_pairs,
        "max_solver_contact_distance_mm": round(max(solver_distances), 3) if solver_distances else None,
        "collision_enabled_geoms": collision_enabled_geoms,
        "residual_policy_active": nonzero_policy_updates > 0,
        "policy_name": policy_names[0] if len(policy_names) == 1 else ",".join(policy_names),
        "policy_updates": policy_updates,
        "nonzero_policy_updates": nonzero_policy_updates,
        "policy_update_rate": round(nonzero_policy_updates / policy_updates, 3) if policy_updates else 0.0,
        "max_policy_residual_norm": round(max(policy_residual_norms), 6) if policy_residual_norms else 0.0,
        "policy_observation_keys": observation_keys,
    }


def summarize_stress_runs(runs: List[Dict]) -> Dict:
    total = len(runs)
    successes = sum(1 for run in runs if run.get("success"))
    worst_slip = max((_number(run, "max_slip_mm") for run in runs), default=0.0)
    worst_error = max((_number(run, "max_placement_error_mm") for run in runs), default=0.0)
    average_score = sum(_number(run, "dexterity_score") for run in runs) / total if total else 0.0
    average_policy_update_rate = sum(_number(run, "policy_update_rate") for run in runs) / total if total else 0.0
    def variance(key: str) -> float:
        if total < 2:
            return 0.0
        values = [_number(run, key) for run in runs]
        mean = sum(values) / total
        return sum((value - mean) ** 2 for value in values) / total

    return {
        "runs": total,
        "successes": successes,
        "success_rate": successes / total if total else 0.0,
        "worst_slip_mm": round(worst_slip, 3),
        "worst_placement_error_mm": round(worst_error, 3),
        "average_dexterity_score": round(average_score, 2),
        "average_policy_update_rate": round(average_policy_update_rate, 3),
        "min_nonzero_policy_updates": min((int(run.get("nonzero_policy_updates", 0)) for run in runs), default=0),
        "metric_variance": {
            "cap_rotation_deg": round(variance("cap_rotation_deg"), 6),
            "max_slip_mm": round(variance("max_slip_mm"), 6),
            "max_placement_error_mm": round(variance("max_placement_error_mm"), 6),
            "dexterity_score": round(variance("dexterity_score"), 6),
        },
    }
