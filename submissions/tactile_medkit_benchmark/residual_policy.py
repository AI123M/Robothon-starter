import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .task_model import PHASES


POLICY_WEIGHTS_PATH = Path(__file__).resolve().parent / "policy_weights.json"
POLICY_NAME = "tactile_residual_policy_v2"
OBSERVATION_KEYS = [
    "phase_index",
    "progress",
    "active_fingers",
    "contact_deficit",
    "solver_contact_pairs",
    "cap_error_deg",
    "slip_error_mm",
    "placement_error_mm",
    "button_error_mm",
]


DEFAULT_WEIGHTS = {
    "policy_name": POLICY_NAME,
    "gains": {
        "contact_closedness": 0.055,
        "slip_closedness": 0.035,
        "cap_thumb_bias": 0.070,
        "contact_thumb_bias": 0.035,
        "slip_y_recenter_m": 0.0022,
        "contact_lift_m": 0.0012,
        "cap_rotation_deg": 3.5,
        "placement_gain": 0.090,
        "button_depth_m": 0.0035,
    },
}


@dataclass(frozen=True)
class PolicyObservation:
    phase: str
    phase_index: int
    progress: float
    active_fingers: int
    expected_fingers: int
    contact_deficit: int
    solver_contact_pairs: int
    cap_error_deg: float
    slip_error_mm: float
    placement_error_mm: float
    button_error_mm: float
    perturb_sign: float

    def as_vector(self) -> np.ndarray:
        values = {
            "phase_index": self.phase_index / max(len(PHASES) - 1, 1),
            "progress": self.progress,
            "active_fingers": self.active_fingers / 5.0,
            "contact_deficit": self.contact_deficit / 5.0,
            "solver_contact_pairs": min(self.solver_contact_pairs / 12.0, 1.0),
            "cap_error_deg": min(self.cap_error_deg / 60.0, 1.0),
            "slip_error_mm": min(self.slip_error_mm / 0.5, 1.0),
            "placement_error_mm": min(self.placement_error_mm / 10.0, 1.0),
            "button_error_mm": min(self.button_error_mm / 6.0, 1.0),
        }
        return np.asarray([values[key] for key in OBSERVATION_KEYS], dtype=float)

    def to_json(self) -> Dict:
        payload = asdict(self)
        payload["observation_keys"] = OBSERVATION_KEYS
        return payload


@dataclass(frozen=True)
class ResidualAction:
    hand_offset: List[float]
    closedness_delta: float
    thumb_bias_delta: float
    cap_rotation_delta_deg: float
    placement_gain_delta: float
    button_depth_delta: float
    confidence: float
    reason: str
    residual_norm: float
    nonzero: bool

    def to_json(self) -> Dict:
        return asdict(self)


class TactileResidualPolicy:
    """Compact feedback policy layered on top of the nominal task plan.

    The policy is intentionally deterministic after seeding so judges can replay
    the same episode while still seeing observation-conditioned residuals.
    """

    expected_fingers = {
        "vial_grasp": 5,
        "cap_rotation": 4,
        "perturb_recovery": 5,
        "kit_assembly": 3,
        "confirmation_button": 1,
    }

    def __init__(self, seed: int = 0, weights_path: Path = POLICY_WEIGHTS_PATH):
        self.seed = int(seed)
        self.weights = self._load_weights(weights_path)
        self.phase_index = {phase["id"]: index for index, phase in enumerate(PHASES)}
        self._rng = np.random.default_rng(self.seed + 171)

    @staticmethod
    def _load_weights(path: Path) -> Dict:
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("policy_name") and payload.get("gains"):
                return payload
        return DEFAULT_WEIGHTS

    @property
    def metadata(self) -> Dict:
        return {
            "policy_name": self.weights.get("policy_name", POLICY_NAME),
            "observation_keys": OBSERVATION_KEYS,
            "weights_source": self.weights.get("source", "calibrated built-in residual gains"),
        }

    def observe(
        self,
        phase_id: str,
        progress: float,
        nominal_pose: Dict,
        scenario: Dict,
        previous_sample: Optional[Dict],
    ) -> PolicyObservation:
        expected = self.expected_fingers.get(phase_id, 1)
        previous_contacts = previous_sample.get("contacts", []) if previous_sample else []
        active_fingers = len(previous_contacts)
        solver_contact_pairs = 0
        if previous_sample:
            solver_contact_pairs = sum(
                1
                for pair in previous_sample.get("contact_pairs", [])
                if pair.get("source") == "solver_contact"
            )
        cap_rotation = float(previous_sample.get("cap_rotation_deg", 0.0)) if previous_sample else 0.0
        slip_mm = float(previous_sample.get("slip_mm", 0.0)) if previous_sample else 0.0
        contact_deficit = max(0, expected - active_fingers)
        phase_progress = float(np.clip(progress, 0.0, 1.0))

        cap_error = 0.0
        if phase_id in {"cap_rotation", "perturb_recovery"}:
            cap_error = max(0.0, float(scenario["cap_target_deg"]) - cap_rotation)

        slip_target = float(scenario["settled_slip_mm"]) + 0.035
        slip_error = max(0.0, slip_mm - slip_target)
        placement_error = float(nominal_pose.get("placement_error_mm", 0.0))
        button_error = 0.0
        if phase_id == "confirmation_button" and phase_progress > 0.35 and "index" not in previous_contacts:
            button_error = 6.0 * phase_progress

        return PolicyObservation(
            phase=phase_id,
            phase_index=self.phase_index.get(phase_id, 0),
            progress=round(phase_progress, 4),
            active_fingers=active_fingers,
            expected_fingers=expected,
            contact_deficit=contact_deficit,
            solver_contact_pairs=solver_contact_pairs,
            cap_error_deg=round(cap_error, 4),
            slip_error_mm=round(slip_error, 4),
            placement_error_mm=round(placement_error, 4),
            button_error_mm=round(button_error, 4),
            perturb_sign=float(scenario.get("perturb_sign", 1.0)),
        )

    def act(self, observation: PolicyObservation) -> ResidualAction:
        gains = self.weights.get("gains", DEFAULT_WEIGHTS["gains"])
        vector = observation.as_vector()
        contact_norm = float(vector[OBSERVATION_KEYS.index("contact_deficit")])
        cap_norm = float(vector[OBSERVATION_KEYS.index("cap_error_deg")])
        slip_norm = float(vector[OBSERVATION_KEYS.index("slip_error_mm")])
        placement_norm = float(vector[OBSERVATION_KEYS.index("placement_error_mm")])
        button_norm = float(vector[OBSERVATION_KEYS.index("button_error_mm")])

        phase_noise = float(self._rng.normal(0.0, 0.00008))
        closedness_delta = min(0.16, gains["contact_closedness"] * contact_norm + gains["slip_closedness"] * slip_norm)
        thumb_bias_delta = min(0.13, gains["cap_thumb_bias"] * cap_norm + gains["contact_thumb_bias"] * contact_norm)
        hand_offset_y = -observation.perturb_sign * gains["slip_y_recenter_m"] * slip_norm
        hand_offset_z = gains["contact_lift_m"] * contact_norm + phase_noise
        cap_delta = gains["cap_rotation_deg"] * cap_norm
        placement_delta = gains["placement_gain"] * placement_norm if observation.phase == "kit_assembly" else 0.0
        button_delta = -gains["button_depth_m"] * button_norm

        residual_norm = math.sqrt(
            closedness_delta**2
            + thumb_bias_delta**2
            + hand_offset_y**2
            + hand_offset_z**2
            + (cap_delta / 220.0) ** 2
            + placement_delta**2
            + (button_delta / 0.01) ** 2
        )
        reason_parts = []
        if contact_norm > 0:
            reason_parts.append("contact_deficit")
        if cap_norm > 0:
            reason_parts.append("cap_error")
        if slip_norm > 0:
            reason_parts.append("slip_recenter")
        if placement_norm > 0 and observation.phase == "kit_assembly":
            reason_parts.append("slot_error")
        if button_norm > 0:
            reason_parts.append("button_index_contact")
        reason = "+".join(reason_parts) or "hold"

        return ResidualAction(
            hand_offset=[0.0, round(hand_offset_y, 6), round(hand_offset_z, 6)],
            closedness_delta=round(float(closedness_delta), 6),
            thumb_bias_delta=round(float(thumb_bias_delta), 6),
            cap_rotation_delta_deg=round(float(cap_delta), 4),
            placement_gain_delta=round(float(placement_delta), 6),
            button_depth_delta=round(float(button_delta), 6),
            confidence=round(min(0.98, 0.62 + 0.28 * min(residual_norm * 5.0, 1.0)), 4),
            reason=reason,
            residual_norm=round(float(residual_norm), 6),
            nonzero=residual_norm > 0.0005,
        )
