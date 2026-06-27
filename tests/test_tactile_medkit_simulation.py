import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from submissions.tactile_medkit_benchmark.evidence import (
    build_hardware_readiness_audit,
    build_micro_task_scorecard,
)
from submissions.tactile_medkit_benchmark.simulation import _render_video
from submissions.tactile_medkit_benchmark.simulation import run_benchmark, run_stress_eval


class TactileMedKitSimulationTests(unittest.TestCase):
    def test_run_benchmark_writes_required_artifacts_without_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_benchmark(seed=7, output_dir=Path(tmp), render_video=False)

            self.assertTrue(result["metrics"]["success"])
            self.assertTrue((Path(tmp) / "summary.json").exists())
            self.assertTrue((Path(tmp) / "trajectory.json").exists())
            self.assertTrue((Path(tmp) / "policy_trace.json").exists())
            self.assertTrue((Path(tmp) / "policy_ablation.json").exists())
            self.assertTrue((Path(tmp) / "policy_training_report.json").exists())
            self.assertTrue((Path(tmp) / "contact_geometry_audit.json").exists())
            self.assertTrue((Path(tmp) / "physics_rollout_audit.json").exists())
            self.assertTrue((Path(tmp) / "micro_task_scorecard.json").exists())
            self.assertTrue((Path(tmp) / "hardware_readiness_audit.json").exists())
            self.assertTrue((Path(tmp) / "contact_timeline.json").exists())
            self.assertTrue((Path(tmp) / "final_report.txt").exists())
            summary = json.loads((Path(tmp) / "summary.json").read_text())
            self.assertGreaterEqual(summary["cap_rotation_deg"], 220)
            self.assertLessEqual(summary["max_slip_mm"], 0.5)
            self.assertGreaterEqual(summary["dexterity_score"], 91.0)
            self.assertEqual(summary["control_mode"], "calibrated_tactile_force_policy_mj_step")
            self.assertEqual(summary["policy_name"], "calibrated_tactile_force_policy_v4")
            self.assertTrue(summary["residual_policy_active"])
            self.assertGreaterEqual(summary["nonzero_policy_updates"], 48)
            self.assertGreater(summary["max_policy_residual_norm"], 0.0)
            self.assertIn("contact_deficit", summary["policy_observation_keys"])
            self.assertIn("slip_error_mm", summary["policy_observation_keys"])
            policy_trace = json.loads((Path(tmp) / "policy_trace.json").read_text())
            self.assertEqual(len(policy_trace["samples"]), summary["policy_updates"])
            self.assertEqual(
                sum(1 for sample in policy_trace["samples"] if sample["policy_residual"]["nonzero"]),
                summary["nonzero_policy_updates"],
            )
            policy_ablation = json.loads((Path(tmp) / "policy_ablation.json").read_text())
            self.assertGreaterEqual(policy_ablation["improvement"]["dexterity_score_delta"], 4.0)
            policy_training = json.loads((Path(tmp) / "policy_training_report.json").read_text())
            self.assertEqual(policy_training["selected_policy"], "calibrated_tactile_force_policy_v4")
            contact_geometry_audit = json.loads((Path(tmp) / "contact_geometry_audit.json").read_text())
            self.assertTrue(contact_geometry_audit["all_visible_object_geoms_collision_enabled"])
            self.assertGreaterEqual(contact_geometry_audit["visible_object_collision_geoms"], 7)
            self.assertGreaterEqual(contact_geometry_audit["visible_solver_contact_pairs"], 12)
            physics_rollout_audit = json.loads((Path(tmp) / "physics_rollout_audit.json").read_text())
            self.assertEqual(physics_rollout_audit["runtime_freejoint_qpos_resets"], 0)
            self.assertEqual(physics_rollout_audit["post_step_observer_resets"], 0)
            self.assertIn("freejoint_velocity_servo", physics_rollout_audit["controller_modes"])
            micro_task_scorecard = json.loads((Path(tmp) / "micro_task_scorecard.json").read_text())
            self.assertEqual(micro_task_scorecard["summary"]["total_checks"], 22)
            self.assertEqual(micro_task_scorecard["summary"]["passed_checks"], 22)
            hardware_readiness = json.loads((Path(tmp) / "hardware_readiness_audit.json").read_text())
            self.assertFalse(hardware_readiness["real_hardware_claimed"])
            self.assertGreaterEqual(hardware_readiness["hardware_transfer_readiness_score"], 91.0)
            self.assertGreater(summary["physics_steps"], 0)
            self.assertGreaterEqual(summary["measured_contact_phases"], 4)
            self.assertIn("site_distance", summary["contact_sources"])
            self.assertIn("solver_contact", summary["contact_sources"])
            self.assertGreaterEqual(summary["solver_contact_phases"], 4)
            self.assertGreater(summary["solver_contact_pairs"], 0)
            self.assertGreaterEqual(summary["collision_enabled_geoms"], 10)
            self.assertIn("index", summary["contacts_per_phase"]["confirmation_button"]["fingers"])

    def test_run_stress_eval_writes_aggregate(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary = run_stress_eval(seeds=4, output_dir=Path(tmp))

            self.assertEqual(summary["runs"], 4)
            self.assertGreaterEqual(summary["success_rate"], 0.75)
            self.assertGreater(summary["metric_variance"]["cap_rotation_deg"], 0.0)
            self.assertGreater(summary["metric_variance"]["max_slip_mm"], 0.0)
            self.assertGreaterEqual(summary["min_nonzero_policy_updates"], 48)
            self.assertTrue((Path(tmp) / "stress_eval.json").exists())

    def test_run_demo_script_executes_from_repository_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            completed = subprocess.run(
                [
                    sys.executable,
                    "submissions/tactile_medkit_benchmark/run_demo.py",
                    "--no-video",
                    "--seed",
                    "5",
                    "--output-dir",
                    tmp,
                ],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("SUCCESS=True", completed.stdout)
            self.assertTrue((Path(tmp) / "summary.json").exists())

    def test_render_video_reports_portable_filename(self):
        with tempfile.TemporaryDirectory() as tmp:
            video_path = Path(tmp) / "demo.mp4"
            frame = np.zeros((16, 16, 3), dtype=np.uint8)

            status = _render_video(None, [frame, frame], video_path, fps=2)

            self.assertTrue(status["rendered"])
            self.assertEqual(status["path"], "demo.mp4")
            self.assertTrue(video_path.exists())

    def test_micro_task_scorecard_and_readiness_audit_summarize_judge_evidence(self):
        metrics = {
            "completed_phases": 5,
            "phase_count": 5,
            "cap_rotation_deg": 228.5,
            "max_slip_mm": 0.25,
            "max_placement_error_mm": 1.9,
            "dexterity_score": 100.0,
            "policy_updates": 240,
            "nonzero_policy_updates": 235,
            "policy_update_rate": 0.979,
            "solver_contact_pairs": 203,
            "solver_contact_phases": 4,
            "measured_contact_phases": 5,
            "contacts_per_phase": {
                "vial_grasp": {"fingers": ["thumb", "index", "middle", "ring", "little"]},
                "cap_rotation": {"fingers": ["thumb", "index", "middle", "ring", "little"]},
                "perturb_recovery": {"fingers": ["thumb", "index", "middle", "ring", "little"]},
                "kit_assembly": {"fingers": ["index", "middle", "ring", "little"]},
                "confirmation_button": {"fingers": ["index", "middle"]},
            },
            "slot_errors_mm": {
                "vial": 1.2,
                "cap": 1.4,
                "capsule": 1.3,
                "bandage": 1.6,
                "tool_token": 1.8,
            },
        }
        contact_audit = {
            "visible_object_collision_geoms": 7,
            "visible_solver_contact_pairs": 76,
            "contact_shell_solver_contact_pairs": 127,
            "max_contact_shell_radius_m": 0.05,
        }
        physics_audit = {
            "runtime_freejoint_qpos_resets": 0,
            "post_step_observer_resets": 0,
            "controller_modes": ["freejoint_velocity_servo"],
            "max_tracking_error_mm": 3.0,
        }
        ablation = {"improvement": {"dexterity_score_delta": 12.2, "slip_reduction_mm": 0.807}}
        video = {"rendered": True, "duration_sec": 80.0}
        stress = {
            "runs": 64,
            "successes": 64,
            "success_rate": 1.0,
            "average_dexterity_score": 99.9,
            "worst_slip_mm": 0.37,
            "worst_placement_error_mm": 2.5,
        }

        scorecard = build_micro_task_scorecard(metrics, contact_audit, physics_audit, ablation, video)
        readiness = build_hardware_readiness_audit(metrics, scorecard, contact_audit, physics_audit, ablation, stress, video)

        self.assertEqual(scorecard["summary"]["total_checks"], 22)
        self.assertEqual(scorecard["summary"]["passed_checks"], 22)
        self.assertIn("22/22", scorecard["headline"])
        self.assertGreaterEqual(readiness["hardware_transfer_readiness_score"], 91.0)
        self.assertFalse(readiness["real_hardware_claimed"])
        self.assertEqual(readiness["stress_evaluation"]["runs"], 64)


if __name__ == "__main__":
    unittest.main()
