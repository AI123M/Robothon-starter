import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

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
            self.assertGreaterEqual(policy_ablation["improvement"]["dexterity_score_delta"], 5.0)
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


if __name__ == "__main__":
    unittest.main()
