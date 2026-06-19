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
            self.assertTrue((Path(tmp) / "contact_timeline.json").exists())
            self.assertTrue((Path(tmp) / "final_report.txt").exists())
            summary = json.loads((Path(tmp) / "summary.json").read_text())
            self.assertGreaterEqual(summary["cap_rotation_deg"], 220)
            self.assertLessEqual(summary["max_slip_mm"], 0.5)
            self.assertEqual(summary["control_mode"], "actuator_position_mj_step")
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
