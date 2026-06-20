import json
import tempfile
import subprocess
import sys
import unittest
from pathlib import Path

from submissions.tactile_medkit_benchmark.simulation import run_benchmark, run_stress_eval
from submissions.tactile_medkit_benchmark.validate_submission import validate_submission


class TactileMedKitValidationTests(unittest.TestCase):
    def test_validate_submission_accepts_generated_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "outputs"
            run_benchmark(seed=11, output_dir=output_dir, render_video=False)
            run_stress_eval(seeds=3, output_dir=output_dir)

            report = validate_submission(output_dir=output_dir, require_video=False)

            self.assertTrue(report["valid"])
            self.assertEqual(report["uuid"], "f74c5b50-5ef8-467b-a141-a28ea9c34333")
            self.assertEqual(report["project_name"], "Reflex MedKit Dexterity Lab")
            self.assertGreaterEqual(report["metrics"]["cap_rotation_deg"], 220)

    def test_validate_submission_script_executes_from_repository_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "outputs"
            run_benchmark(seed=13, output_dir=output_dir, render_video=False)
            run_stress_eval(seeds=3, output_dir=output_dir)

            completed = subprocess.run(
                [
                    sys.executable,
                    "submissions/tactile_medkit_benchmark/validate_submission.py",
                    "--no-video",
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("VALID", completed.stdout)

    def test_validate_submission_rejects_missing_physical_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "outputs"
            run_benchmark(seed=17, output_dir=output_dir, render_video=False)
            run_stress_eval(seeds=3, output_dir=output_dir)
            summary_path = output_dir / "summary.json"
            summary = json.loads(summary_path.read_text())
            summary["control_mode"] = "scripted_qpos"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")

            report = validate_submission(output_dir=output_dir, require_video=False)

            self.assertFalse(report["valid"])
            self.assertTrue(any("control mode" in error for error in report["errors"]))

    def test_validate_submission_rejects_missing_policy_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "outputs"
            run_benchmark(seed=21, output_dir=output_dir, render_video=False)
            run_stress_eval(seeds=3, output_dir=output_dir)
            summary_path = output_dir / "summary.json"
            summary = json.loads(summary_path.read_text())
            summary["residual_policy_active"] = False
            summary["nonzero_policy_updates"] = 0
            summary["policy_observation_keys"] = []
            summary_path.write_text(json.dumps(summary), encoding="utf-8")

            report = validate_submission(output_dir=output_dir, require_video=False)

            self.assertFalse(report["valid"])
            self.assertTrue(any("residual policy" in error or "policy observation" in error for error in report["errors"]))

    def test_validate_submission_rejects_proxy_only_contact_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "outputs"
            run_benchmark(seed=18, output_dir=output_dir, render_video=False)
            run_stress_eval(seeds=3, output_dir=output_dir)
            summary_path = output_dir / "summary.json"
            summary = json.loads(summary_path.read_text())
            summary["contact_sources"] = ["site_distance"]
            summary["solver_contact_phases"] = 0
            summary["solver_contact_pairs"] = 0
            summary_path.write_text(json.dumps(summary), encoding="utf-8")

            report = validate_submission(output_dir=output_dir, require_video=False)

            self.assertFalse(report["valid"])
            self.assertTrue(any("solver contact" in error for error in report["errors"]))

    def test_validate_submission_rejects_missing_index_button_contact(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "outputs"
            run_benchmark(seed=20, output_dir=output_dir, render_video=False)
            run_stress_eval(seeds=3, output_dir=output_dir)
            summary_path = output_dir / "summary.json"
            summary = json.loads(summary_path.read_text())
            summary["contacts_per_phase"]["confirmation_button"]["fingers"] = ["little", "thumb"]
            summary_path.write_text(json.dumps(summary), encoding="utf-8")

            report = validate_submission(output_dir=output_dir, require_video=False)

            self.assertFalse(report["valid"])
            self.assertTrue(any("index button" in error for error in report["errors"]))

    def test_validate_submission_rejects_short_video_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "outputs"
            run_benchmark(seed=19, output_dir=output_dir, render_video=False)
            run_stress_eval(seeds=3, output_dir=output_dir)
            (output_dir / "demo.mp4").write_bytes(b"not a real mp4, only validator presence evidence")
            (output_dir / "video_status.json").write_text(
                json.dumps({"rendered": True, "path": "demo.mp4", "frames": 20, "fps": 3, "duration_sec": 6.667}),
                encoding="utf-8",
            )

            report = validate_submission(output_dir=output_dir, require_video=True)

            self.assertFalse(report["valid"])
            self.assertTrue(any("video duration" in error for error in report["errors"]))


if __name__ == "__main__":
    unittest.main()
