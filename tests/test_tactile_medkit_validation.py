import tempfile
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
            self.assertEqual(report["project_name"], "Tactile MedKit Manipulation Benchmark")
            self.assertGreaterEqual(report["metrics"]["cap_rotation_deg"], 220)


if __name__ == "__main__":
    unittest.main()
