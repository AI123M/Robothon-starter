# Tactile MedKit Manipulation Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a runnable MuJoCo five-finger emergency-kit manipulation benchmark, validate it locally, and prepare it for judge-style review without opening the official PR.

**Architecture:** The submission lives entirely under `submissions/tactile_medkit_benchmark/` with focused Python modules for simulation, metrics, validation, and judge evidence. The simulator uses a deterministic MuJoCo scene and phase controller to produce demo artifacts, telemetry, and scorecard evidence that map directly to the official rubric.

**Tech Stack:** Python 3, MuJoCo, NumPy, ImageIO, standard-library `unittest`, JSON artifacts, MJCF XML.

---

## File Structure

- Create `submissions/tactile_medkit_benchmark/scene.xml`: MJCF workcell with five-finger hand, emergency-kit objects, contacts, cameras, joints, actuators, and sensors.
- Create `submissions/tactile_medkit_benchmark/__init__.py`: package marker and project constants.
- Create `submissions/tactile_medkit_benchmark/task_model.py`: phase definitions, object targets, and artifact paths.
- Create `submissions/tactile_medkit_benchmark/metrics.py`: deterministic metrics and scorecard helpers.
- Create `submissions/tactile_medkit_benchmark/simulation.py`: MuJoCo loading, controller phase loop, telemetry, and optional video rendering.
- Create `submissions/tactile_medkit_benchmark/run_demo.py`: CLI entrypoint for the main run.
- Create `submissions/tactile_medkit_benchmark/run_stress_eval.py`: CLI entrypoint for multi-seed robustness evaluation.
- Create `submissions/tactile_medkit_benchmark/validate_submission.py`: required-file and metric validation.
- Create `submissions/tactile_medkit_benchmark/README.md`: judge-facing project summary and run instructions.
- Create `submissions/tactile_medkit_benchmark/registration.json`: UUID metadata.
- Create `submissions/tactile_medkit_benchmark/JUDGE_BRIEF.md`: concise rubric alignment.
- Create `submissions/tactile_medkit_benchmark/rubric_scorecard.json`: declared target evidence.
- Create `submissions/tactile_medkit_benchmark/submission_manifest.json`: expected artifacts and commands.
- Create tests under `tests/test_tactile_medkit_metrics.py`, `tests/test_tactile_medkit_simulation.py`, and `tests/test_tactile_medkit_validation.py`.

## Task 1: Metrics Contract

**Files:**
- Create: `tests/test_tactile_medkit_metrics.py`
- Create: `submissions/tactile_medkit_benchmark/metrics.py`
- Create: `submissions/tactile_medkit_benchmark/task_model.py`

- [ ] **Step 1: Write the failing metric tests**

```python
import unittest

from submissions.tactile_medkit_benchmark.metrics import (
    build_contact_timeline,
    compute_run_metrics,
    summarize_stress_runs,
)
from submissions.tactile_medkit_benchmark.task_model import PHASES


class TactileMedKitMetricsTests(unittest.TestCase):
    def test_compute_run_metrics_reports_high_score_evidence(self):
        samples = [
            {"phase": "vial_grasp", "cap_rotation_deg": 0, "slip_mm": 0.0, "placement_error_mm": 0.0, "contacts": ["thumb", "index", "middle"]},
            {"phase": "cap_rotation", "cap_rotation_deg": 226.0, "slip_mm": 0.2, "placement_error_mm": 0.0, "contacts": ["thumb", "index", "middle", "ring"]},
            {"phase": "perturb_recovery", "cap_rotation_deg": 226.0, "slip_mm": 0.34, "placement_error_mm": 0.0, "contacts": ["thumb", "index", "middle", "ring", "little"]},
            {"phase": "kit_assembly", "cap_rotation_deg": 226.0, "slip_mm": 0.28, "placement_error_mm": 7.5, "contacts": ["thumb", "index", "middle"]},
            {"phase": "confirmation_button", "cap_rotation_deg": 226.0, "slip_mm": 0.2, "placement_error_mm": 6.0, "contacts": ["index"]},
        ]

        metrics = compute_run_metrics(samples, seed=42)

        self.assertEqual(metrics["seed"], 42)
        self.assertEqual(metrics["phase_count"], len(PHASES))
        self.assertGreaterEqual(metrics["cap_rotation_deg"], 220.0)
        self.assertLessEqual(metrics["max_slip_mm"], 0.5)
        self.assertLessEqual(metrics["max_placement_error_mm"], 10.0)
        self.assertGreaterEqual(metrics["dexterity_score"], 90.0)
        self.assertTrue(metrics["success"])

    def test_contact_timeline_groups_contacts_by_phase(self):
        samples = [
            {"time": 0.0, "phase": "vial_grasp", "contacts": ["thumb", "index"]},
            {"time": 0.1, "phase": "vial_grasp", "contacts": ["middle"]},
            {"time": 0.2, "phase": "kit_assembly", "contacts": ["ring"]},
        ]

        timeline = build_contact_timeline(samples)

        self.assertEqual(timeline["vial_grasp"]["samples"], 2)
        self.assertEqual(timeline["vial_grasp"]["fingers"], ["index", "middle", "thumb"])
        self.assertEqual(timeline["kit_assembly"]["fingers"], ["ring"])

    def test_stress_summary_counts_success_rate_and_worst_case(self):
        runs = [
            {"success": True, "max_slip_mm": 0.3, "max_placement_error_mm": 8.0},
            {"success": True, "max_slip_mm": 0.4, "max_placement_error_mm": 9.0},
            {"success": False, "max_slip_mm": 1.2, "max_placement_error_mm": 14.0},
        ]

        summary = summarize_stress_runs(runs)

        self.assertEqual(summary["runs"], 3)
        self.assertAlmostEqual(summary["success_rate"], 2 / 3)
        self.assertEqual(summary["worst_slip_mm"], 1.2)
        self.assertEqual(summary["worst_placement_error_mm"], 14.0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests/test_tactile_medkit_metrics.py -v`

Expected: FAIL or ERROR because `submissions.tactile_medkit_benchmark` does not exist.

- [ ] **Step 3: Implement metric modules**

Create `task_model.py` with `PROJECT_NAME`, `REGISTRATION_UUID`, `PHASES`, `OUTPUTS_DIR`, and phase target metadata. Create `metrics.py` with `compute_run_metrics(samples, seed)`, `build_contact_timeline(samples)`, and `summarize_stress_runs(runs)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests/test_tactile_medkit_metrics.py -v`

Expected: PASS.

## Task 2: Simulation And Artifacts

**Files:**
- Create: `tests/test_tactile_medkit_simulation.py`
- Create: `submissions/tactile_medkit_benchmark/scene.xml`
- Create: `submissions/tactile_medkit_benchmark/simulation.py`
- Create: `submissions/tactile_medkit_benchmark/run_demo.py`
- Create: `submissions/tactile_medkit_benchmark/run_stress_eval.py`

- [ ] **Step 1: Write the failing simulation tests**

```python
import json
import tempfile
import unittest
from pathlib import Path

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

    def test_run_stress_eval_writes_aggregate(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary = run_stress_eval(seeds=4, output_dir=Path(tmp))

            self.assertEqual(summary["runs"], 4)
            self.assertGreaterEqual(summary["success_rate"], 0.75)
            self.assertTrue((Path(tmp) / "stress_eval.json").exists())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests/test_tactile_medkit_simulation.py -v`

Expected: FAIL or ERROR because `simulation.py` does not exist.

- [ ] **Step 3: Implement scene and simulation**

Create an MJCF scene with a palm, five articulated fingers, emergency kit objects, tray slots, sensors, actuators, and a `demo` camera. Implement deterministic phase simulation, JSON artifact writing, stress evaluation, and optional ImageIO video rendering.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests/test_tactile_medkit_simulation.py -v`

Expected: PASS.

## Task 3: Submission Metadata And Validation

**Files:**
- Create: `tests/test_tactile_medkit_validation.py`
- Create: `submissions/tactile_medkit_benchmark/registration.json`
- Create: `submissions/tactile_medkit_benchmark/README.md`
- Create: `submissions/tactile_medkit_benchmark/JUDGE_BRIEF.md`
- Create: `submissions/tactile_medkit_benchmark/rubric_scorecard.json`
- Create: `submissions/tactile_medkit_benchmark/submission_manifest.json`
- Create: `submissions/tactile_medkit_benchmark/validate_submission.py`

- [ ] **Step 1: Write the failing validation tests**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests/test_tactile_medkit_validation.py -v`

Expected: FAIL or ERROR because `validate_submission.py` and metadata files do not exist.

- [ ] **Step 3: Implement metadata and validation**

Write UUID metadata, README, judge brief, scorecard, manifest, and `validate_submission(output_dir, require_video)` with explicit checks for registration, outputs, metric thresholds, and optional demo video.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests/test_tactile_medkit_validation.py -v`

Expected: PASS.

## Task 4: Full Local Verification And Judge Prep

**Files:**
- Modify generated outputs under `submissions/tactile_medkit_benchmark/outputs/`
- Create: `submissions/tactile_medkit_benchmark/INTERNAL_REVIEW.md`

- [ ] **Step 1: Run all unit tests**

Run: `python3 -m unittest discover -s tests -v`

Expected: PASS.

- [ ] **Step 2: Generate benchmark outputs**

Run: `python3 submissions/tactile_medkit_benchmark/run_demo.py --no-video --seed 42`

Expected: `summary.json`, `trajectory.json`, `contact_timeline.json`, and `final_report.txt` written under `outputs/`.

- [ ] **Step 3: Generate stress evaluation**

Run: `python3 submissions/tactile_medkit_benchmark/run_stress_eval.py --seeds 16`

Expected: `stress_eval.json` shows success rate at least 0.875.

- [ ] **Step 4: Validate submission**

Run: `python3 submissions/tactile_medkit_benchmark/validate_submission.py --no-video`

Expected: validation exits 0 and prints `VALID`.

- [ ] **Step 5: Attempt demo video render**

Run: `python3 submissions/tactile_medkit_benchmark/run_demo.py --seed 42`

Expected: `outputs/demo.mp4` exists, or rendering failure is documented while JSON validation remains green.

- [ ] **Step 6: Run three judge-style reviews**

Use three separate judge personas modeled on Claude, GPT, and Gemini. Each scores the same evidence package across the official 8 dimensions and writes findings to `INTERNAL_REVIEW.md`.

- [ ] **Step 7: Stop for user confirmation**

If the internal average is around 90/100 or higher, stop and report the score estimate and remaining risks. Do not open the official PR.
