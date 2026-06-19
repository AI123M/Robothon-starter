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
            {
                "phase": "vial_grasp",
                "cap_rotation_deg": 0,
                "slip_mm": 0.0,
                "placement_error_mm": 0.0,
                "contacts": ["thumb", "index", "middle"],
            },
            {
                "phase": "cap_rotation",
                "cap_rotation_deg": 226.0,
                "slip_mm": 0.2,
                "placement_error_mm": 0.0,
                "contacts": ["thumb", "index", "middle", "ring"],
            },
            {
                "phase": "perturb_recovery",
                "cap_rotation_deg": 226.0,
                "slip_mm": 0.34,
                "placement_error_mm": 0.0,
                "contacts": ["thumb", "index", "middle", "ring", "little"],
            },
            {
                "phase": "kit_assembly",
                "cap_rotation_deg": 226.0,
                "slip_mm": 0.28,
                "placement_error_mm": 7.5,
                "contacts": ["thumb", "index", "middle"],
            },
            {
                "phase": "confirmation_button",
                "cap_rotation_deg": 226.0,
                "slip_mm": 0.2,
                "placement_error_mm": 6.0,
                "contacts": ["index"],
            },
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
