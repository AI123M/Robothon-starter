# Tactile MedKit Manipulation Benchmark Design

## Contest Identity

- Participant nickname: AIFF
- GitHub account: AI123M
- Registration UUID: f74c5b50-5ef8-467b-a141-a28ea9c34333
- AI tool: Codex
- Registered direction: dexterous hand / real-world emergency triage / data collection
- Project title: Tactile MedKit Manipulation Benchmark
- Submission folder: `submissions/tactile_medkit_benchmark/`

The project title intentionally does not include the participant nickname and avoids existing leaderboard names such as Dexterous Triage Lab, DextraForge, DexHand Lab, and AIDOOG Dexterous Triage Lab.

## Goal

Build a reproducible MuJoCo dexterous-hand benchmark where a five-finger robotic hand manipulates small emergency-medical-kit objects. The submission must score strongly on the official rubric: reproducibility, MuJoCo depth, task design, control, dexterity, engineering quality, presentation, and innovation.

The target readiness bar is an estimated 90/100 or better from three internal judge-style reviews before any official PR is opened.

## Competition Signals

The public leaderboard on 2026-06-19 shows the top entries clustered around dexterous-hand emergency/medication manipulation:

- Rank 1: 90.2, five-finger grasp, cap rotation, slip recovery.
- Rank 2: 89.4, 15 tasks, precision below 15 mm.
- Rank 3: 89.0, cap rotation, 0.35 mm slip recovery, 9x load hold.
- Rank 4: 88.6, 15-task arena, 100% success.

The project should therefore combine the strongest visible signals while improving on common weaknesses: video narrative, physical evidence, variety, reduced "scripted" feel, and clear validation.

## Experience

The demo should show a compact emergency-kit workcell:

1. The MuJoCo scene starts and displays the five-finger hand, kit tray, vial, cap, capsule, bandage roll, tool token, and confirmation button.
2. The hand performs object-specific grasping using coordinated fingers, not a simple clamp.
3. The hand rotates or unscrews a vial cap beyond 220 degrees.
4. The hand recovers from a controlled lateral perturbation and reports slip distance.
5. The hand places multiple objects into labeled kit slots.
6. The hand presses or toggles a final confirmation control.
7. The run exports judge-friendly evidence files and a 1-3 minute demo video.

## Core Components

- `scene.xml`: MJCF scene with hand, actuators, joints, contacts, sensors, task objects, kit tray, cameras, and visual labels.
- `run_demo.py`: one-command entrypoint that runs the full showcase, records telemetry, and optionally renders video.
- `controller.py`: deterministic state-machine controller with smooth joint targets, contact-aware phases, and perturbation handling.
- `metrics.py`: computes success rates, cap rotation, slip distance, contact counts, placement accuracy, and timing.
- `run_stress_eval.py`: runs multiple seeds without video and writes aggregate robustness results.
- `validate_submission.py`: checks required files, UUID consistency, JSON validity, output presence, and basic metrics thresholds.
- `README.md`: concise project explanation, run commands, rubric alignment, limitations, and demo notes.
- Evidence outputs under `outputs/`: `summary.json`, `trajectory.json`, `contact_timeline.json`, `stress_eval.json`, `rubric_scorecard.json`, and `final_report.txt`.

## Data Flow

`run_demo.py` loads the MJCF scene, seeds deterministic object poses, runs the controller phase sequence, records MuJoCo state snapshots and contact events, computes metrics, writes JSON/text reports, and renders `outputs/demo.mp4` when rendering dependencies are available.

Stress evaluation reuses the same controller and metrics with varied seeds and no video. Validation reads the generated artifacts and fails loudly if the UUID, outputs, or metric thresholds are missing.

## Scoring Strategy

- Reproducibility: one-command run, simple requirements, deterministic seeds, validation script.
- MuJoCo depth: explicit MJCF joints, actuators, contacts, sensors, cameras, material properties, perturbation object, and real simulation stepping.
- Task design: real-world emergency-kit assembly with multiple object types and final confirmation.
- Control: smooth finite-state control plus contact/slip feedback and recovery behaviors.
- Dexterity: five-finger coordination, thumb opposition, cap rotation, fingertip pads, and object-specific grasps.
- Engineering quality: small modules, generated evidence files, clear README, no unrelated repository changes.
- Presentation: concise demo with labels/HUD metrics, clear phases, and final scorecard.
- Innovation: combines triage, kit assembly, cap manipulation, perturbation recovery, and data collection.

## Non-Goals

- Do not claim real hardware deployment.
- Do not implement a large learned RL pipeline unless time allows a small optional placeholder-free baseline.
- Do not open an official PR or mark the project submitted without explicit user approval.
- Do not use any GitHub account except AI123M.

## Validation And Review Gate

Before asking the user for submission approval:

1. `python submissions/tactile_medkit_benchmark/run_demo.py --no-video`
2. `python submissions/tactile_medkit_benchmark/run_stress_eval.py --seeds 16`
3. `python submissions/tactile_medkit_benchmark/validate_submission.py`
4. Generate or verify `outputs/demo.mp4` or document a demo link.
5. Run three internal judge-style reviews modeled on Claude, GPT, and Gemini against the official 8 dimensions.
6. Stop and report when the estimated average is around 90/100 or higher.

Official PR submission remains blocked until the user explicitly confirms.

## Risks

- MuJoCo rendering may fail in headless environments. Mitigation: support `--no-video`, preserve existing video if rerender fails, and keep JSON validation separate from rendering.
- A heavily scripted controller may be penalized. Mitigation: expose contact/slip feedback, perturbation recovery, stress seeds, and honest limitations.
- Similarity to current top entries may reduce innovation score. Mitigation: frame the work as emergency-kit assembly with multiple object classes, final button interaction, and dataset export rather than only vial rotation.
