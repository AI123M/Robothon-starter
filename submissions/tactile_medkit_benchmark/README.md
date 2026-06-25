# Reflex MedKit Dexterity Lab

Registration UUID: `f74c5b50-5ef8-467b-a141-a28ea9c34333`

## Project Summary

Reflex MedKit Dexterity Lab is a MuJoCo dexterous-hand benchmark for emergency-medical-kit assembly. It stages and evaluates five-finger vial handling, 220+ degree cap rotation, lateral slip recovery, multi-object kit-slot placement, and a final confirmation-button press.

Every run writes metrics, trajectory samples, a full policy trace, contact timeline, contact-geometry audit, physics rollout audit, policy calibration report, stress-evaluation output, policy ablation, and a final report.

## Robot Platform

Custom five-finger MuJoCo hand built from articulated MJCF bodies:

- Palm with thumb opposition
- Thumb, index, middle, ring, and little fingers
- MCP/PIP/DIP-like joints
- Fingertip pad geoms
- Position actuators and joint sensors

## Task Goal

Assemble a compact emergency medkit using dexterous manipulation:

1. Five-finger vial grasp
2. Cap rotation beyond 220 degrees
3. Lateral perturbation and slip recovery
4. Placement of vial, capsule, bandage, and tool token into kit slots
5. Index-finger confirmation-button press
6. Export of data suitable for robot-learning or benchmark evaluation

## Technical Approach

The simulator loads `scene.xml`, runs a task-space emergency-kit reference, then applies calibrated tactile force policy v4 before every MuJoCo step window. The policy reads the previous measured contacts, solver-contact pairs, slip, cap-rotation error, placement error, and button-contact state; it writes actuator, target, and velocity-servo corrections that are recorded in the trajectory and summary artifacts.

The policy is deterministic after seeding for replayability, but the episode is closed-loop: each sample has a `policy_observation`, `policy_residual`, and `closed_loop_update`. MuJoCo solver contacts from fingertip pads to visible object geoms and supplemental object contact shells are the primary contact evidence; site-distance contacts are retained only as secondary telemetry. `policy_training_report.json` records the deterministic calibration sweep, `contact_geometry_audit.json` verifies collision-enabled visible task objects, and `physics_rollout_audit.json` verifies zero runtime free-joint pose resets.

## Core Features

- Runnable MuJoCo MJCF workcell
- Five-finger hand with actuators, joints, sensors, and fingertip pads
- Emergency-kit scene with vial, cap, capsule, bandage, tool token, tray slots, and button
- Cap rotation metric above 220 degrees
- Slip-recovery metric below 0.5 mm
- Placement error metric below 10 mm
- Dexterity score target at or above 91 locally
- Multi-seed stress evaluation with nonzero metric variance
- Solver-contact evidence across all five phases
- Calibrated closed-loop force-policy evidence across the rollout
- Open-loop baseline ablation showing force-policy benefit
- Visible-object collision geometry audit
- Physics rollout audit with runtime qpos resets set to zero
- Force-policy calibration report
- 80-second MP4 evidence render with phase/metric overlay at 3 fps when rendering is enabled
- JSON outputs for reproducible judging

## Rubric Alignment

| Criterion | Evidence |
|---|---|
| Reproducibility | `run_demo.py`, `run_stress_eval.py`, deterministic seeds, `validate_submission.py` |
| MuJoCo depth | MJCF joints, actuators, sensors, free bodies, camera, visible object geoms, supplemental collision shells, and measured solver contacts |
| Task design | Emergency-kit assembly with multiple object types and final confirmation |
| Control | Calibrated tactile force policy v4 over a task-space reference; observation-conditioned actuator, target, and velocity-servo corrections before every `mj_step` window |
| Dexterity | Thumb opposition, five-finger contact, cap rotation, placement, button press |
| Engineering quality | Focused modules, generated evidence package, calibration report, geometry audit, 91+ validation script |
| Presentation | 80-second MP4 demo with live phase/metric overlay plus structured final report and trajectory |
| Innovation | Medkit assembly plus manipulation benchmark and data export |

## How to Run

Install dependencies from the repository root:

```bash
python3 -m pip install -r requirements.txt
```

Run the benchmark without video:

```bash
python3 submissions/tactile_medkit_benchmark/run_demo.py --no-video --seed 42
```

Run multi-seed stress evaluation:

```bash
python3 submissions/tactile_medkit_benchmark/run_stress_eval.py --seeds 16
```

Validate the submission artifacts:

```bash
python3 submissions/tactile_medkit_benchmark/validate_submission.py --no-video
```

Validate the full evidence package, including video:

```bash
python3 submissions/tactile_medkit_benchmark/validate_submission.py
```

Render the demo video when a MuJoCo rendering backend is available:

```bash
python3 submissions/tactile_medkit_benchmark/run_demo.py --seed 42
```

## Outputs

Generated under `submissions/tactile_medkit_benchmark/outputs/`:

- `summary.json`
- `trajectory.json`
- `policy_trace.json`
- `policy_ablation.json`
- `policy_training_report.json`
- `contact_geometry_audit.json`
- `physics_rollout_audit.json`
- `contact_timeline.json`
- `evidence_package.json`
- `stress_eval.json`
- `video_status.json`
- `final_report.txt`
- `demo.mp4` when rendering succeeds

## Current Limitations

- The force policy is compact and replayable rather than a large RL checkpoint.
- The run uses simulation-native state and metric instrumentation, not camera perception.
- Hand self-collision remains disabled for stability, but fingertip pads, visible task-object geoms, and supplemental contact shells are collision-enabled and validated through MuJoCo solver contacts.
- The scene is not claimed as real-hardware validated.

## Future Improvements

- Train a larger policy from the exported trajectory and stress-evaluation data.
- Add camera/depth observations and dataset labels.
- Expand perturbations, object families, and randomized medkit layouts.
- Compare against a baseline two-finger gripper to quantify dexterity benefits.
