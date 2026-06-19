# Tactile MedKit Manipulation Benchmark

Registration UUID: `f74c5b50-5ef8-467b-a141-a28ea9c34333`

## Project Summary

Tactile MedKit Manipulation Benchmark is a MuJoCo dexterous-hand benchmark for emergency-medical-kit assembly. A five-finger robotic hand grasps a vial, rotates the cap beyond 220 degrees, recovers from a lateral slip disturbance, places multiple medical-kit objects into labeled tray slots, and presses a final confirmation button.

The project is designed as a judge-friendly evidence package: every run writes metrics, trajectory samples, contact timeline, stress-evaluation output, and a final report.

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

The simulator loads `scene.xml`, runs a deterministic multi-phase controller, evaluates the MuJoCo scene at each frame, records telemetry, computes metrics, and optionally renders `outputs/demo.mp4`.

The controller is intentionally deterministic for reproducibility. It sends joint targets through MuJoCo position actuators, advances every sample with `mj_step`, and reports slip plus MuJoCo solver contacts from selective fingertip-to-object contact shells. Site-distance contacts are retained only as supplemental telemetry.

## Core Features

- Runnable MuJoCo MJCF workcell
- Five-finger hand with actuators, joints, sensors, and fingertip pads
- Emergency-kit scene with vial, cap, capsule, bandage, tool token, tray slots, and button
- Cap rotation metric above 220 degrees
- Slip-recovery metric below 0.5 mm
- Placement error metric below 10 mm
- Multi-seed stress evaluation with nonzero metric variance
- Solver-contact evidence across all five phases
- 80-second MP4 evidence render at 3 fps when rendering is enabled
- JSON outputs for reproducible judging

## Rubric Alignment

| Criterion | Evidence |
|---|---|
| Reproducibility | `run_demo.py`, `run_stress_eval.py`, deterministic seeds, `validate_submission.py` |
| MuJoCo depth | MJCF joints, actuators, sensors, free bodies, camera, object geoms, selective collision shells, and measured solver contacts |
| Task design | Emergency-kit assembly with multiple object types and final confirmation |
| Control | Smooth phase controller with actuator targets, `mj_step` physics steps, contact/slip evidence, and perturbation recovery |
| Dexterity | Thumb opposition, five-finger contact, cap rotation, placement, button press |
| Engineering quality | Focused modules, generated evidence package, validation script |
| Presentation | 80-second MP4 demo plus structured final report and trajectory |
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

Render the demo video when a MuJoCo rendering backend is available:

```bash
python3 submissions/tactile_medkit_benchmark/run_demo.py --seed 42
```

## Outputs

Generated under `submissions/tactile_medkit_benchmark/outputs/`:

- `summary.json`
- `trajectory.json`
- `contact_timeline.json`
- `evidence_package.json`
- `stress_eval.json`
- `video_status.json`
- `final_report.txt`
- `demo.mp4` when rendering succeeds

## Current Limitations

- The controller is deterministic and benchmark-oriented, not a learned RL policy.
- The run uses simulation-native state and metric instrumentation, not camera perception.
- Hand self-collision remains disabled for stability, but fingertip pads and task-object contact shells are collision-enabled and validated through MuJoCo solver contacts.
- The scene is not claimed as real-hardware validated.

## Future Improvements

- Add learned residual control over the deterministic phase controller.
- Add camera/depth observations and dataset labels.
- Expand perturbations, object families, and randomized medkit layouts.
- Compare against a baseline two-finger gripper to quantify dexterity benefits.
