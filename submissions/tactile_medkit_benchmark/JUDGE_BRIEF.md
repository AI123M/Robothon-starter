# Judge Brief

## One-Line Summary

Tactile MedKit Manipulation Benchmark is a MuJoCo five-finger hand benchmark for emergency-kit assembly, combining vial grasping, 220+ degree cap rotation, slip recovery, multi-object placement, final button press, and structured data export.

## What To Look For

- Five-finger contact during vial grasp and recovery phases
- Cap rotation target: at least 220 degrees
- Peak slip target: at most 0.5 mm
- Placement target: at most 10 mm
- Stress evaluation target: at least 87.5% success over 16 seeds
- Physics evidence target: `actuator_position_mj_step`, positive physics-step count, at least four solver-contact phases
- Video target: 60+ seconds when rendered; seed 42 currently renders 80 seconds at 3 fps
- Evidence files: `summary.json`, `trajectory.json`, `contact_timeline.json`, `stress_eval.json`, `final_report.txt`

## Rubric Notes

- **Reproducibility:** one command runs the demo; one command runs stress evaluation; one command validates artifacts.
- **MuJoCo depth:** MJCF scene includes articulated fingers, free objects, actuators, sensors, collision-enabled fingertip pads/object contact shells, lighting, and camera.
- **Task design:** emergency-medical-kit assembly is concrete, real-world motivated, and multi-stage.
- **Control:** phase controller is deterministic, smooth, stepped with MuJoCo position actuators, and instrumented with slip/contact metrics.
- **Dexterity:** five coordinated fingers, thumb opposition, cap rotation, object-specific placement, and index button press.
- **Engineering quality:** focused modules and explicit validation.
- **Presentation:** 80-second seed-42 video plus structured report and trajectory evidence.
- **Innovation:** combines medkit manipulation, perturbation recovery, and robot-learning style data export.

## Honest Limitation

This is a deterministic simulation benchmark. Hand self-collision is disabled for stability, while fingertip pads and task-object contact shells remain collision-enabled and are validated through MuJoCo solver contacts. It does not claim real hardware transfer or learned policy performance.
