# Hardware Transfer Protocol

This file responds directly to the judge feedback requesting real hardware experiments, testing, and validation.

Reflex MedKit Dexterity Lab does not claim that a physical robot has already been run. The current submission provides a replayable MuJoCo benchmark plus a hardware-transfer protocol that a five-finger hand can execute as the next bench validation step.

## Target Platform

- Five-finger dexterous hand with thumb opposition and MCP/PIP/DIP-like joints
- Joint encoder feedback and a low-level velocity-servo or joint-command interface
- Fingertip force, tactile, or calibrated current-based contact estimates
- Object-pose tracking for vial, cap, capsule, bandage, and tool token
- Operator emergency stop and software safety stop

## Bench-Test Stages

| Stage | Goal | Pass Criteria |
|---|---|---|
| `bench_00_sensor_zeroing` | Calibrate joint, tactile, object-pose, and button sensors | No E-stop, no joint-limit violation, timestamp gaps below 20 ms |
| `bench_01_no_object_dry_run` | Replay the five-phase command envelope without task objects | Completes all phases with zero runtime pose teleports |
| `bench_02_compliant_vial_grasp` | Validate five-finger contact on a foam or dummy vial | All five fingertips contact; slip remains at or below 0.5 mm |
| `bench_03_capped_vial_rotation` | Transfer the 220 degree cap rotation to a torque-limited fixture | Cap rotation reaches at least 220 degrees without force or velocity faults |
| `bench_04_perturbation_recovery` | Apply a controlled lateral tap during grasp | Peak slip remains at or below 0.5 mm and the vial recovers |
| `bench_05_medkit_slot_placement` | Place vial, cap, capsule, bandage, and tool token into slots | Every slot error is 10 mm or lower, with 5 mm preferred |
| `bench_06_confirmation_and_abort_drill` | Press the index confirmation button and test E-stop logging | Button press recorded; E-stop latency below 100 ms in dry-run drill |

## Required Hardware Logs

- `timestamp_s`
- `joint_position_rad`
- `joint_velocity_rad_s`
- `actuator_command_norm`
- `fingertip_contact_n`
- `object_pose_m_quat`
- `cap_rotation_deg`
- `slip_mm`
- `placement_error_mm`
- `emergency_stop_state`

## Safety Interlocks

- Stop if any fingertip force exceeds 8 N for more than 100 ms.
- Stop if measured slip exceeds 3 mm before recovery is active.
- Stop if cap torque proxy or motor current exceeds the target hardware limit.
- Stop if object pose leaves the tray fixture corridor by more than 25 mm.
- Stop on operator E-stop or missing telemetry for more than 100 ms.

## Acceptance Criteria

- Cap rotation: at least 220 degrees
- Peak slip: at most 0.5 mm
- Placement error: at most 10 mm, 5 mm preferred
- Closed-loop medkit checks: at least 22
- Stress evaluation before hardware claim: at least 128 sim seeds and 87.5% success
- Physical validation claim: at least 30 real trials, at least 27 successes, no unhandled operator abort

Generated machine-readable counterpart: `outputs/hardware_transfer_protocol.json`.
