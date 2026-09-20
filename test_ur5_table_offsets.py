"""Visually evaluate one UR5 policy over a grid of table position offsets."""

import argparse
import csv
import itertools
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Render a UR5 checkpoint at each requested table x/y offset and "
            "print success, reward, distance, orientation, and door angle."
        )
    )
    parser.add_argument(
        "--model", required=True,
        help="Checkpoint directory name or path, e.g. 20260901_1205121234",
    )
    parser.add_argument(
        "--model_id", type=int, default=None,
        help="Checkpoint ID. If omitted, use the largest available ID.",
    )
    parser.add_argument(
        "--limit_cm", type=float, default=5.0,
        help="Generate offsets from -LIMIT_CM to +LIMIT_CM (default: 5 cm).",
    )
    parser.add_argument(
        "--step_cm", type=float, default=1.0,
        help="Spacing between automatically generated offsets (default: 1 cm).",
    )
    parser.add_argument(
        "--x_values", type=float, nargs="+", default=None,
        help="Optional explicit x offsets in metres; overrides automatic values.",
    )
    parser.add_argument(
        "--y_values", type=float, nargs="+", default=None,
        help="Optional explicit y offsets in metres; overrides automatic values.",
    )
    parser.add_argument(
        "--sweep", choices=("grid", "axes"), default="grid",
        help=(
            "grid tests every x/y combination; axes varies x with y=0 and "
            "then y with x=0."
        ),
    )
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--action_range", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--fixed_gripper_flexion", type=float, default=None,
        help="Flexion for a 6D fixed-gripper policy; inferred as 0.1 if needed.",
    )

    ft_group = parser.add_mutually_exclusive_group()
    ft_group.add_argument(
        "--no_ft", dest="ft_mode", action="store_const", const="none",
        help="Use an 11D observation without F/T values.",
    )
    ft_group.add_argument(
        "--use_ft", dest="ft_mode", action="store_const", const="raw",
        help="Use a 17D observation with raw F/T values.",
    )
    ft_group.add_argument(
        "--normalized_ft", dest="ft_mode", action="store_const",
        const="normalized",
        help="Use a 17D observation with normalized F/T values.",
    )
    parser.set_defaults(ft_mode=None)

    render_group = parser.add_mutually_exclusive_group()
    render_group.add_argument(
        "--render", dest="render", action="store_true",
        help="Show the viewer while testing (default).",
    )
    render_group.add_argument(
        "--no_render", dest="render", action="store_false",
        help="Run without the viewer.",
    )
    parser.set_defaults(render=True)
    parser.add_argument(
        "--output", type=Path, default=None,
        help=(
            "Optional CSV output path. By default results are saved inside "
            "the tested model directory."
        ),
    )
    parser.add_argument(
        "--step_delay", type=float, default=0.005,
        help="Seconds to wait after every rendered simulation step.",
    )
    parser.add_argument(
        "--position_delay", type=float, default=1.0,
        help="Seconds to show a new table position before running the policy.",
    )
    parser.add_argument(
        "--wait_for_enter", action="store_true",
        help="Wait for Enter before starting every table position.",
    )
    parser.add_argument("--camera_id", type=int, default=0)
    return parser.parse_args()


def resolve_model_dir(model):
    supplied = Path(model).expanduser()
    candidates = [
        supplied,
        PROJECT_ROOT / supplied,
        PROJECT_ROOT / "data" / "weights" / supplied,
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        "Model directory not found. Checked: {}".format(
            ", ".join(str(path) for path in candidates)
        )
    )


def resolve_policy_path(model_dir, model_id):
    if model_id is not None:
        policy_path = model_dir / "{}_td3_policy".format(model_id)
        if not policy_path.is_file():
            raise FileNotFoundError("Policy checkpoint not found: {}".format(policy_path))
        return policy_path, model_id

    checkpoints = []
    for path in model_dir.glob("*_td3_policy"):
        prefix = path.name.split("_", 1)[0]
        if prefix.isdigit():
            checkpoints.append((int(prefix), path))
    if not checkpoints:
        raise FileNotFoundError(
            "No numbered *_td3_policy checkpoints found in {}".format(model_dir)
        )
    return max(checkpoints, key=lambda item: item[0])[1], max(checkpoints)[0]


def load_policy_state(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def infer_ft_mode(requested_mode, input_dim):
    if requested_mode is not None:
        return requested_mode
    if input_dim == 11:
        print("Auto-selected F/T mode: none (11D checkpoint)")
        return "none"
    if input_dim == 17:
        print(
            "Auto-selected F/T mode: normalized (17D checkpoint). "
            "Use --use_ft if this checkpoint was trained with raw F/T."
        )
        return "normalized"
    raise ValueError(
        "Cannot infer F/T mode from checkpoint input dimension {}. "
        "Select a mode explicitly.".format(input_dim)
    )


def automatic_offsets(limit_cm, step_cm):
    values_cm = np.arange(-limit_cm, limit_cm + 0.5 * step_cm, step_cm)
    if values_cm[-1] < limit_cm - 1e-9:
        values_cm = np.append(values_cm, limit_cm)
    values_cm = np.clip(values_cm, -limit_cm, limit_cm)
    return [float(value / 100.0) for value in values_cm]


def build_positions(args):
    automatic = automatic_offsets(args.limit_cm, args.step_cm)
    x_values = args.x_values if args.x_values is not None else automatic
    y_values = args.y_values if args.y_values is not None else automatic

    if args.sweep == "grid":
        return list(itertools.product(x_values, y_values))

    positions = [(x, 0.0) for x in x_values]
    positions.extend(
        (0.0, y) for y in y_values if (0.0, y) not in positions
    )
    return positions


def resolve_output_path(requested_path, model_dir, model_id, ft_mode):
    if requested_path is not None:
        output_path = requested_path.expanduser()
        if not output_path.is_absolute():
            output_path = Path.cwd() / output_path
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = (
            model_dir
            / "table_offset_tests"
            / "checkpoint_{}_{}_{}.csv".format(model_id, ft_mode, timestamp)
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path.resolve()


def main():
    args = parse_args()
    if args.max_steps < 1:
        raise ValueError("--max_steps must be at least 1")
    if args.limit_cm < 0.0:
        raise ValueError("--limit_cm cannot be negative")
    if args.step_cm <= 0.0:
        raise ValueError("--step_cm must be greater than zero")
    if args.step_delay < 0.0 or args.position_delay < 0.0:
        raise ValueError("Viewer delays cannot be negative")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)

    model_dir = resolve_model_dir(args.model)
    policy_path, model_id = resolve_policy_path(model_dir, args.model_id)
    state_dict = load_policy_state(policy_path)
    input_dim = int(state_dict["linear1.weight"].shape[1])
    hidden_dim = int(state_dict["linear1.weight"].shape[0])
    output_dim = int(state_dict["output_linear.weight"].shape[0])
    ft_mode = infer_ft_mode(args.ft_mode, input_dim)

    if args.fixed_gripper_flexion is None:
        fixed_flexion = None if output_dim == 7 else 0.1
    else:
        fixed_flexion = args.fixed_gripper_flexion

    from environment.ur5opendoorfktactile import ur5opendoorfktactile
    from rl.policy_networks import DPG_PolicyNetwork
    from mujoco_py import MujocoException

    def make_env():
        return ur5opendoorfktactile(
            fixed_gripper_flexion=fixed_flexion,
            ft_mode=ft_mode,
            has_renderer=args.render,
            has_offscreen_renderer=False,
        )

    env = make_env()
    if env.observation_space.shape[0] != input_dim:
        raise ValueError(
            "Checkpoint expects {} observation values, but mode '{}' creates {}."
            .format(input_dim, ft_mode, env.observation_space.shape[0])
        )
    if env.action_space.shape[0] != output_dim:
        raise ValueError(
            "Checkpoint outputs {} actions, but the selected gripper creates {}."
            .format(output_dim, env.action_space.shape[0])
        )

    policy = DPG_PolicyNetwork(
        env.observation_space,
        env.action_space,
        hidden_dim,
        args.action_range,
        machine_type="cpu",
    )
    policy.load_state_dict(state_dict)
    policy.eval()

    positions = build_positions(args)
    output_path = resolve_output_path(args.output, model_dir, model_id, ft_mode)
    results = []
    print("Model directory:", model_dir)
    print("Checkpoint:", model_id)
    print("Observation / action dimensions:", input_dim, "/", output_dim)
    print("F/T mode:", ft_mode)
    print("Render:", args.render)
    print("Results CSV:", output_path)
    if args.x_values is None and args.y_values is None:
        print(
            "Automatic offset range: -{0:g} cm to +{0:g} cm, step {1:g} cm"
            .format(args.limit_cm, args.step_cm)
        )
    print("Positions:", len(positions))

    csv_fieldnames = [
        "model", "checkpoint", "ft_mode", "render", "sweep", "seed",
        "offset_x_m", "offset_y_m", "offset_x_cm", "offset_y_cm",
        "success", "steps", "episode_reward", "max_door_angle_rad",
        "final_door_angle_rad", "final_door_angle_deg",
        "min_knob_distance_m", "min_orientation_error", "error",
    ]
    output_file = output_path.open("w", newline="")
    csv_writer = csv.DictWriter(output_file, fieldnames=csv_fieldnames)
    csv_writer.writeheader()
    output_file.flush()

    try:
        for index, (offset_x, offset_y) in enumerate(positions, start=1):
            print(
                "\n[{}/{}] table_position_offset_x={:+.4f} m ({:+.1f} cm), "
                "table_position_offset_y={:+.4f} m ({:+.1f} cm)".format(
                    index, len(positions), offset_x, offset_x * 100.0,
                    offset_y, offset_y * 100.0
                ),
                flush=True,
            )
            state = env.reset(
                table_position_offset_x=offset_x,
                table_position_offset_y=offset_y,
            )
            if args.render and env.viewer is not None:
                env.viewer.set_camera(camera_id=args.camera_id)
                env.render()
                if args.wait_for_enter:
                    input("Press Enter to start this position...")
                elif args.position_delay:
                    time.sleep(args.position_delay)

            episode_reward = 0.0
            success = False
            max_door_angle = 0.0
            min_knob_distance = float("inf")
            min_orientation_error = float("inf")
            mujoco_error = ""

            for step in range(args.max_steps):
                if not np.all(np.isfinite(state)):
                    raise RuntimeError("Non-finite observation at step {}".format(step))
                action = policy.get_action(state, noise_scale=0.0)
                try:
                    state, reward, done, info = env.step(action)
                except MujocoException as exc:
                    mujoco_error = str(exc)
                    print(
                        "MuJoCo numerical error at step {}; recording this "
                        "position as failed and continuing.".format(step + 1),
                        flush=True,
                    )
                    break
                episode_reward += float(reward)
                success = success or bool(info.get("success", False))
                max_door_angle = max(
                    max_door_angle, float(info.get("door_open_angle", 0.0))
                )
                min_knob_distance = min(
                    min_knob_distance,
                    float(info.get("distance_to_knob", float("inf"))),
                )
                min_orientation_error = min(
                    min_orientation_error,
                    float(info.get("orientation_error", float("inf"))),
                )
                if args.render:
                    env.render()
                    if args.step_delay:
                        time.sleep(args.step_delay)
                if done:
                    break

            final_door_angle = abs(float(
                env.sim.data.get_joint_qpos("hinge0")
            ))
            result = {
                "x": offset_x,
                "y": offset_y,
                "success": success,
                "steps": step + 1,
                "reward": episode_reward,
                "max_angle": max_door_angle,
                "final_angle": final_door_angle,
                "min_distance": min_knob_distance,
                "min_orientation": min_orientation_error,
                "error": mujoco_error,
            }
            results.append(result)
            csv_writer.writerow({
                "model": model_dir.name,
                "checkpoint": model_id,
                "ft_mode": ft_mode,
                "render": args.render,
                "sweep": args.sweep,
                "seed": args.seed,
                "offset_x_m": offset_x,
                "offset_y_m": offset_y,
                "offset_x_cm": offset_x * 100.0,
                "offset_y_cm": offset_y * 100.0,
                "success": success,
                "steps": step + 1,
                "episode_reward": episode_reward,
                "max_door_angle_rad": max_door_angle,
                "final_door_angle_rad": final_door_angle,
                "final_door_angle_deg": np.degrees(final_door_angle),
                "min_knob_distance_m": min_knob_distance,
                "min_orientation_error": min_orientation_error,
                "error": mujoco_error,
            })
            # Preserve completed positions even if the user interrupts later.
            output_file.flush()
            print(
                "Result: success={} steps={} reward={:.3f} max_angle={:.4f} "
                "final_angle={:.4f} rad ({:.2f} deg) "
                "min_distance={:.4f} min_orientation={:.4f} error={}".format(
                    success,
                    step + 1,
                    episode_reward,
                    max_door_angle,
                    final_door_angle,
                    np.degrees(final_door_angle),
                    min_knob_distance,
                    min_orientation_error,
                    mujoco_error or "none",
                ),
                flush=True,
            )
            if mujoco_error:
                # Do not reuse a simulation after a numerical instability.
                env.close()
                env = make_env()
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")
    finally:
        output_file.close()
        env.close()

    print("\n=== TABLE OFFSET SUMMARY ===")
    for result in results:
        print(
            "x={x:+.4f} y={y:+.4f} | {status:7s} | steps={steps:4d} | "
            "reward={reward:9.3f} | final_angle={final_angle:.4f} | "
            "max_angle={max_angle:.4f} | dist={min_distance:.4f}"
            .format(
                status=("ERROR" if result["error"] else
                        "SUCCESS" if result["success"] else "FAIL"),
                **result
            )
        )
    if results:
        successes = sum(result["success"] for result in results)
        print("Success rate: {}/{} ({:.1%})".format(
            successes, len(results), successes / len(results)
        ))
        print("Average episode length: {:.2f} steps".format(
            np.mean([result["steps"] for result in results])
        ))
        mean_final_angle = float(np.mean([
            result["final_angle"] for result in results
        ]))
        print("Average final door angle: {:.4f} rad ({:.2f} deg)".format(
            mean_final_angle, np.degrees(mean_final_angle)
        ))
        numerical_errors = sum(bool(result["error"]) for result in results)
        print("MuJoCo numerical errors: {}/{}".format(
            numerical_errors, len(results)
        ))
    print("Results saved to:", output_path)


if __name__ == "__main__":
    main()
