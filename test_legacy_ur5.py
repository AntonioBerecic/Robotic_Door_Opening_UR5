"""Test legacy UR5 checkpoints trained with a 7D movable-gripper policy."""

import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Test an old 7D UR5 checkpoint without changing new training."
    )
    parser.add_argument("--model", required=True, help="Checkpoint directory name")
    parser.add_argument("--model_id", type=int, required=True,
                        help="Checkpoint step, for example 10000")
    parser.add_argument("--render", action="store_true",
                        help="Show the MuJoCo viewer")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--debug_ft", action="store_true",
                        help="Print wrist force/torque readings")
    parser.add_argument("--debug_ft_every", type=int, default=10)
    parser.add_argument("--ft_log", default=None,
                        help="Optional force/torque CSV output path")
    args = parser.parse_args()

    from environment.ur5opendoorfktactile import ur5opendoorfktactile
    from rl.td3.train_td3 import train_td3

    env = ur5opendoorfktactile(
        fixed_gripper_flexion=None,
        ft_mode="normalized",
        legacy_reward=True,
        has_renderer=args.render,
        has_offscreen_renderer=False,
    )
    print("Legacy UR5 test mode: 7D policy, normalized F/T, 2026-08-31 reward")
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)

    try:
        train_td3(
            env=env,
            envs={},
            train=False,
            test=True,
            finetune=False,
            path=args.model,
            model_id=args.model_id,
            render=args.render,
            process=1,
            seed=args.seed,
            debug_ft=args.debug_ft,
            debug_ft_every=args.debug_ft_every,
            ft_log=args.ft_log,
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
