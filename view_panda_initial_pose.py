"""Display the Panda door environment in its initial pose."""

import argparse
import time

import glfw

from environment.pandaopendoorfktactile import PandaOpenDoorFKTactile


def parse_args():
    parser = argparse.ArgumentParser(
        description="Show the initial Panda pose without applying actions."
    )
    parser.add_argument(
        "--camera_id",
        type=int,
        default=0,
        help="MuJoCo fixed camera ID (default: 0)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    env = PandaOpenDoorFKTactile(
        has_renderer=True,
        has_offscreen_renderer=False,
        render_visual_mesh=True,
        use_camera_obs=False,
    )

    try:
        env.reset()
        env.viewer.set_camera(camera_id=args.camera_id)
        print("Panda initial pose is displayed. Close the window or press Ctrl+C to exit.")

        # Render only: do not call env.step() or sim.step(), so the reset pose
        # remains unchanged.
        while not glfw.window_should_close(env.viewer.viewer.window):
            env.render()
            time.sleep(1.0 / 60.0)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
