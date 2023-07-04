import argparse
import time

import numpy as np
from gymnasium.wrappers import RecordVideo

from mobrob import get_env, load_policy
from mobrob.utils import BulletVideoRecorder


def simulate(
    env_name: str,
    policy_name: str,
    epochs,
    no_gui: bool,
    video_path: str,
):
    env = get_env(env_name, enable_gui=not no_gui, terminate_on_goal=True)
    policy = load_policy(env_name, policy_name)

    rewards = []

    video_recoder = None
    if env_name in ("point", "car", "doggo") and video_path is not None:
        env.toggle_render_mode()
        env = RecordVideo(env, video_path)
        no_gui = (
            True  # no gui when recording video, this is the limitation of mujoco_py
        )
    elif env_name in ("drone", "turtlebot3"):
        video_recoder = BulletVideoRecorder(env.env.client_id, video_path)

    def run():
        nonlocal rewards
        for _ in range(epochs):
            cum_reward = 0
            obs, _ = env.reset()
            for _ in range(1000):
                action, _ = policy.predict(obs, deterministic=True)
                obs, r, terminated, _, _ = env.step(action)

                if terminated:
                    obs, _ = env.reset()

                cum_reward += r

                if not no_gui:
                    env.render()

                if env_name in ("drone", "turtlebot3"):
                    time.sleep(0.005)

            rewards.append(cum_reward)

    if video_recoder is not None:
        with video_recoder:
            run()
    else:
        run()

    print(f"average reward: {np.mean(rewards)}")
    print(f"reward stds: {np.std(rewards)}")
    print(f"rewards: {rewards}")


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--env-name", type=str, default="point")
    args_parser.add_argument("--policy-name", type=str, default="ppo")
    args_parser.add_argument("--epochs", type=int, default=5)
    args_parser.add_argument("--no-gui", action="store_true", default=False)
    args_parser.add_argument("--video-path", type=str, default=None)

    args = args_parser.parse_args()

    simulate(
        env_name=args.env_name,
        policy_name=args.policy_name,
        epochs=args.epochs,
        no_gui=args.no_gui,
        video_path=args.video_path,
    )
