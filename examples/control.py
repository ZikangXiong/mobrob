import argparse
import time

import numpy as np

from mobrob import get_env, load_policy


def simulate(env_name: str, policy_name: str):
    env = get_env(env_name, enable_gui=True, terminate_on_goal=True)
    policy = load_policy(env_name, policy_name)

    rewards = []

    for _ in range(5):
        cum_reward = 0
        obs, _ = env.reset()
        for _ in range(1000):
            action, _ = policy.predict(obs, deterministic=True)
            obs, r, terminated, _, _ = env.step(action)

            if terminated:
                obs, _ = env.reset()

            cum_reward += r
            env.render()

            if env_name in ("drone", "turtlebot3"):
                time.sleep(0.005)

        rewards.append(cum_reward)

    print(f"average reward: {np.mean(rewards)}")
    print(f"reward stds: {np.std(rewards)}")
    print(f"rewards: {rewards}")


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--env_name", type=str, default="point")
    args_parser.add_argument("--policy_name", type=str, default="ppo")

    args = args_parser.parse_args()

    simulate(env_name=args.env_name, policy_name=args.policy_name)
