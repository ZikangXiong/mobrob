from mobrob import load_policy
from mobrob.utils import DATA_DIR


def load_and_save_policy():
    env_names = [
        "point",
        "car",
        "doggo",
        "drone",
        "turtlebot3",
    ]

    for env_name in env_names:
        policy = load_policy(env_name, "ppo")
        policy.save(f"{DATA_DIR}/policies/{env_name}-ppo.zip")


if __name__ == "__main__":
    load_and_save_policy()
