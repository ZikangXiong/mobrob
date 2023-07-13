import os
import sys
from contextlib import contextmanager
from os.path import abspath, dirname

import pybullet as p
from stable_baselines3 import PPO

import mobrob

DATA_DIR = os.path.join(dirname(dirname(dirname(abspath(mobrob.__file__)))), "data")
PROJ_DIR = dirname(abspath(mobrob.__file__))


def load_policy(env_name: str, policy_name: str):
    return PPO.load(f"{DATA_DIR}/policies/{env_name}-{policy_name}.zip")


class BulletVideoRecorder:
    def __init__(self, client_id: int, store_path: str):
        self.client_id = client_id
        self.store_path = store_path
        self.logging_unique_id = None

    def __enter__(self):
        if self.store_path is not None:
            os.makedirs(f"{os.path.dirname(self.store_path)}", exist_ok=True)
            self.logging_unique_id = p.startStateLogging(
                loggingType=p.STATE_LOGGING_VIDEO_MP4,
                fileName=self.store_path,
                physicsClientId=self.client_id,
            )
            return self.logging_unique_id

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.store_path is not None:
            p.stopStateLogging(self.logging_unique_id)


@contextmanager
def suppress_stdout():
    fd = sys.stdout.fileno()

    def _redirect_stdout(to):
        sys.stdout.close()  # + implicit flush()
        os.dup2(to.fileno(), fd)  # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, "w")  # Python writes to fd

    with os.fdopen(os.dup(fd), "w") as old_stdout:
        with open(os.devnull, "w") as file:
            _redirect_stdout(to=file)
        try:
            yield  # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout)  # restore stdout.
            # buffering and flags such as
            # CLOEXEC may be different
