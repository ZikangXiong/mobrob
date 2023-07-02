from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from mobrob.envs.wrapper import get_env
from mobrob.utils import DATA_DIR

try:
    import tensorboard
except ImportError:
    tensorboard = None


class PPOCtrl:
    def __init__(
        self,
        ppo_kwargs: dict,
        env_name: str,
        time_limit: int,
        n_env: int,
        vec_env_type: str = "dummy",
        enable_gui: bool = False,
        seed: int = 0,
    ) -> None:
        self.ppo_kwargs = ppo_kwargs
        self.env_name = env_name
        self.time_limit = time_limit
        self.n_env = n_env

        if vec_env_type == "subproc":
            vec_env_cls = SubprocVecEnv
        elif vec_env_type == "dummy":
            vec_env_cls = DummyVecEnv
        else:
            raise ValueError(f"Unknown vec_env_type: {vec_env_type}")

        vec_env = make_vec_env(
            get_env,
            n_envs=n_env,
            env_kwargs={
                "env_name": env_name,
                "enable_gui": enable_gui,
                "terminate_on_goal": True,
                "time_limit": time_limit,
            },
            vec_env_cls=vec_env_cls,
            seed=seed,
        )

        self.ppo = PPO(
            env=vec_env,
            seed=seed,
            tensorboard_log=(
                f"{DATA_DIR}/policies/tmp/{env_name}-ppo/tensorboard"
                if tensorboard is not None
                else None
            ),
            **ppo_kwargs,
        )

    @classmethod
    def from_config(cls, config: dict) -> "PPOCtrl":
        return cls(
            ppo_kwargs=config["ppo_kwargs"],
            env_name=config["env_name"],
            time_limit=config["time_limit"],
            n_env=config["n_envs"],
            vec_env_type=config["vec_env_type"],
            enable_gui=config["enable_gui"],
            seed=config["seed"],
        )

    def learn(self, *args, **kwargs) -> None:
        self.ppo.learn(*args, **kwargs)

    def save_model(self, save_path: str) -> None:
        self.ppo.save(save_path)
