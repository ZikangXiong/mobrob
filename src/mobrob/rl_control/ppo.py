from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from mobrob.envs.wrapper import get_env


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

        def get_one_env_fn(seed: int):
            def create_env():
                env = get_env(env_name, enable_gui=enable_gui, terminate_on_goal=True)
                env = TimeLimit(env, time_limit)
                env = Monitor(env)

                env.seed(seed)

                return env

            return create_env

        env_fns = [get_one_env_fn(seed + i) for i in range(n_env)]

        if vec_env_type == "subproc":
            vec_env = SubprocVecEnv(env_fns)
        elif vec_env_type == "dummy":
            vec_env = DummyVecEnv(env_fns)
        else:
            raise ValueError(f"Unknown vec_env_type: {vec_env_type}")

        self.ppo = PPO(env=vec_env, **ppo_kwargs)

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

    def learn(self, total_timesteps: int = 1_000_000) -> None:
        self.ppo.learn(total_timesteps=total_timesteps)

    def save_model(self, save_path: str) -> None:
        self.ppo.save(save_path)
