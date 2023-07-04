# Mobile Robot Control via Goal-Conditioned Reinforcement Learning

A collection of mobile robot environments and their goal-conditioned reinforcement learning controllers.

## Setup

Install the package via pip:

```shell
pip install git+https://github.com/ZikangXiong/mobrob
```

This project partially relies on mujoco-py. Follow the [official guide](https://github.com/openai/mujoco-py#synopsis) for setup.

## Features

### Environments: 

The repository provides five mobile robot environments:

| Body Type  | Description                  | Simulator | State dim | Action dim | Control type | Video                                                                                                               |
| ---------- | ---------------------------- | --------- | --------- | ---------- | ------------ | ------------------------------------------------------------------------------------------------------------------- |
| point      | point mass                   | mujoco-py | 14        | 2          | Control CMD  | <video src='https://github.com/ZikangXiong/mobrob/assets/73256697/4a4e5280-6b2c-46be-b44d-94d5a6c96d34' width=100/> |
| car        | car-like kinematics          | mujoco-py | 26        | 2          | Control CMD  | <video src='https://github.com/ZikangXiong/mobrob/assets/73256697/6ac6a44e-7f20-4a1a-91a9-5829989896af' width=100/> |
| doggo      | quadruped dog kinematics     | mujoco-py | 58        | 12         | Control CMD  | <video src='https://github.com/ZikangXiong/mobrob/assets/73256697/a9be5df1-3df4-4b81-a81f-2e826588c186' width=100/> |
| drone      | drone kinematics             | pybullet  | 12        | 18         | Neural PID   | <video src='https://github.com/ZikangXiong/mobrob/assets/73256697/456281ea-03f7-4368-8a60-1a36d67f009f' width=100/> |
| turtlebot3 | turtlebot3-waffle kinematics | pybullet  | 43        | 2          | Neural Prop  | <video src='https://github.com/ZikangXiong/mobrob/assets/73256697/d14d6713-557e-4980-a82b-998ad29c104e' width=100/> |

**Control CMD**: control commands directly sent to the engines.   
**Neural PID**: a neural network that maps the current state to the desired PID coefficients.  
**Neural Prop**: a neural network that maps the current state to the desired proportional control coefficients.  


### Reinforcement Learning Controllers: 

Controllers are trained using Proximal Policy Optimization (PPO). 

- **Pretrained policies**: Available at [data/policies](/data/policies/). 
- **Training parameters**: Available at [data/configs](/data/configs/). Refer to [stable-baselines3 PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html) for all supported parameters.
- **Training**: Use scripts in [example/train.py](/examples/train.py). For instance, to train the point robot:

```shell
python examples/train.py --env-name point 
```

To finetune a trained policy:

```shell
python examples/train.py --env-name point --finetune
```

Training logs and intermediate policies are saved in `data/tmp`.

- **Evaluation**: Use scripts in [example/control.py](/examples/control.py). For instance, to evaluate the point robot:

```shell
python examples/control.py --env-name point 
```

To disable the GUI:

```shell
python examples/control.py --env-name point --no-gui
```

## Customization

For users intending to reuse the training and simulation code provided in the repository, the following abstract functions in the abstract class `EnvWrapper` of [envs/wrapper.py](/src/mobrob/envs/wrapper.py) should be rewritten according to the specific needs of the new robot environment. The functions, along with their brief explanations, are given in the table below:

| Function Name                 | Description                                                      |
| ----------------------------- | ---------------------------------------------------------------- |
| `_set_goal(self, goal)`       | Sets the goal position of the robot. Example: [x, y, z]          |
| `build_env(self)`             | Constructs the environment, i.e., loads the robot and the world. |
| `get_pos(self)`               | Retrieves the current position of the robot. Example: [x, y, z]  |
| `set_pos(self, pos)`          | Sets the position of the robot. Example: [x, y, z]               |
| `get_obs(self)`               | Obtains the current observation of the robot. Example: [x, y, z] |
| `get_observation_space(self)` | Gets the observation space of the robot. Example: Box(3,)        |
| `get_action_space(self)`      | Retrieves the action space of the robot. Example: Box(3,)        |
| `get_init_space(self)`        | Fetches the initial space of the robot. Example: Box(3,)         |
| `get_goal_space(self)`        | Acquires the goal space of the robot. Example: Box(3,)           |

One may refer to the other robot environment wrappers in [wrapper.py](/src/mobrob/envs/wrapper.py) for more details.

## Publications

This repository is used in the following papers as the benchmark environment:

```bibtex
@inproceedings{mfnlc,
  author={Xiong, Zikang and Eappen, Joe and Qureshi, Ahmed H. and Jagannathan, Suresh},
  booktitle={2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, 
  title={Model-free Neural Lyapunov Control for Safe Robot Navigation}, 
  year={2022},
  pages={5572-5579},
  doi={10.1109/IROS47612.2022.9981632}}

@article{dscrl,
  title={Co-learning Planning and Control Policies Using Differentiable Formal Task Constraints},
  author={Xiong, Zikang and Eappen, Joe and Lawson, Daniel and Qureshi, Ahmed H and Jagannathan, Suresh},
  journal={arXiv preprint arXiv:2303.01346},
  year={2023}
}
```
