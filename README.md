# Mobile Robot Control via Goal-Conditioned Reinforcement Learning

## Setup

Install the package via pip:

```shell
pip install git+https://github.com/ZikangXiong/mobrob
```

This project partially relies on [mujoco-py](https://github.com/ZikangXiong/mobrob). Follow the [official guide](https://github.com/openai/mujoco-py#synopsis) for setup.

## Features

### Environments: 

The repository provides five mobile robot environments:

| Body Type  | Description                  | Framework | State dim | Action dim | Control type | Video                                           |
| ---------- | ---------------------------- | --------- | --------- | ---------- | ------------ | ----------------------------------------------- |
| point      | point mass                   | mujoco-py | 14        | 2          | Control CMD  | <video src='/assets/point.mp4' width=180/>      |
| car        | car-like kinematics          | mujoco-py | 26        | 2          | Control CMD  | <video src='/assets/car.mp4' width=180/>        |
| doggo      | quadruped dog kinematics     | mujoco-py | 58        | 12         | Control CMD  | <video src='/assets/doggo.mp4' width=180/>      |
| drone      | drone kinematics             | pybullet  | 12        | 18         | Neural PID   | <video src='/assets/drone.mp4' width=180/>      |
| turtlebot3 | turtlebot3-waffle kinematics | pybullet  | 43        | 2          | Neural Prop  | <video src='/assets/turtlebot3.mp4' width=180/> |

* Control CMD: control commands directly sent to the engines. 
* Neural PID: a neural network that maps the current state to the desired PID coefficients.
* Neural Prop: a neural network that maps the current state to the desired propotional control coefficients.


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

## Publications

This repository is used in the following papers as the benchmark environments:

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
