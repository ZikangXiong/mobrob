# Goal-Conditioned Reinforcement Learning Control for Mobile Robots

## Installation

```shell
pip install git+https://github.com/ZikangXiong/mobrob
```

The simulation partially depends on [mujoco-py](https://github.com/ZikangXiong/mobrob). Configure the environment according to the [official guide](https://github.com/openai/mujoco-py#synopsis). 


## What Are Provided

### Robots: 
3 mobile robot environment based on mujoco-py

| Body Type | Description                            | Video |
| --------- | -------------------------------------- | ----- |
| Point     | A rigid body with a point mass         |       |
| Car       | A rigid body with car-like kinematics  |       |
| Doggo     | A rigid body with quadruped kinematics |       |

2 mobile robot environments based on pybullet
    
| Body Type | Description                                    | Video |
| --------- | ---------------------------------------------- | ----- |
| Drone     | A rigid body with drone kinematics             |       |
| Turtlebot | A rigid body with turtlebot3-waffle kinematics |       |

### RL Controllers: 

> All the control is trained with Proximal Policy Optimization (PPO). 

**Pretrained policies**: The pretrained policies are provided in [data/policies](/data/policies/). 

**Training parameters**: The training parameters are provided in [data/configs](/data/configs/). You can tune the parameters according to your needs. For all supported parameters, please refer to [stable-baselines3 PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)

**Training**: The training scripts are provided in [example/train.py](/examples/train.py). For example, if you want to train the point robot, run the following command:

```shell
python examples/train.py --env-name point 
```

You can also finetune the trained policy by adding the `--finetune` flag. 

```shell
python examples/train.py --env-name point --finetune
```

During training, the training logs and the intermediate policies will be saved in `data/tmp`.

**Evaluation**: The evaluation scripts are provided in [example/control.py](/examples/control.py). For example, if you want to evaluate the point robot, run the following command:

```shell
python examples/control.py --env-name point 
```

This will enable GUI for visualization. You can also disable the GUI by adding the `--no-gui` flag. 

```shell
python examples/control.py --env-name point --no-gui
```

## Related Publications

This repository is used in the following publications as benchmark environments.

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
