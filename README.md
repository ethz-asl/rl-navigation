Reinforced Imitation
====================

This repository contains the tensorflow implementation for training a reinforcement learning based map-less navigation model, as described in the paper:\
[Reinforced Imitation: Sample Efficient Deep Reinforcement Learning for Map-less Navigation by Leveraging Prior Demonstrations](https://arxiv.org/abs/1805.07095)

# Requirements
1. Ubuntu
2. Python 2.7
3. [ROS Indigo](http://wiki.ros.org/indigo) or [ROS Kinetic](http://wiki.ros.org/kinetic)
4. Stage-ros simulator, with `add_pose_sub` enabled. Can be found in this [branch](https://github.com/ros-simulation/stage_ros/tree/add_pose_sub) of the repository.

# Training the Model
1. First run the stage simulator:
`roslaunch reinforcement_learning_navigation stage_sim.launch`
2. In a separate terminal, run the training code:
`rosrun reinforcement_learning_navigation train_cpo.py --output_name $experiment_name$`\
In order to use pre-trained weights from imitation learning, add the arguments `--jump_start 1 --model_init $path_to_policy_weights$`

# Citation
If you use our code in your research, please cite our paper.
```
@ARTICLE{pfeiffer2018ral,
author={M. Pfeiffer and S. Shukla and M. Turchetta and C. Cadena Lerma and A. Krause and R. Siegwart and J. Nieto},
journal={IEEE Robotics and Automation Letters},
title={{Reinforced Imitation: Sample Efficient Deep Reinforcement Learning for Map-less Navigation by Leveraging Prior Demonstrations}},
year={2018},
volume={3},
number={4},
pages={4423-4430}
}
```

# References
Our training model uses Constrained Policy Optimization : \[[Paper](https://arxiv.org/abs/1705.10528)\] \[[Code](https://github.com/jachiam/cpo)\]
