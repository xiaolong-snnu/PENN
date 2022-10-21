# PENN
Parameter estimation neural network for alpha-stable Lévy noise driven SDEs

# Introduction

Parameter estimation neural network (PENN) is a data-driven method that can estimate the governing parameters from a short sample trajectory of an alpha-stable Lévy noise-driven stochastic differential equation (SDE), including the system parameters and the parameters of the Lévy noise (noise intensity and the stability index). Please see the following article for the full detail.

Wang, Xiaolong, Jing Feng, Qi Liu, Yongge Li and Yong Xu. Neural network-based parameter estimation of stochastic differential equations driven by Lévy noise. Physica A: Statistical Mechanics and its Applications 606 (2022): 128146. https://doi.org/10.1016/j.physa.2022.128146

# Installation

The code was sucessfully tested on Ubuntu 18.04+Anaconda+CUDA 10.2+PyTorch 1.5.0 or 1.8.1 with a RTX 2080Ti GPU. An error occurred when we tested it on an environment using CUDA 11. Please install a conda environment with PyTorch and run the following commmands for generating Lévy noise and recording logs. 

pip install signalz 

pip install tensorboard

# Usage

1. test

Run test.py to generate a few sample trajectories of the Ornstein-Uhlenbeck process with uniformly sampled parameters and see the estimation results on a pre-trained model named model_00988.ckpt.

2. training

Run train.py to train a new model.

3. Modification

Go to config.py to change setting. The model model_00988.ckpt was trained using 'train_num'=600,000 trajectories for 'num_epochs' = 988 epochs.
