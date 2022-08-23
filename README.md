# PENN
Parameter estimation neural network for alpha-stable Lévy noise driven SDEs

# Introduction

Parameter estimation neural network (PENN) is a data-driven method that can estimate the governing parameters from a short sample trajectory of an alpha-stable Lévy noise-driven stochastic differential equation (SDE), including the system parameters and the parameters of the Lévy noise (noise intensity and the stability index). Please see the article 'Neural network-based parameter estimation of stochastic differential equations driven by Lévy noise' for more detail. It is being under review.

# Installation

Pytorch should be installed first. Then download the code and run the following command to install necessary dependence.

pip install signalz 

pip install tensorboard

# Usage

1. test

Run the test.py to generate a few sample trajectories of the Ornstein-Uhlenbeck process with uniformly sampled parameters and see the estimation results on a pre-trained model named model_00988.ckpt.

2. training

Run the train.py to train a new model.

3. Modification

Go to the config.py to change setting. The model model_00988.ckpt was trained using 'train_num'=600,000 trajectories for 'num_epochs' = 988 epochs.
