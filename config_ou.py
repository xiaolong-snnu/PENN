import os
import signalz as signalz
import numpy as np
from data import generate_param
import time
import pickle

class Config:
    def __init__(self):
        # The path:
        # data_root/system_name
        # data_root/system_name/model -- for saving model weights
        # data_root/system_name/runs -- for saving tensorboard logs

        self.data_root = './data'
        self.system_name = 'ou'
        self.save_path = os.path.join(self.data_root, self.system_name)

        param_dic = {
            #####################################
            # Parameters for sampling
            #####################################

            # Variable parameter ranges of the system
            'eta': [0, 5],
            'epsilon': [0, 0.05],
            'alpha': [1.01, 2],
            'param_name': ['eta', 'epsilon', 'alpha'],  # Parameter names for sampling
            'param_name_latex': ['$\\eta$', '$\\epsilon$', '$\\alpha$'],  # Parameter names for plotting

            # The lengths of trajectories (integers)
            'N': [100, 400],

            # the time spans
            'T': [5, 15],

            # Numbers and file names of trajectories for training, evaluation and test, respectively.
            # More than 100,000 trajectories are recommended to train a good model.
            # 5000 is too small. It is only for debugging.
            'train_num': 5000,
            'eval_num': 500,
            'test_num': 2000,

            'train_file': os.path.join(self.save_path, 'ou_01_train.pkl'),
            'eval_file': os.path.join(self.save_path, 'ou_01_eval.pkl'),
            'test_file': os.path.join(self.save_path, 'ou_01_test.pkl'),

            #####################################
            # Parameters for training
            #####################################

            # init_weight_file = '' for starting a new train, or
            # put a model file init_weight_file = './data_root/system_name/architecture_name/model/model_#####.ckpt'
            # to continue an old training. The epoch number will be extracted from the name 'model_#####.ckpt'

            'init_weight_file': '',

            # The number of epochs  # 1000 is recommanded. Please check the tensorboard files to monitor the training.
            # the command is tensorboard --logdir=./data_root/system_name/architecture_name/runs
            'num_epochs': 100,

            'learning_rate': 0.001,  # Used in the ADAM optimizer

            # If the error 'out of memory' occurs, decrease the batch_size.
            # 1600 is for the 2080Ti GPU with 11GB memory.
            'batch_size': 1600,

            # The architecture of the PENN
            'lstm_layers': 4,
            'lstm_fea_dim': 25,  # The dimension of the LSTM features
            'activation': 'elu',  # A slightly better than ReLU and atan
            'drop_last': True,  # Throw away the last mini batch in each epoch

            # The weight in the loss function for every parameter.
            # It should be inversely proportional to the training range.
            # You can also try to increase the weights of parameters that are
            # hard to learn, i.e., increase the penalty of errors.
            'loss_weight': [3, 20, 1],

            'architecture_name': 'ou_model_1'  # Change this to create a new folder to save models and logs
        }
        self.param = param_dic

def sampling_func(M=1000, eta=[0, 5], epsilon=[0, 0.05], alpha=[1.2, 2], N=[100, 400], T=[5, 15], save_name='ou_1d.pkl'):
    discard_T = 10  # discard transient states spanning [0, discard_T]
    xs = []
    ys = []
    NUM_FOR_SHOW = 2000
    N = generate_param(N, M)
    T = generate_param(T, M)
    dt = T / N
    discard_T = generate_param(discard_T, M)
    discard_N = discard_T / dt
    eta = generate_param(eta, M)
    epsilon = generate_param(epsilon, M)
    alpha = generate_param(alpha, M)
    time_start = time.time()
    for id in range(M):
        if (id+1) % NUM_FOR_SHOW == 0:
            time_end = time.time()
            time_used = time_end - time_start
            time_left = time_used / (id + 1) * (M - id)
            print(id+1, 'of', M, '%.1fs used %.1fs left' % (time_used, time_left))
        _N = int(discard_N[id])
        _N2 = int(_N + N[id])
        x = trajectory_1d_ou(_N2, dt[id], 0.0, eta[id], epsilon[id], alpha[id], x0=0)
        x = x[_N:]
        xs.append(np.array([x]).T)
        ys.append([eta[id], epsilon[id], alpha[id]])
    with open(save_name, 'wb') as f:
        pickle.dump([xs, T[:len(xs)].tolist(), ys], f)

def trajectory_1d_ou(N, dt, mu, eta, epsilon, alpha, x0=0):
    time_ratio = np.power(dt, 1.0 / alpha)
    dL = signalz.levy_noise(N, alpha, 0.0, 1.0, 0.)  # Generating alpha-stable Levy random variable
    data = np.zeros(N)
    data[0] = x0
    for i in range(N - 1):
        data[i + 1] = data[i] - eta * (data[i] - mu) * dt + epsilon * time_ratio * dL[i]
    return data