import os
import signalz as signalz
import numpy as np
from data import generate_param
import time
import pickle

class Config:
    def __init__(self):
        self.data_root = './data'
        self.system_name = 'gene_switch'
        self.save_path = os.path.join(self.data_root, self.system_name)

        param_dic = {
            #####################################
            # Parameters for sampling
            #####################################

            'r': [2, 10],
            'k': [2, 20],
            'epsilon': [0.02, 0.1],
            'alpha': [1.2, 2],
            'N': [500, 1000],
            'T': [50, 100],
            'train_num': 200000,
            'eval_num': 1000,
            'test_num': 1000,
            'train_file': os.path.join(self.save_path, 'gene_switch_train.pkl'),
            'eval_file': os.path.join(self.save_path, 'gene_switch_eval.pkl'),
            'test_file': os.path.join(self.save_path, 'gene_switch_test.pkl'),
            'param_name': ['r', 'k', 'epsilon', 'alpha'],
            'param_name_latex': ['$r$', '$k$', '$\\epsilon$', '$\\alpha$'],

            #####################################
            # Parameters for training
            #####################################
            'init_weight_file': '',
            'num_epochs': 1000,
            'learning_rate': 0.001,
            'batch_size': 1600,
            'lstm_layers': 4,
            'lstm_fea_dim': 25,
            'activation': 'elu',
            'drop_last': True,
            'loss_weight': [0.1, 0.1, 30.0, 1],
            'architecture_name': 'gene_switch_model_1'
        }
        self.param = param_dic

def sampling_func(M=1000, r=[2, 10], k=[2, 20], epsilon=[0.05, 0.5], alpha=[1.4, 2], N=[500, 1000], T=[50, 100], save_name='gene_switch_1d.pkl'):
    discard_T = 50
    xs = []
    ys = []
    NUM_FOR_SHOW = 1000
    N = generate_param(N, M)
    T = generate_param(T, M)
    dt = T / N
    discard_T = generate_param(discard_T, M)
    discard_N = discard_T / dt
    r = generate_param(r, M)
    k = generate_param(k, M)
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
        x0 = np.random.rand()
        x = trajectory_1d_gene_switch(_N2, dt[id], r[id], k[id], epsilon[id], alpha[id], x0)
        x = x[_N:]
        xs.append(np.array([x]).T)
        ys.append([r[id], k[id], epsilon[id], alpha[id]])
    with open(save_name, 'wb') as f:
        pickle.dump([xs, T[:len(xs)].tolist(), ys], f)

def func_gene_switch(x):
    return (2.0 * x ** 2 + 50.0 * x ** 4) / (25 + 29 * x ** 2 + 52 * x ** 4 + 4 * x ** 6)

def trajectory_1d_gene_switch(N, dt, r, k, epsilon, alpha, x0=0):
    time_ratio = np.power(dt, 1.0 / alpha)
    dL = signalz.levy_noise(N, alpha, 0.0, 1.0, 0.)
    data = np.zeros(N)
    data[0] = x0
    for i in range(N - 1):
        x = data[i]
        data[i + 1] = x + (func_gene_switch(x) * k - r * x + 1) * dt + epsilon * time_ratio * dL[i]
    return data