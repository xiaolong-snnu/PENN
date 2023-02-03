import os
import signalz as signalz
import numpy as np
from data import generate_param
import time
import pickle

class Config:
    def __init__(self):

        self.data_root = './data'
        self.system_name = 'duffing'
        self.save_path = os.path.join(self.data_root, self.system_name)

        param_dic = {
            #####################################
            # Parameters for sampling
            #####################################

            'gamma': [0.5, 1],
            'epsilon': [0.05, 0.5],
            'alpha': [1.4, 2],
            'param_name': ['gamma', 'epsilon', 'alpha'],
            'param_name_latex': ['$\\gamma$', '$\\epsilon$', '$\\alpha$'],

            'N': [1000, 1500],
            'T': [50, 100],
            'train_num': 200000,
            'eval_num': 1000,
            'test_num': 1000,
            'train_file': os.path.join(self.save_path, '/duffing_train.pkl'),
            'eval_file': os.path.join(self.save_path, '/duffing_eval.pkl'),
            'test_file': os.path.join(self.save_path, '/duffing_test.pkl'),

            #####################################
            # Parameters for training
            #####################################
            'init_weight_file': '',
            'num_epochs': 1000,
            'learning_rate': 0.001,
            'batch_size': 1100,
            'lstm_layers': 4,
            'lstm_fea_dim': 25,
            'activation': 'elu',
            'drop_last': True,
            'loss_weight': [1.0, 1.0, 1.0],
            'architecture_name': 'duffing_model_1'
        }
        self.param = param_dic

def sampling_func(M=1000, gamma=[0.5, 1], epsilon=[0.05, 0.5], alpha=[1.4, 2], N=[1000, 1500], T=[50, 100], save_name='duffing_1d.pkl', save_y=False):
    discard_T = 50
    xs = []
    ys = []
    NUM_FOR_SHOW = 1000
    N = generate_param(N, M)
    T = generate_param(T, M)
    dt = T / N
    discard_T = generate_param(discard_T, M)
    discard_N = discard_T / dt
    gamma = generate_param(gamma, M)
    epsilon = generate_param(epsilon, M)
    alpha = generate_param(alpha, M)
    time_start = time.time()
    count_discard = 0
    for id in range(M):
        if (id+1) % NUM_FOR_SHOW == 0:
            time_end = time.time()
            time_used = time_end - time_start
            time_left = time_used / (id + 1) * (M - id)
            print(id+1, 'of', M, '%.1fs used %.1fs left' % (time_used, time_left),
                  'discard %.02f%%' % (count_discard / (id + 1 + count_discard) * 100))
        _N = int(discard_N[id])
        _N2 = int(_N + N[id])
        while 1:
            # Repeatedly generating until a convergent trajectory is obtained.
            good, xy = trajectory_1d_duffing(_N2, dt[id], gamma[id], epsilon[id], alpha[id])
            if good:
                break
            else:
                count_discard = count_discard + 1
        if save_y:
            xs.append(xy[_N:])
        else:
            x = xy[0, _N:]  # only use the first dimension of the 2-D system
            xs.append(np.array([x]).T)
        ys.append([gamma[id], epsilon[id], alpha[id]])
    with open(save_name, 'wb') as f:
        pickle.dump([xs, T[:len(xs)].tolist(), ys], f)

def trajectory_1d_duffing(N, dt, gamma, epsilon, alpha):
    time_ratio = np.power(dt, 1.0 / alpha)
    dL = signalz.levy_noise(N, alpha, 0.0, 1.0, 0.)
    x0 = 0.01
    y0 = 0.01
    data = np.zeros((2, N))
    data[0] = x0
    good = True
    THR = 10000
    for i in range(N - 1):
        x = data[0, i]
        y = data[1, i]
        if abs(x) > THR or abs(y) > THR:
            good = False
            return good, []
        data[0, i + 1] = x + y * dt
        data[1, i + 1] = y + (- gamma * y + x - x ** 3) * dt + epsilon * time_ratio * dL[i]
    return good, data