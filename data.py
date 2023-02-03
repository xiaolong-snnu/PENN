import numpy as np
import matplotlib.pyplot as plt
import signalz as signalz
import pickle
import time
import os

def prepare_samples_from_list(files, max_sample=100000):
    xs = []
    ys = []
    hs = []
    count = 0
    for name in files:
        with open(name, 'rb') as f:
            print('loading', name)
            x, h, y = pickle.load(f)
            xs.extend(x)
            ys.extend(y)
            hs.extend(h)
            count = count + len(x)
        if count >=max_sample:
            break
    if len(xs) > max_sample:
        xs = xs[:max_sample]
        ys = ys[:max_sample]
        hs = hs[:max_sample]
    dic = {'X': xs, 'Y': ys, 'H': hs}  # sample, parameter, time span
    print(len(xs), 'samples prepared')
    return dic

def generate_param(param, n, is_int=False):
    if isinstance(param, list):
        if is_int:
            a = np.random.randint(param[0], param[1], size=n)
        else:
            a = np.random.uniform(param[0], param[1], n)
    else:
        a = np.array([param] * n)
    return a

def _sample_data(save_name, trajectory_number, param_dic, func, samples_per_temp_file=10000):
    path = os.path.dirname(save_name)
    if not os.path.exists(path):
        os.makedirs(path)
    raw_data_format = os.path.join(path, 'temp_%03d.pkl')
    data_list = []
    K = int(np.ceil(trajectory_number / samples_per_temp_file))
    # As the data generation may be slow, many files will be generated
    # and finally they are assembled into one file.
    for k in range(K):
        if k < (K - 1):
            M = samples_per_temp_file
        else:
            M = trajectory_number - (K - 1) * samples_per_temp_file
        print('generating', k + 1, 'of', K)
        file_name = raw_data_format % k
        data_list.append(file_name)
        # Do not repeatedly generate files if they are already existed.
        if not os.path.exists(file_name):
            func(M=M, **param_dic, save_name=file_name)
    dic = prepare_samples_from_list(data_list, max_sample=trajectory_number)
    with open(save_name, 'wb') as f:
        pickle.dump([dic, param_dic], f)
    print('samples saved in:', save_name)
    for _ in data_list:
        os.remove(_)

def sample_data(config, sampling_func, mode='train'):
    print('mode', mode)
    param_dic = {i: config.param[i] for i in config.param['param_name']}
    param_dic['N'] = config.param['N']
    param_dic['T'] = config.param['T']
    if mode == 'train':
        # Generating the training dataset and the evaluation dataset
        _sample_data(config.param['train_file'], config.param['train_num'], param_dic, sampling_func)
        _sample_data(config.param['eval_file'], config.param['eval_num'], param_dic, sampling_func)
    elif mode == 'test':
        # Generating only the the test dataset
        _sample_data(config.param['test_file'], config.param['test_num'], param_dic, sampling_func)
    elif mode == 'eval':
        # Generating only the the eval dataset
        _sample_data(config.param['eval_file'], config.param['eval_num'], param_dic, sampling_func)

if __name__ == '__main__':
    pass