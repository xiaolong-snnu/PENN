import numpy as np
import matplotlib.pyplot as plt
import signalz as signalz
import pickle
import time
import os

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

def gen_samples_duffing(M=1000, gamma=[0.5, 1], epsilon=[0.05, 0.5], alpha=[1.4, 2], N=[1000, 1500], T=[50, 100], save_name='duffing_1d.pkl', save_y=False):
    discard_T = 50
    xs = []
    ys = []
    MM = 1000
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
        if (id+1) % MM == 0:
            time_end = time.time()
            time_used = time_end - time_start
            time_left = time_used / (id + 1) * (M - id)
            print(id+1, 'of', M, '%.1fs used %.1fs left' % (time_used, time_left),
                  'discard %.02f%%' % (count_discard / (id + 1 + count_discard) * 100))
        _N = int(discard_N[id])
        _N2 = int(_N + N[id])
        while 1:
            good, xy = trajectory_1d_duffing(_N2, dt[id], gamma[id], epsilon[id], alpha[id])
            if good:
                break
            else:
                count_discard = count_discard + 1
        if save_y:
            xs.append(xy[_N:])
        else:
            x = xy[0, _N:]
            xs.append(np.array([x]).T)
        ys.append([gamma[id], epsilon[id], alpha[id]])
    with open(save_name, 'wb') as f:
        pickle.dump([xs, T[:len(xs)].tolist(), ys], f)

def gen_samples_gene_switch(M=1000, r=[2, 10], k=[2, 20], epsilon=[0.05, 0.5], alpha=[1.4, 2], N=[500, 1000], T=[50, 100], save_name='gene_switch_1d.pkl'):
    discard_T = 50
    xs = []
    ys = []
    MM = 1000
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
        if (id+1) % MM == 0:
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

def trajectory_1d_ou(N, dt, mu, eta, epsilon, alpha, x0=0):
    time_ratio = np.power(dt, 1.0 / alpha)
    dL = signalz.levy_noise(N, alpha, 0.0, 1.0, 0.)
    data = np.zeros(N)
    data[0] = x0
    for i in range(N - 1):
        data[i + 1] = data[i] - eta * (data[i] - mu) * dt + epsilon * time_ratio * dL[i]
    return data

def gen_samples_ou(M=1000, eta=[0, 5], epsilon=[0, 0.05], alpha=[1.2, 2], N=[100, 400], T=[5, 15], save_name='ou_1d.pkl'):
    discard_T = 10
    xs = []
    ys = []
    MM = 2000
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
        if (id+1) % MM == 0:
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

def prepare_samples_from_list(files, max_sample=100000):
    print('max samples =', max_sample)
    xs = []
    ys = []
    hs = []
    count = 0
    for name in files:
        with open(name, 'rb') as f:
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

def _sample_data(config, save_name, train_num, param_dic, func, samples_per_temp_file=50000):
    path = os.path.dirname(save_name)
    if not os.path.exists(path):
        os.makedirs(path)
    raw_data_format = os.path.join(path, 'temp_%03d.pkl')
    data_list = os.path.join(path, 'temp_list.txt')
    if 1:
        K = int(np.ceil(train_num / samples_per_temp_file))
        with open(data_list, 'w') as f:
            for k in range(K):
                if k < (K - 1):
                    M = samples_per_temp_file
                else:
                    M = train_num - (K - 1) * samples_per_temp_file
                print('=' * 30, k + 1, 'of', K, '=' * 30)
                file_name = raw_data_format % k
                func(M=M, **param_dic, save_name=file_name)
                f.write('%s\n' % file_name)
    with open(data_list, 'r') as f:
        file_list = f.readlines()
        file_list = [f.strip() for f in file_list]
        print(len(file_list), 'files loaded.')
    dic = prepare_samples_from_list(file_list, max_sample=train_num)
    with open(save_name, 'wb') as f:
        pickle.dump([dic, config], f)
    print('samples saved in:', save_name)

def sample_data(_config, mode='train', generating=True):
    if not generating:
        return
    config = _config.param['sample']
    print('mode', mode)
    sample_funcs = {
        # If you want to add new SDEs, extend the following lines
        'duffing': [gen_samples_duffing, ['gamma', 'epsilon', 'alpha', 'N', 'T']],
        'ou': [gen_samples_ou, ['eta', 'epsilon', 'alpha', 'N', 'T']],
        'gene_switch': [gen_samples_gene_switch, ['r', 'k', 'epsilon', 'alpha', 'N', 'T']],
    }
    if _config.system_name in sample_funcs:
        func = sample_funcs[_config.system_name][0]
        param_dic = {i: config[i] for i in sample_funcs[_config.system_name][1]}
        if mode == 'train':
            _sample_data(config, config['trai_file'], config['train_num'], param_dic, func)
            _sample_data(config, config['eval_file'], config['eval_num'], param_dic, func)
        elif mode == 'test':
            _sample_data(config, config['test_file'], config['test_num'], param_dic, func)
        elif mode == 'eval':
            _sample_data(config, config['eval_file'], config['eval_num'], param_dic, func)
    else:
        raise RuntimeError('Unknown model')

if __name__ == '__main__':
    pass