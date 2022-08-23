import numpy as np
import matplotlib.pyplot as plt
import signalz as signalz
import pickle
import time
import os

def trajectory_1d_ou(N, dt, mu, eta, sigma, alpha, x0=0):
    time_ratio = np.power(dt, 1.0 / alpha)
    dL = signalz.levy_noise(N, alpha, 0.0, 1.0, 0.)
    data = np.zeros(N)
    data[0] = x0
    for i in range(N - 1):
        data[i + 1] = data[i] - eta * (data[i] - mu) * dt + sigma * time_ratio * dL[i]
    return data

def gen_samples_ou_varying_h(M=1000, eta=[0, 5], sigma=[0, 0.05], alpha=[1.2, 2], N=[100, 400], T=[5, 15], save_name='ou_1d.pkl'):
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
    sigma = generate_param(sigma, M)
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
        x = trajectory_1d_ou(_N2, dt[id], 0.0, eta[id], sigma[id], alpha[id], x0=0)
        x = x[_N:]
        xs.append(np.array([x]).T)
        ys.append([eta[id], sigma[id], alpha[id]])
    with open(save_name, 'wb') as f:
        pickle.dump([xs, T[:len(xs)].tolist(), ys], f)

def prepare_samples_from_list_h_duffing(files, max_sample=100000):
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

def sample_ou(config, save_name, train_num, generating_raw_data=True):
    eta = config['eta']
    sigma = config['sigma']
    alpha = config['alpha']
    N = config['N']
    T = config['T']
    samples_per_temp_file = 100000
    path = os.path.dirname(save_name)
    if not os.path.exists(path):
        os.makedirs(path)
    raw_data_format = os.path.join(path, 'temp_%03d.pkl')
    data_list = os.path.join(path, 'temp_list.txt')
    if generating_raw_data:
        K = int(np.ceil(train_num / samples_per_temp_file))
        with open(data_list, 'w') as f:
            for k in range(K):
                if k < (K - 1):
                    M = samples_per_temp_file
                else:
                    M = train_num - (K - 1) * samples_per_temp_file
                print('=' * 30, k + 1, 'of', K, '=' * 30)
                file_name = raw_data_format % k
                gen_samples_ou_varying_h(M=M, eta=eta, sigma=sigma, alpha=alpha, N=N, T=T, save_name=file_name)
                f.write('%s\n' % file_name)
    with open(data_list, 'r') as f:
        file_list = f.readlines()
        file_list = [f.strip() for f in file_list]
        print(len(file_list), 'files loaded.')
    dic = prepare_samples_from_list_h_duffing(file_list, max_sample=train_num)
    with open(save_name, 'wb') as f:
        pickle.dump([dic, config], f)
    print('samples saved in:', save_name)

def sample_data(_config, mode='train', generating=True):
    config = _config.param['sample']
    print('mode', mode)
    if mode == 'train':
        if generating:
            if _config.system_name in ['ou', 'ou2']:
                sample_ou(config, config['trai_file'], config['train_num'])
                sample_ou(config, config['eval_file'], config['eval_num'])
    elif mode == 'test':
        if generating:
            if _config.system_name in ['ou', 'ou2']:
                sample_ou(config, config['test_file'], config['test_num'])

if __name__ == '__main__':
    pass