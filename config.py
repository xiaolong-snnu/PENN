import os

class Config:
    def __init__(self, system_name='ou', data_root='./data'):
        self.system_name = system_name
        self.data_root = data_root
        self.save_path = self.data_root + '/' + system_name
        if self.system_name not in ['ou']:
            raise RuntimeError('Unknown model')
        if self.system_name == 'ou':
            self.param = self.init_ornstein_uhlenbeck()

    def init_ornstein_uhlenbeck(self):
        sample = {
            'eta': [0, 5],
            'sigma': [0, 0.05],
            'alpha': [1.01, 2],
            'N': [100, 400],
            'T': [5, 15],

            'train_num': 5000,
            'eval_num': 500,
            'test_num': 2000,
            'trai_file': self.save_path + '/ou_01_train_debug.pkl',
            'eval_file': self.save_path + '/ou_01_eval_debug.pkl',
            'test_file': self.save_path + '/gou_01_test_debug.pkl',
        }

        train = {
            # '' for start a new train,
            # put the path of the model file to continue a new train
            'init_weight_file': '',
            'num_epochs': 1000,
            'num_epochs': 5,  # just for debugging
            'learning_rate': 0.001,  # in the ADAM optimizer
            'batch_size': 1600,  # if out of memory occurs, decrease this

            # The architecture of the PENN
            'lstm_layers': 4,
            'lstm_fea_dim': 25,
            'activation': 'elu',
            'drop_last': False,  #
            'loss_weight': [3, 20, 1],  # the weight in the loss function for every parameter.
            'param_name': ['eta', 'epsilon', 'alpha'],  # parameter names for plotting
            'param_name_latex': ['$\\eta$', '$\\epsilon$', '$\\alpha$'],
            'architecture_name': 'ou_model_1'  # change this to create a new folder for model files
        }

        test = {
            'last_epoch': False,
            'test_epoch': 4
        }
        return self.handle_param(sample, train, test)

    def handle_param(self, sample, train, test):
        fixed_alpha = not isinstance(sample['alpha'], list)
        train['fixed_alpha'] = fixed_alpha
        train['model_path'] = os.path.join(self.save_path, train['architecture_name'])
        if train['fixed_alpha']:
            train['param_name'] = train['param_name'][:-1]  # delete alpha
        if test['last_epoch']:
            model_folder = os.path.join(train['model_path'], 'model')
            model_files = os.listdir(model_folder)
            model_files.sort()
            test['test_model_file'] = os.path.join(model_folder, model_files[-1])
            print(test['test_model_file'])
        else:
            test['test_model_file'] = os.path.join(train['model_path'], 'model/model_%05d.ckpt' % test['test_epoch'])
        return {'sample': sample, 'train': train, 'test': test}