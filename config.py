import os

class Config:
    def __init__(self, system_name='ou', data_root='./data'):
        self.system_name = system_name
        self.data_root = data_root
        self.save_path = self.data_root + '/' + system_name
        if self.system_name not in ['ou', 'duffing', 'gene_switch']:
            raise RuntimeError('Unknown model')
        if self.system_name == 'ou':
            self.param = self.init_ornstein_uhlenbeck()
        elif self.system_name == 'duffing':
            self.param = self.init_duffing()
        elif self.system_name == 'gene_switch':
            self.param = self.init_gene_switch()

    def update(self):
        self.param = self.handle_param(self.param['sample'], self.param['train'], self.param['test'])

    def init_ornstein_uhlenbeck(self):
        sample = {
            'eta': [0, 5],
            'epsilon': [0, 0.05],
            'alpha': [1.01, 2],
            'N': [100, 400],
            'T': [5, 15],

            'train_num': 5000,
            'eval_num': 500,
            'test_num': 2000,
            'trai_file': self.save_path + '/ou_01_train.pkl',
            'eval_file': self.save_path + '/ou_01_eval.pkl',
            'test_file': self.save_path + '/ou_01_test.pkl',
        }
        train = {
            # init_weight_file = '' for starting a new train, or
            # put a model file init_weight_file = 'xxx.ckpt' to continue an old train
            'init_weight_file': '',
            'num_epochs': 100,
            # 'num_epochs': ,  # Just for debugging
            'learning_rate': 0.001,  # Used in the ADAM optimizer
            'batch_size': 1600,  # If the error 'out of memory' occurs, decrease this.

            # The architecture of the PENN
            'lstm_layers': 4,
            'lstm_fea_dim': 25,
            'activation': 'elu',
            'drop_last': False,  # Throw away the last mini batch in each epoch
            # The weight in the loss function for every parameter.
            # It should be inversely proportional to the training range.
            'loss_weight': [3, 20, 1],
            'param_name': ['eta', 'epsilon', 'alpha'],  # parameter names for plotting
            'param_name_latex': ['$\\eta$', '$\\epsilon$', '$\\alpha$'],
            'architecture_name': 'ou_model_1'  # change this to create a new folder to save models and logs
        }
        test = {
            'last_epoch': False,
            'test_epoch': 4
        }
        return self.handle_param(sample, train, test)

    def init_duffing(self):
        sample = {
            'gamma': [0.5, 1],
            'epsilon': [0.05, 0.5],
            'alpha': [1.4, 2],
            'N': [1000, 1500],
            'T': [50, 100],
            'train_num': 200000,
            'eval_num': 1000,
            'test_num': 100,
            'trai_file': self.save_path + '/duffing_train.pkl',
            'eval_file': self.save_path + '/duffing_eval.pkl',
            'test_file': self.save_path + '/duffing_test.pkl',
        }
        train = {
            # 'init_weight_file': './data/duffing/duffing_600k/model/model_00109.ckpt',
            'init_weight_file': '',
            'num_epochs': 1000,
            'learning_rate': 0.001,
            'batch_size': 1100,
            'lstm_layers': 4,
            'lstm_fea_dim': 25,
            'activation': 'elu',
            'drop_last': True,
            'loss_weight': [1.0, 1.0, 1.0],
            'param_name': ['eta', 'epsilon', 'alpha'],
            'param_name_latex': ['$\\gamma$', '$\\epsilon$', '$\\alpha$'],
            'architecture_name': 'duffing_model_1'
        }
        test = {
            'last_epoch': False,
            'test_epoch': 4
        }
        return self.handle_param(sample, train, test)

    def init_gene_switch(self):
        sample = {
            'r': [2, 10],
            'k': [2, 20],
            'epsilon': [0.02, 0.1],
            'alpha': [1.2, 2],
            'N': [500, 1000],
            'T': [50, 100],
            'train_num': 200000,
            'eval_num': 1000,
            'test_num': 100,
            'trai_file': self.save_path + '/gene_switch_train.pkl',
            'eval_file': self.save_path + '/gene_switch_eval.pkl',
            'test_file': self.save_path + '/gene_switch_test.pkl',
        }
        train = {
            'init_weight_file': '',
            'num_epochs': 1000,
            'learning_rate': 0.001,
            'batch_size': 1600,
            'lstm_layers': 4,
            'lstm_fea_dim': 25,
            'activation': 'elu',
            'drop_last': False,
            'loss_weight': [0.1, 0.1, 30.0, 1],
            'param_name': ['r', 'k', 'epsilon', 'alpha'],
            'param_name_latex': ['$r$', '$k$', '$\\epsilon$', '$\\alpha$'],
            'architecture_name': 'gene_switch_model_1'
        }
        test = {
            'last_epoch': False,
            'test_epoch': 4
        }
        return self.handle_param(sample, train, test)

    def handle_param(self, sample, train, test):
        train['model_path'] = os.path.join(self.save_path, train['architecture_name'])
        if test['last_epoch']:  # Find the model file with the largest number in the 'model_folder' for the test
            model_folder = os.path.join(train['model_path'], 'model')
            model_files = os.listdir(model_folder)
            model_files.sort()
            test['test_model_file'] = os.path.join(model_folder, model_files[-1])
            print(test['test_model_file'])
        else:  # Otherwise use the specified model id for the test
            test['test_model_file'] = os.path.join(train['model_path'], 'model/model_%05d.ckpt' % test['test_epoch'])
        return {'sample': sample, 'train': train, 'test': test}