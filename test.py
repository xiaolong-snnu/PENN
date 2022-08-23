from train import test
from config import Config
if __name__ == '__main__':
    system = 'ou'
    config = Config(system_name=system)

    # load the latest model file to test or
    # use the following line to specify the model file

    config.param['test']['test_model_file'] = './data/ou/model_00988.ckpt'
    test(config, sample_generating=True)









