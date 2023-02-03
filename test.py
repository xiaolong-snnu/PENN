from train import test
from data import sample_data

if __name__ == '__main__':
    # Step 1: select one config file and the corresponding model file
    ################################
    # The OU system
    ################################
    from config_ou import Config, sampling_func  # train the OU process
    config = Config()
    config.param['test_model_file'] = './data/ou/model_00988.ckpt'

    ################################
    # The genetic toggle switch system
    ################################
    # from config_gene_switch import Config, sampling_func  # train the genetic toggle switch system
    # config = Config()
    # config.param['test_model_file'] = './data/gene_switch/model_00986.ckpt'

    ################################
    # The duffing system
    ################################
    # from config_duffing import Config, sampling_func  # train the Duffing system
    # config = Config()
    # config.param['test_model_file'] = './data/duffing/model_00475.ckpt'

    # Step 2: Sample the test data.
    # Comment out this line if the data has been generated.
    sample_data(config, sampling_func, mode='test')

    # Step 3: Test
    test(config)





