from train import test
from data import sample_data

if __name__ == '__main__':
    # Step 1: select a config file
    from config_ou import Config, sampling_func  # train the OU process
    # from config_gene_switch import Config, sampling_func  # train the genetic toggle switch system
    # from config_duffing import Config, sampling_func  # train the Duffing system
    config = Config()

    # Step 2: Sample the data.
    # Comment this line if the data has been generated.
    sample_data(config, sampling_func, mode='test')

    # Step 3: Select the model file
    config.param['test_model_file'] = './data/ou/model_00988.ckpt'

    # Step 4: Test
    test(config)



    # Another example
    # from config_gene_switch import Config, sampling_func
    # config = Config()
    # config.param['test_model_file'] = './data/gene_switch/debug/model/model_00003.ckpt'
    # config.param['test_file'] = config.param['train_file']
    # test(config)





