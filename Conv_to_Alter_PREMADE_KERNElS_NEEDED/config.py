config = {
    'dataset': 'mnist', #choices=['mnist', 'mnist20x20', 'mnist_bin', 'mnist20x20_bin'])
    'model': 'both',    #choices=['regular', 'conv', 'both']
    'storage_root': str(ROOT / 'dataset_storage'),
    'batch_size': 128,
    'train_steps': 5000,
    'seed': 0,

    'learning_rate': 0.01,
    'weight_decay': 1e-4,
    'clip_value': 1.0,
    'sum_tau': 1.0,
    'gumb_tau': 1.0,
    'dirichlet_concentration': 1.0

    'regular_neurons': 1200,
    'regular_layers': 5,

    'regular_logic_family': 'full', #choices=['full', 'light']
    'regular_architecture': 'softmax',

    'conv_channels': 32,
    'conv_depth': 3,
    'conv_kernel_size': 3,
    'conv_stride': 1,
    'conv_head_neurons': 1020,
    'conv_head_layers': 3,

    'conv_logic_family': 'full', #choices=['full', 'light']
    'conv_architecture': 'softmax',

    'pool_size': 2,
    'pool_stride': 2,

}