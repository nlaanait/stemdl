sys.path.append('../')
from stemdl import runtime


def get_space():
    space = [
        (1e-4, 0.1), # initial_learning_rate
        (0.0, 0.9),  # weight_decay
        (1e-4, 0.1)  # LARC_eta
    ]
    return space


def objective(hparams, network_config, hyper_params, params):
    """Objective function for hyperprameter optimization.
    
    Parameters
    ----------
    hparams : list
        Next settings of each hyperparameter.
        Controlled by Hyperspace.

    hyper_params : dict
        All Hyperparameters for stemdl

    params : dict
        Stemdl params
    """
    initial_learning_rate, weight_decay, LARC_eta = hparams

    hyper_params['initial_learning_rate'] = initial_learning_rate
    hyper_params['weight_decay'] = weight_decay
    hyper_params['LARC_eta'] = LARC_eta

    loss = runtime.train(network_config, hyper_params, params, hyper_optimization=True)
    return loss