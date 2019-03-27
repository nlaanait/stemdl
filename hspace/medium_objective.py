import sys
sys.path.append('../')
from stemdl import runtime


def get_space():
    space = [
        (1e-4, 0.1),           # initial_learning_rate
        (0.0, 0.9),            # weight_decay
        (1e-4, 0.1),           # LARC_eta
        #('clip', 'scale'),     # LARC_mode
        (1e-5, 0.1),           # LARC_min_update
        ('Backoff', 'LogMax'), # loss_scaling
        (1e-7, 1e-4)           # LARC_epsilon
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
    hyper_params['initial_learning_rate'] = hparams[0]
    hyper_params['weight_decay'] = hparams[1] 
    hyper_params['LARC_eta'] = hparams[2]
    #hyper_params['LARC_mode'] = hparams[3]
    hyper_params['LARC_min_update'] = hparams[3]
    hyper_params['loss_scaling'] = hparams[4]
    hyper_params['LARC_epsilon'] = hparams[5]

    train_loss = runtime.train(network_config, hyper_params, params, hyper_optimization=True)
    # Sanity check
    params['mode'] = 'eval'
    params['IMAGE_FP16'] = False
    valid_loss = runtime.validate_ckpt(network_config, hyper_params, params, last_model=True, hyper_optimization=True)
    return valid_loss 
