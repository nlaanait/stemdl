"""
Created on 12/15/17.
@author: Suhas Somnath
"""

import tensorflow as tf
import sys
import os
import horovod.tensorflow as hvd

sys.path.append('../')
from stemdl import runtime
from stemdl import io_utils


# NOTE because of summitdev/container problems, we can't pass any flags
# whatsoever, so we have to hard code the name of parameters file
from json_flags import JSON_FLAGS, num_gpus


def main():
    # initiate horovod
    hvd.init()

    # read all parameters to a python dictionary
    params = io_utils.get_dict_from_json(JSON_FLAGS)
    # update the number of GPUs from the python file:
    params['num_gpus'] = num_gpus

    checkpt_dir = params['checkpt_dir']
    # Also need a directory within the checkpoint dir for event files coming from eval
    eval_dir = os.path.join(checkpt_dir, '_eval')

    # Create directories
    if tf.gfile.Exists(checkpt_dir):
        print('Directory "%s" exists already.\nReloading model from latest checkpoint.' % format(checkpt_dir))
    elif tf.gfile.Exists(eval_dir):
        print('Removing "%s"' % format(eval_dir))
        tf.gfile.DeleteRecursively(eval_dir)
    else:
        tf.gfile.MakeDirs(checkpt_dir)
        tf.gfile.MakeDirs(eval_dir)

    params['train_dir'] = checkpt_dir
    params['eval_dir'] = eval_dir

    # load network config file and hyper_parameters
    network_config = io_utils.load_json_network_config(params['network_config'])
    hyper_params = io_utils.load_json_hyper_params(params['hyper_params'])

    # train or evaluate
    if params['mode'] == 'train':
        runtime.train_horovod_mod(network_config, hyper_params, params['data_dir'], params, num_GPUS=params['num_gpus'])
    else:
        runtime.eval(network_config, hyper_params, params['data_dir'], params, num_GPUS=params['num_gpus'])


if __name__ == '__main__':
    main()
