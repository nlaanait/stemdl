"""
Created on 10/9/17.
@author: Numan Laanait.
email: laanaitn@ornl.gov
"""

from collections import OrderedDict
import json
from stemdl import io_utils


#################################
# templates for network_config  #
#################################

layer_keys_list = ['conv1', 'conv2', 'conv3', 'conv4', 'pool1', 'conv5', 'conv6', 'conv7', 'conv8',
                   'pool2', 'conv9', 'conv10', 'conv11', 'conv12', 'pool3', 'fc_1', 'linear_output']

# parameters dictionary
conv_layer_1 = OrderedDict({'type': 'convolutional', 'stride': [1, 1], 'kernel': [4, 4], 'features': 64,
                            'activation':'relu', 'padding':'SAME'})
conv_layer_2 = OrderedDict({'type': 'convolutional', 'stride': [1, 1], 'kernel': [4, 4], 'features': 128,
                            'activation':'relu', 'padding':'SAME'})
pool_avg = OrderedDict({'type': 'pooling', 'stride': [2, 2], 'kernel': [2, 2], 'pool_type': 'avg','padding':'SAME'})
pool_max = OrderedDict({'type': 'pooling', 'stride': [2, 2], 'kernel': [2, 2], 'pool_type': 'max','padding':'SAME'})
conv_layer_3 = OrderedDict({'type': 'convolutional', 'stride': [1, 1], 'kernel': [4, 4], 'features': 256,
                            'activation':'relu', 'padding':'SAME'})
fully_connected = OrderedDict({'type': 'fully_connected','weights': 1024,'bias': 1024, 'activation': 'relu',
                               'regularize': True})
linear_ouput = OrderedDict({'type': 'linear_output','weights': 3,'bias': 3,'regularize': False})

layer_params_list = [conv_layer_1]*4 + [pool_max] + [conv_layer_2]*4 + [pool_max] + [conv_layer_3]*4 + [pool_avg] + \
                    [fully_connected] + [linear_ouput]

#################################
# templates for hyper-parameters #
#################################

# Regression
hyper_params_regression = {'network_type': 'regressor', #'network_type': 'classifier'
                           'optimization': 'SGD', #'optimization': 'SGD'
                           'warm_up': False,
                           'num_epochs_per_decay':30,
                           'learning_rate_decay_factor': 0.5,
                           'initial_learning_rate': 1.e-3,
                           'num_epochs_per_ramp': 10,
                           'num_epochs_in_warm_up': 100,
                           'warm_up_max_learning_rate': 1e-3,
                           'weight_decay': 1.e-4,
                           'moving_average_decay': 0.9999,
                           'loss_function': {'type': 'MSE',
                                             'residual_num_epochs_decay': 30,
                                             'residual_initial': 5.0,
                                             'residual_minimum': 1.0,
                                             'residual_decay_factor': 0.75},
                           }

# Classification
#TODO

if __name__ == '__main__':
    io_utils.write_json_network_config('network_regressor.json', layer_keys_list, layer_params_list)
    _ = io_utils.load_json_network_config('network_regressor.json')
    io_utils.write_json_hyper_params('hyper_params_regressor.json', hyper_params_regression)
    _ = io_utils.load_json_hyper_params('hyper_params_regressor.json')

