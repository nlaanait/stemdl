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

layer_keys_list = ['conv1', 'conv2', 'pool1', 'conv3', 'conv4', 'conv5', 'conv6', 'pool2',
                   'conv7', 'conv8', 'conv9', 'conv10', 'pool3', 'conv11','conv12', 'conv13', 'conv14', 'pool4',
                   'fc_1', 'linear_output']

# parameters dictionary
conv_layer_1 = OrderedDict({'type': 'convolutional', 'stride': [1, 1], 'kernel': [10, 10], 'features': 64,
                            'activation':'relu', 'padding':'SAME'})
conv_layer_2 = OrderedDict({'type': 'convolutional', 'stride': [1, 1], 'kernel': [4, 4], 'features': 128,
                            'activation':'relu', 'padding':'SAME'})
conv_layer_3 = OrderedDict({'type': 'convolutional', 'stride': [1, 1], 'kernel': [4, 4], 'features': 256,
                            'activation':'relu', 'padding':'SAME'})
conv_layer_4 = OrderedDict({'type': 'convolutional', 'stride': [1, 1], 'kernel': [4, 4], 'features': 512,
                            'activation':'relu', 'padding':'SAME'})
pool_avg = OrderedDict({'type': 'pooling', 'stride': [2, 2], 'kernel': [2, 2], 'pool_type': 'avg','padding':'SAME'})
pool_max = OrderedDict({'type': 'pooling', 'stride': [2, 2], 'kernel': [2, 2], 'pool_type': 'max','padding':'SAME'})
conv_layer_3 = OrderedDict({'type': 'convolutional', 'stride': [1, 1], 'kernel': [2, 2], 'features': 64,
                            'activation':'relu', 'padding':'SAME'})
fully_connected = OrderedDict({'type': 'fully_connected','weights': 1000,'bias': 1000, 'activation': 'tanh',
                               'regularize': True})
linear_ouput = OrderedDict({'type': 'linear_output','weights': 3,'bias': 3,'regularize': False})

layer_params_list = [conv_layer_1]*2 + [pool_avg] + [conv_layer_2]*4 + [pool_avg] + [conv_layer_3]*4 + [pool_avg] + \
                     [conv_layer_4]*4 + [pool_avg] + [fully_connected] + [linear_ouput]

################
# VGG-16-like #
##############

# layer_keys_list = ['conv1', 'conv2', 'pool1', 'conv3', 'conv4','pool2', 'conv5','conv6', 'conv7','conv8','pool3', 'conv9', 'conv10','conv11','conv12', 'pool3','conv13', 'conv14', 'conv15', 'conv16','pool4', 'fc_1','fc_2', 'fc_3', 'linear_output']
# conv_1 =OrderedDict({'type': 'convolutional', 'stride': [1, 1], 'kernel': [3, 3], 'features': 64,'activation':'relu', 'padding':'SAME'})
# conv_2 =OrderedDict({'type': 'convolutional', 'stride': [1, 1], 'kernel': [3, 3], 'features': 128,'activation':'relu', 'padding':'SAME'})
# conv_3 =OrderedDict({'type': 'convolutional', 'stride': [1, 1], 'kernel': [3, 3], 'features': 256,'activation':'relu', 'padding':'SAME'})
# conv_4 =OrderedDict({'type': 'convolutional', 'stride': [1, 1], 'kernel': [3, 3], 'features': 512,'activation':'relu', 'padding':'SAME'})
# pool = OrderedDict({'type': 'pooling', 'stride': [2, 2], 'kernel': [2, 2], 'pool_type': 'max','padding':'SAME'})
# fully_connected_1 = OrderedDict({'type': 'fully_connected','weights': 4096,'bias': 4096, 'activation': 'relu','regularize': False})
# fully_connected_2 = OrderedDict({'type': 'fully_connected','weights': 1000,'bias': 1000, 'activation': 'relu','regularize': True})
# linear_output = OrderedDict({'type': 'linear_output','weights': 3,'bias': 3,'regularize': False})
# layer_params_list = [conv_1]*2 +[pool] + [conv_2]*2 + [pool] + [conv_3]*4 + [pool] + [conv_4]*4 + [pool] + [conv_4]*4 + [pool] + [fully_connected_1]*2 + [fully_connected_2] + [linear_output]


#################################
# templates for hyper-parameters #
#################################

# Regression
hyper_params_regression = {'network_type': 'regressor', #'network_type': 'classifier'
                           'optimization': 'ADAM', #'optimization': 'SGD'
                           'warm_up': False,
                           'num_epochs_per_decay':3,
                           'learning_rate_decay_factor': 0.5,
                           'initial_learning_rate': 0.001,
                           'num_epochs_per_ramp': 10,
                           'num_epochs_in_warm_up': 100,
                           'warm_up_max_learning_rate': 1e-3,
                           'weight_decay': 1.e-4,
                           'moving_average_decay': 0.9999,
                           'loss_function': {'type': 'Huber',
                                             'residual_num_epochs_decay': 3,
                                             'residual_initial': 5.0,
                                             'residual_minimum': 1.0,
                                             'residual_decay_factor': 0.75},
                           }

# Classification
#TODO

if __name__ == '__main__':
    io_utils.write_json_network_config('network_regressor_custom.json', layer_keys_list, layer_params_list)
    _ = io_utils.load_json_network_config('network_regressor_custom.json')
    io_utils.write_json_hyper_params('hyper_params_regressor_custom.json', hyper_params_regression)
    _ = io_utils.load_json_hyper_params('hyper_params_regressor_custom.json')
