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
conv_layer_1 = OrderedDict({'type': 'convolutional', 'stride': [1, 1], 'kernel': [11, 11], 'features': 64,
                            'activation':'relu', 'padding':'SAME','batch_norm':True})
conv_layer_2 = OrderedDict({'type': 'convolutional', 'stride': [1, 1], 'kernel': [5, 5], 'features': 128,
                            'activation':'relu', 'padding':'SAME', 'batch_norm':True})
conv_layer_3 = OrderedDict({'type': 'convolutional', 'stride': [1, 1], 'kernel': [5, 5], 'features': 256,
                            'activation':'relu', 'padding':'SAME', 'batch_norm':True})
conv_layer_4 = OrderedDict({'type': 'convolutional', 'stride': [1, 1], 'kernel': [5, 5], 'features': 512,
                            'activation':'relu', 'padding':'SAME', 'batch_norm':True})
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
# VGG-16 #
##############

layer_keys_list = ['conv1', 'conv2', 'pool1', 'conv3', 'conv4','pool2', 'conv5','conv6', 'conv7','conv8','pool3', 'conv9', 'conv10','conv11','conv12', 'pool3','conv13', 'conv14', 'conv15', 'conv16','pool4', 'fc_1','fc_2', 'fc_3', 'linear_output']
conv_1 =OrderedDict({'type': 'convolutional', 'stride': [1, 1], 'kernel': [3, 3], 'features': 64,'activation':'relu',
                     'padding':'SAME', 'batch_norm':False})
conv_2 =OrderedDict({'type': 'convolutional', 'stride': [1, 1], 'kernel': [3, 3], 'features': 128,'activation':'relu',
                     'padding':'SAME', 'batch_norm':False})
conv_3 =OrderedDict({'type': 'convolutional', 'stride': [1, 1], 'kernel': [3, 3], 'features': 256,'activation':'relu',
                     'padding':'SAME', 'batch_norm':False})
conv_4 =OrderedDict({'type': 'convolutional', 'stride': [1, 1], 'kernel': [3, 3], 'features': 512,'activation':'relu',
                     'padding':'SAME', 'batch_norm':False})
pool = OrderedDict({'type': 'pooling', 'stride': [2, 2], 'kernel': [2, 2], 'pool_type': 'max','padding':'SAME'})
fully_connected_1 = OrderedDict({'type': 'fully_connected','weights': 4096,'bias': 4096, 'activation': 'relu',
                                 'regularize': False})
fully_connected_2 = OrderedDict({'type': 'fully_connected','weights': 1000,'bias': 1000, 'activation': 'relu',
                                 'regularize': True})
linear_output = OrderedDict({'type': 'linear_output','weights': 3,'bias': 3,'regularize': False})
layer_params_list = [conv_1]*2 +[pool] + [conv_2]*2 + [pool] + [conv_3]*4 + [pool] + [conv_4]*4 + [pool] + [conv_4]*4 + [pool] + [fully_connected_1]*2 + [fully_connected_2] + [linear_output]


# ###########
# ResNet-18 #
# ###########


def modify_layer(standard, new_parms):
    modified = standard.copy()
    modified.update(new_parms)
    return modified


def simple_res_block(conv_layer):
    return OrderedDict({'type': 'residual', 'conv1': conv_layer, 'conv2': conv_layer})


def build_network(sequence):
    names = list()
    parms = list()
    for batch in sequence:
        layer_name, layer, reps = batch
        if reps > 1:
            for index in range(1, reps + 1):
                names.append(layer_name + '_' + str(index))
                parms.append(layer)
        else:
            names.append(layer_name)
            parms.append(layer)
    return names, parms

std_conv = OrderedDict({'type': 'convolutional', 'stride': [1, 1], 'kernel': [3, 3], 'features': 64,
                        'activation': 'relu', 'padding': 'SAME', 'batch_norm': True})
conv_0 = modify_layer(std_conv, {'stride': [2, 2], 'kernel': [7, 7]})
pool_0 = OrderedDict({'type': 'pooling', 'stride': [2, 2], 'kernel': [3, 3], 'pool_type': 'max', 'padding': 'SAME'})
max_pool_2 = OrderedDict({'type': 'pooling', 'stride': [2, 2], 'kernel': [2, 2], 'pool_type': 'max', 'padding': 'SAME'})
res_1 = simple_res_block(modify_layer(std_conv, {'features': 64}))
res_2 = simple_res_block(modify_layer(std_conv, {'features': 128}))
res_3 = simple_res_block(modify_layer(std_conv, {'features': 256}))
res_4 = simple_res_block(modify_layer(std_conv, {'features': 512}))
avg_pool_7 = OrderedDict({'type': 'pooling', 'stride': [7, 7], 'kernel': [7, 7], 'pool_type': 'avg', 'padding': 'SAME'})
fully_connected = OrderedDict({'type': 'fully_connected', 'weights': 1000, 'bias': 1000, 'activation': 'relu',
                               'regularize': True})
linear_output = OrderedDict({'type': 'linear_output', 'weights': 27, 'bias': 27, 'regularize': False})

sequence = [('conv0', conv_0, 1), ('pool0', pool_0, 1),
            ('res1', res_1, 2), ('pool1', max_pool_2, 1),
            ('res2', res_2, 2), ('pool2', max_pool_2, 1),
            ('res3', res_3, 2), ('pool3', max_pool_2, 1),
            ('res4', res_4, 2), ('pool4', avg_pool_7, 1),
            ('fc', fully_connected, 1), ('linear_output', linear_output, 1)]

resnet_18_names, resnet_18_parms = build_network(sequence)
io_utils.write_json_network_config('network_ResNet_18.json', resnet_18_names, resnet_18_parms)


# ###########
# ResNet-34 #
# ###########

sequence = [('conv0', conv_0, 1), ('pool0', pool_0, 1),
            ('res1', res_1, 3), ('pool1', max_pool_2, 1),
            ('res2', res_2, 4), ('pool2', max_pool_2, 1),
            ('res3', res_3, 6), ('pool3', max_pool_2, 1),
            ('res4', res_4, 3), ('pool4', avg_pool_7, 1),
            ('fc', fully_connected, 1), ('linear_output', linear_output, 1)]

resnet_34_names, resnet_34_parms = build_network(sequence)
io_utils.write_json_network_config('network_ResNet_34.json', resnet_34_names, resnet_34_parms)


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
hyper_params_classification = {'network_type': 'classifier', #'network_type': 'classifier'
                           'optimization': 'SGD', #'optimization': 'SGD'
                           'warm_up': False,
                           'num_epochs_per_decay':3,
                           'learning_rate_decay_factor': 0.5,
                           'initial_learning_rate': 0.1,
                           'num_epochs_per_ramp': 10,
                           'num_epochs_in_warm_up': 100,
                           'warm_up_max_learning_rate': 1e-3,
                           'weight_decay': 1.e-4,
                           'moving_average_decay': 0.9999
                           }
# Classification
#TODO

if __name__ == '__main__':
    io_utils.write_json_network_config('network_VGG16.json', layer_keys_list, layer_params_list)
    _ = io_utils.load_json_network_config('network_VGG16.json')
    io_utils.write_json_hyper_params('hyper_params_classifier_VGG16.json', hyper_params_classification)
    _ = io_utils.load_json_hyper_params('hyper_params_classifier_VGG16.json')
