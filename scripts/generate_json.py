"""
Created on 10/9/17.
@author: Numan Laanait.
email: laanaitn@ornl.gov
"""

from collections import OrderedDict
import sys
sys.path.append('../')
from stemdl import io_utils


#################################
# templates for network_config  #
#################################

def generate_alex_net_custom():

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

    io_utils.write_json_network_config('network_regressor.json', layer_keys_list, layer_params_list)

#######
# VGG #
#######


def modify_layer(standard, new_parms):
    modified = standard.copy()
    modified.update(new_parms)
    return modified


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


def generate_vgg_net_json(num_layers=16, output_features=4):

    assert num_layers in [11, 13, 16, 19], "Allowed number of layers:{}".format([11, 13, 16, 19])

    conv_64 = OrderedDict({'type': 'convolutional', 'stride': [1, 1], 'kernel': [3, 3], 'features': 64,
                           'activation': 'relu', 'padding': 'SAME', 'batch_norm': True})
    conv_128 = modify_layer(conv_64, {'features': 128})
    conv_256 = modify_layer(conv_64, {'features': 256})
    conv_512 = modify_layer(conv_64, {'features': 512})
    max_pool_2 = OrderedDict({'type': 'pooling', 'stride': [2, 2], 'kernel': [2, 2], 'pool_type': 'max',
                              'padding': 'SAME'})
    fully_connected_4 = OrderedDict({'type': 'fully_connected', 'weights': 4096, 'bias': 4096, 'activation': 'relu',
                                     'regularize': True})
    fully_connected_1 = OrderedDict({'type': 'fully_connected', 'weights': 1000, 'bias': 1000, 'activation': 'relu',
                                     'regularize': True})
    linear_output = OrderedDict({'type': 'linear_output', 'weights': output_features, 'bias': output_features,
                                 'regularize': False})

    if num_layers == 11:
        sequence = [('conv0', conv_64, 1), ('pool0', max_pool_2, 1),
                    ('conv1', conv_128, 1), ('pool1', max_pool_2, 1),
                    ('conv2', conv_256, 2), ('pool2', max_pool_2, 1),
                    ('conv3', conv_512, 2), ('pool3', max_pool_2, 1),
                    ('conv4', conv_512, 2), ('pool4', max_pool_2, 1),
                    ('fc4', fully_connected_4, 2), ('fc1', fully_connected_1, 1),
                    ('linear_output', linear_output, 1)]
    elif num_layers == 13:
        sequence = [('conv0', conv_64, 2), ('pool0', max_pool_2, 1),
                    ('conv1', conv_128, 2), ('pool1', max_pool_2, 1),
                    ('conv2', conv_256, 2), ('pool2', max_pool_2, 1),
                    ('conv3', conv_512, 2), ('pool3', max_pool_2, 1),
                    ('conv4', conv_512, 2), ('pool4', max_pool_2, 1),
                    ('fc4', fully_connected_4, 2), ('fc1', fully_connected_1, 1),
                    ('linear_output', linear_output, 1)]
    elif num_layers == 16:
        sequence = [('conv0', conv_64, 2), ('pool0', max_pool_2, 1),
                    ('conv1', conv_128, 2), ('pool1', max_pool_2, 1),
                    ('conv2', conv_256, 3), ('pool2', max_pool_2, 1),
                    ('conv3', conv_512, 3), ('pool3', max_pool_2, 1),
                    ('conv4', conv_512, 3), ('pool4', max_pool_2, 1),
                    ('fc4', fully_connected_4, 2), ('fc1', fully_connected_1, 1),
                    ('linear_output', linear_output, 1)]
    elif num_layers == 19:
        sequence = [('conv0', conv_64, 2), ('pool0', max_pool_2, 1),
                    ('conv1', conv_128, 2), ('pool1', max_pool_2, 1),
                    ('conv2', conv_256, 4), ('pool2', max_pool_2, 1),
                    ('conv3', conv_512, 4), ('pool3', max_pool_2, 1),
                    ('conv4', conv_512, 4), ('pool4', max_pool_2, 1),
                    ('fc4', fully_connected_4, 2), ('fc1', fully_connected_1, 1),
                    ('linear_output', linear_output, 1)]

    vgg_names, vgg_parms = build_network(sequence)
    io_utils.write_json_network_config('network_VGGNet_' + str(num_layers) + '_w_batch_norm.json', vgg_names, vgg_parms)


# ########
# ResNet #
# ########


def generate_res_net_json(num_layers=18, output_features=4):

    # ################# ################# ################# ################# ################# ################

    assert num_layers in [18, 34, 50, 101, 152]
    output_features = int(output_features)
    assert output_features > 0

    std_conv = OrderedDict({'type': 'convolutional', 'stride': [1, 1], 'kernel': [3, 3], 'features': 64,
                            'activation': 'relu', 'padding': 'SAME', 'batch_norm': True})
    conv_0 = modify_layer(std_conv, {'stride': [2, 2], 'kernel': [7, 7]})
    pool_0 = OrderedDict({'type': 'pooling', 'stride': [2, 2], 'kernel': [3, 3], 'pool_type': 'max', 'padding': 'SAME'})
    max_pool_2 = OrderedDict({'type': 'pooling', 'stride': [2, 2], 'kernel': [2, 2], 'pool_type': 'max',
                              'padding': 'SAME'})
    avg_pool_7 = OrderedDict({'type': 'pooling', 'stride': [7, 7], 'kernel': [7, 7], 'pool_type': 'avg',
                              'padding': 'SAME'})
    fully_connected = OrderedDict({'type': 'fully_connected', 'weights': 1000, 'bias': 1000, 'activation': 'relu',
                                   'regularize': True})
    linear_output = OrderedDict({'type': 'linear_output', 'weights': output_features, 'bias': output_features,
                                 'regularize': False})
    if num_layers < 50:

        def simple_res_block(conv_layer):
            return OrderedDict({'type': 'residual', 'conv1': conv_layer, 'conv2': conv_layer})

        res_1 = simple_res_block(modify_layer(std_conv, {'features': 64}))
        res_2 = simple_res_block(modify_layer(std_conv, {'features': 128}))
        res_3 = simple_res_block(modify_layer(std_conv, {'features': 256}))
        res_4 = simple_res_block(modify_layer(std_conv, {'features': 512}))

        if num_layers == 18:
            sequence = [('conv0', conv_0, 1), ('pool0', pool_0, 1),
                        ('res1', res_1, 2), ('pool1', max_pool_2, 1),
                        ('res2', res_2, 2), ('pool2', max_pool_2, 1),
                        ('res3', res_3, 2), ('pool3', max_pool_2, 1),
                        ('res4', res_4, 2), ('pool4', avg_pool_7, 1),
                        ('fc', fully_connected, 1), ('linear_output', linear_output, 1)]
        else:  # 34
            sequence = [('conv0', conv_0, 1), ('pool0', pool_0, 1),
                        ('res1', res_1, 3), ('pool1', max_pool_2, 1),
                        ('res2', res_2, 4), ('pool2', max_pool_2, 1),
                        ('res3', res_3, 6), ('pool3', max_pool_2, 1),
                        ('res4', res_4, 3), ('pool4', avg_pool_7, 1),
                        ('fc', fully_connected, 1), ('linear_output', linear_output, 1)]
    else:

        def bneck_res_block(conv_layer, chans_1, chans_2):
            bn_in = modify_layer(conv_layer, {'features': chans_1, 'kernel': [1, 1], 'stride': [1, 1]})
            conv_layer = modify_layer(conv_layer, {'features': chans_1, 'kernel': [3, 3], 'stride': [1, 1]})
            bn_out = modify_layer(bn_in, {'features': chans_2})
            basic_dict = {'type': 'residual', 'conv1': bn_in, 'conv2': conv_layer, 'conv3': bn_out}
            return OrderedDict(sorted(basic_dict.items(), key=lambda t: t[0]))

        bn_res_1 = bneck_res_block(std_conv, 64, 256)
        bn_res_2 = bneck_res_block(std_conv, 128, 512)
        bn_res_3 = bneck_res_block(std_conv, 256, 1024)
        bn_res_4 = bneck_res_block(std_conv, 512, 2048)

        if num_layers == 50:
            sequence = [('conv0', conv_0, 1), ('pool0', pool_0, 1),
                        ('res1', bn_res_1, 3), ('pool1', max_pool_2, 1),
                        ('res2', bn_res_2, 4), ('pool2', max_pool_2, 1),
                        ('res3', bn_res_3, 6), ('pool3', max_pool_2, 1),
                        ('res4', bn_res_4, 3), ('pool4', avg_pool_7, 1),
                        ('fc', fully_connected, 1), ('linear_output', linear_output, 1)]

        elif num_layers == 101:
            sequence = [('conv0', conv_0, 1), ('pool0', pool_0, 1),
                        ('res1', bn_res_1, 3), ('pool1', max_pool_2, 1),
                        ('res2', bn_res_2, 4), ('pool2', max_pool_2, 1),
                        ('res3', bn_res_3, 23), ('pool3', max_pool_2, 1),
                        ('res4', bn_res_4, 3), ('pool4', avg_pool_7, 1),
                        ('fc', fully_connected, 1), ('linear_output', linear_output, 1)]

        else:  # 152
            sequence = [('conv0', conv_0, 1), ('pool0', pool_0, 1),
                        ('res1', bn_res_1, 3), ('pool1', max_pool_2, 1),
                        ('res2', bn_res_2, 8), ('pool2', max_pool_2, 1),
                        ('res3', bn_res_3, 36), ('pool3', max_pool_2, 1),
                        ('res4', bn_res_4, 3), ('pool4', avg_pool_7, 1),
                        ('fc', fully_connected, 1), ('linear_output', linear_output, 1)]

    resnet_names, resnet_parms = build_network(sequence)
    io_utils.write_json_network_config('network_ResNet_' + str(num_layers) + '.json', resnet_names, resnet_parms)


#################################
# templates for hyper-parameters #
#################################

# Regression
hyper_params_regression = {'network_type': 'regressor',  # ' network_type': 'classifier'
                           'optimization': 'ADAM',  # 'optimization': 'SGD'
                           'warm_up': False,
                           'num_epochs_per_decay': 3,
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
                                             'residual_decay_factor': 0.75}}
# Classification
hyper_params_classification = {'network_type': 'classifier',  # 'network_type': 'classifier'
                               'optimization': 'SGD',  # 'optimization': 'SGD'
                               'warm_up': False,
                               'num_epochs_per_decay': 3,
                               'learning_rate_decay_factor': 0.5,
                               'initial_learning_rate': 0.1,
                               'num_epochs_per_ramp': 10,
                               'num_epochs_in_warm_up': 100,
                               'warm_up_max_learning_rate': 1e-3,
                               'weight_decay': 1.e-4,
                               'moving_average_decay': 0.9999}

if __name__ == '__main__':
    io_utils.write_json_hyper_params('hyper_params_classifier_VGG16.json', hyper_params_classification)
    _ = io_utils.load_json_hyper_params('hyper_params_classifier_VGG16.json')
