"""
Created on 10/9/17.
@author: Numan Laanait.
email: laanaitn@ornl.gov
"""

from collections import OrderedDict
from copy import deepcopy
import sys
sys.path.append('../')
from stemdl import io_utils
import numpy as np
import os

#################################
# templates for network_config  #
#################################

def generate_alex_net_custom(conv_type="conv_2D"):

    layer_keys_list = ['conv1', 'conv2', 'pool1', 'conv3', 'conv4', 'conv5', 'conv6', 'pool2',
                       'conv7', 'conv8', 'conv9', 'conv10', 'pool3', 'conv11','conv12', 'conv13', 'conv14', 'pool4',
                       'fc_1', 'linear_output']

    # parameters dictionary
    conv_layer_1 = OrderedDict({'type': conv_type, 'stride': [1, 1], 'kernel': [11, 11], 'features': 64,
                                'activation':'relu', 'padding':'SAME','batch_norm':True})
    conv_layer_2 = OrderedDict({'type': conv_type, 'stride': [1, 1], 'kernel': [5, 5], 'features': 128,
                                'activation':'relu', 'padding':'SAME', 'batch_norm':True})
    conv_layer_3 = OrderedDict({'type': conv_type, 'stride': [1, 1], 'kernel': [5, 5], 'features': 256,
                                'activation':'relu', 'padding':'SAME', 'batch_norm':True})
    conv_layer_4 = OrderedDict({'type': conv_type, 'stride': [1, 1], 'kernel': [5, 5], 'features': 512,
                                'activation':'relu', 'padding':'SAME', 'batch_norm':True})
    pool_avg = OrderedDict({'type': 'pooling', 'stride': [2, 2], 'kernel': [2, 2], 'pool_type': 'avg','padding':'SAME'})
    pool_max = OrderedDict({'type': 'pooling', 'stride': [2, 2], 'kernel': [2, 2], 'pool_type': 'max','padding':'SAME'})
    conv_layer_3 = OrderedDict({'type': conv_type, 'stride': [1, 1], 'kernel': [2, 2], 'features': 64,
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


def generate_vgg_net_json(num_layers=16, output_features=4, conv_type="conv_2D"):

    assert num_layers in [11, 13, 16, 19], "Allowed number of layers:{}".format([11, 13, 16, 19])

    conv_64 = OrderedDict({'type': conv_type, 'stride': [1, 1], 'kernel': [3, 3], 'features': 64,
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


def generate_fcnet_json(conv_type="conv_2D", input_channels= 64, features=64, kernel=[5,5], n_pool=5, n_layers_per_path=2,
                        output_channels=1, dropout_prob=None, save=True, model='fcnet', output_input_ratio=2, f_c_block=True):
    
    # if type(n_layers) == int:
    #     n_layers = n_layers * (2 * n_pool + 1)
    if conv_type == "coord_conv":
        features -= 2
    pool = OrderedDict({'type': 'pooling', 'stride': [2, 2], 'kernel': [2, 2], 'pool_type': 'max','padding':'SAME'})
    deconv_layer_base = OrderedDict({'type': "deconv_2D", 'stride': [2, 2], 'kernel': kernel, 'features': features,
                             'padding': 'SAME', 'upsample': pool['kernel'][0]})

    conv_layer_base = OrderedDict({'type': conv_type, 'stride': [1, 1], 'kernel': kernel, 'features': features,
                            'activation': 'relu', 'padding': 'SAME', 'batch_norm': True, 'dropout':dropout_prob})
    if conv_type == 'depth_conv':
        conv_layer_base = OrderedDict({'type': conv_type, 'stride': [1, 1], 'kernel': kernel, 'features': features,
                            'activation': 'relu', 'padding': 'SAME', 'batch_norm': True, 'dropout':dropout_prob})

    layers_params = []
    layers_keys = []

    # Encoder path
    rank = 0
    for i in range(n_pool):
        for _ in range(n_layers_per_path):
            conv_layer = deepcopy(conv_layer_base)
            conv_layer['features'] = int(conv_layer['features'])
            layers_keys.append('conv_%s' % rank)
            layers_params.append(conv_layer)
            rank += 1
        conv_layer_base['features'] *= 2
        if conv_type == 'depth_conv':
           conv_layer_base['features'] = features 
        input_channels *= features
        layers_keys.append('pool_%s' % i)
        layers_params.append(pool)

    # Decoder path
    deconv_layer_base['features'] = conv_layer_base['features']

    if conv_type == 'depth_conv':
        deconv_layer_base['features'] = input_channels 
    extra = output_input_ratio//2
    if f_c_block:
        dense_layers_block = OrderedDict({'type': 'dense_layers_block', 'activation': 'relu',
                                   'regularize': True, 'n_layers': 2})
        layers_keys.append('dense_layers_block')
        layers_params.append(dense_layers_block)
    for i in range(n_pool - extra):
        deconv_layer = deepcopy(deconv_layer_base)
        # deconv_layer['features'] //= 2
        # layers_keys.append('deconv_%s' % i )
        # layers_params.append(deconv_layer)
        for _ in range(n_layers_per_path):
            conv_layer = deepcopy(conv_layer_base)
            if conv_type == 'depth_conv':
               conv_layer['features'] = features
            else:
                conv_layer['features'] = deconv_layer['features']
            layers_keys.append('conv_%s' % rank )
            layers_params.append(conv_layer)
            rank += 1
        deconv_layer = deepcopy(deconv_layer_base)
        deconv_layer['features'] //= 2
        layers_keys.append('deconv_%s' % i )
        layers_params.append(deconv_layer)
    
    # final conv 1x1
    conv_1by1 = OrderedDict({'type': 'conv_2D', 'stride': [1, 1], 'kernel': [1, 1], 'features': output_channels,
                            'activation': 'relu', 'padding': 'VALID', 'batch_norm': False})
    layers_keys.append('CONV_FIN')
    layers_params.append(conv_1by1)

    if conv_type == 'coord_conv':
        features += 2

    if save:
        name ='network_'+ model + '_%s_%s_%s.json' %(str(features), str(n_layers_per_path * n_pool), conv_type) 
        io_utils.write_json_network_config(name, layers_keys, layers_params)
    return OrderedDict(zip(layers_keys,layers_params)) 
        

def generate_fc_dense_json(conv_type="conv_2D", input_channels= 64, fc_layers= 4, input_size = 256, features=64, kernel=[5,5], n_pool=5, n_layers_per_path=2,
                        output_channels=1, output_size=128, dropout_prob=None, save=True, model='fc_dense'):
    
    # if type(n_layers) == int:
    #     n_layers = n_layers * (2 * n_pool + 1)
    if conv_type == "coord_conv":
        features -= 2
    pool = OrderedDict({'type': 'pooling', 'stride': [2, 2], 'kernel': [2, 2], 'pool_type': 'max','padding':'SAME'})
    conv_layer_base = OrderedDict({'type': conv_type, 'stride': [1, 1], 'kernel': kernel, 'features': features,
                            'activation': 'relu', 'padding': 'SAME', 'batch_norm': True, 'dropout':dropout_prob})
    deconv_layer_base = OrderedDict({'type': "deconv_2D", 'stride': [2, 2], 'kernel': kernel, 'features': features,
                             'padding': 'SAME', 'upsample': pool['kernel'][0]})
    if conv_type == 'depth_conv':
        features = 2
        conv_layer_base = OrderedDict({'type': conv_type, 'stride': [1, 1], 'kernel': kernel, 'features': features,
                            'activation': 'relu', 'padding': 'SAME', 'batch_norm': True, 'dropout':dropout_prob})

    layers_params = []
    layers_keys = []

    # Encoder path
    rank = 0
    for i in range(n_pool):
        for _ in range(n_layers_per_path):
            conv_layer = deepcopy(conv_layer_base)
            conv_layer['features'] = int(conv_layer['features'])
            layers_keys.append('conv_%s' % rank)
            layers_params.append(conv_layer)
            rank += 1
        conv_layer_base['features'] *= 2
        if conv_type == 'depth_conv':
           conv_layer_base['features'] = features 
        # input_channels *= features
        layers_keys.append('pool_%s' % i)
        layers_params.append(pool)

    # Decoder path
  
    dense_layers_block = OrderedDict({'type': 'dense_layers_block', 'activation': 'relu',
                                   'regularize': True, 'n_layers': fc_layers, 'conv_type':conv_type})
    layers_keys.append('dense_layers_block')
    layers_params.append(dense_layers_block)

    # conv_1by1 = OrderedDict({'type': 'conv_2D', 'stride': [1, 1], 'kernel': [1, 1], 'features': output_channels,
                            # 'activation': 'relu', 'padding': 'VALID', 'batch_norm': True})
    # layers_keys.append('CONV_1by1')
    # layers_params.append(conv_1by1)
    conv_type = 'conv_2D'
    features_2d = int((input_size/2**n_pool)**2)
    # features_2d = features
    conv_layer_base = OrderedDict({'type': conv_type, 'stride': [1, 1], 'kernel': kernel, 'features': features_2d,
                            'activation': 'relu', 'padding': 'SAME', 'batch_norm': True, 'dropout':dropout_prob})
    num_upsamples = int(np.log2(int(output_size/int(np.sqrt(input_channels))))) 
    for i in range(num_upsamples):
        deconv_layer = deepcopy(deconv_layer_base)
        deconv_layer['features'] = features_2d  
        layers_keys.append('deconv_%s' % i )
        layers_params.append(deconv_layer)
        for _ in range(n_layers_per_path):
            conv_layer = deepcopy(conv_layer_base)
            conv_layer['features'] = features_2d 
            layers_keys.append('conv_%s' % rank )
            layers_params.append(conv_layer)
            rank += 1
        features_2d = features_2d //2
    
    # final conv 1x1
    conv_1by1 = OrderedDict({'type': 'conv_2D', 'stride': [1, 1], 'kernel': [1, 1], 'features': output_channels,
                            'activation': 'relu', 'padding': 'VALID', 'batch_norm': False})
    layers_keys.append('CONV_FIN')
    layers_params.append(conv_1by1)

    if save:
        name ='network_'+ model + '_%s_%s_%s.json' %(str(features), str(n_layers_per_path * n_pool), conv_type) 
        io_utils.write_json_network_config(name, layers_keys, layers_params)
    return OrderedDict(zip(layers_keys,layers_params)) 

def generate_freq2space_json(out_dir= 'json_files', conv_type="conv_2D", input_channels= 64, fc_layers= 4, input_size = 256, init_features=2048, kernel=[5,5], n_layers_per_path=3,
                        output_channels=1, output_size=128, dropout_prob=0.25, save=True, model='freq2space', batch_norm=True, attention=False, CVAE=True):
    
    # if type(n_layers) == int:
    #     n_layers = n_layers * (2 * n_pool + 1)

    pool = OrderedDict({'type': 'pooling', 'stride': [2, 2], 'kernel': [2, 2], 'pool_type': 'max','padding':'SAME'})
    conv_layer_base = OrderedDict({'type': conv_type, 'stride': [1, 1], 'kernel': kernel, 'features': None,
                            'activation': 'relu', 'padding': 'SAME', 'batch_norm': True, 'dropout':dropout_prob})
    deconv_layer_base = OrderedDict({'type': "deconv_2D", 'stride': [2, 2], 'kernel': [1,1], 'features': None,
                             'padding': 'VALID', 'upsample': pool['kernel'][0]})
    
    layers_params = []
    layers_keys = []

    # depth to space
    freq2space_block = OrderedDict({'type': 'freq2space', 'activation': 'relu', 'dropout': dropout_prob, 
                                    'init_features':init_features, 'batch_norm': batch_norm})
    if attention:
        freq2space_block['type'] = 'freq2space_attention' 
        layers_keys.append('freq2space_attention')
    elif CVAE:
        layers_keys.append('freq2space_CVAE')
        freq2space_block['type'] = 'freq2space_CVAE' 
    else:
        layers_keys.append('freq2space')
    layers_params.append(freq2space_block)

    # Encoder path
    rank = 0
    num_upsamples = int(np.log2(int(output_size/int(np.sqrt(input_channels)))))  
    # features = init_features // num_upsamples
    features = init_features
    for i in range(num_upsamples):
        deconv_layer = deepcopy(deconv_layer_base)
        deconv_layer['features'] = features  
        layers_keys.append('deconv_%s' % i )
        layers_params.append(deconv_layer) 
        for _ in range(n_layers_per_path):
            conv_layer = deepcopy(conv_layer_base)
            conv_layer['features'] = features 
            layers_keys.append('conv_%s' % rank)
            layers_params.append(conv_layer)
            rank += 1
        features = features // 2

    # final conv 1x1
    # conv_1by1 = OrderedDict({'type': 'conv_2D', 'stride': [1, 1], 'kernel': [1, 1], 'features': output_channels,
    #                         'activation': 'relu', 'padding': 'VALID', 'batch_norm': False})
    # layers_keys.append('CONV_FIN')
    # layers_params.append(conv_1by1)

    if attention:
        model += "_attention"
    elif CVAE:
        model += '_CVAE'

    if save:
        name ='network_'+ model + '_%s_%s_%s.json' %(str(init_features), str(n_layers_per_path), conv_type) 
        io_utils.write_json_network_config(os.path.join(out_dir,name), layers_keys, layers_params)
    return OrderedDict(zip(layers_keys,layers_params)) 

def generate_coordconv_json(features=64, kernel=[1,1], n_layers=4,
                        output_channels=1, dropout_prob=None, save=True, model='coordconv'):
    
    conv_layer_base = OrderedDict({'type': 'coord_conv', 'stride': [1, 1], 'kernel': kernel, 'features': 64,
                            'activation': 'relu', 'padding': 'SAME', 'batch_norm': True, 'dropout':dropout_prob})

    features -= 2
    rank = 0
    layers_params = []
    layers_keys = []

    for i in range(n_layers): 
        conv_layer = deepcopy(conv_layer_base)
        conv_layer['features'] = int(conv_layer['features'])
        layers_keys.append('conv_%s' % rank)
        layers_params.append(conv_layer)
        rank += 1
        conv_layer_base['features'] *= 2
    conv_1by1 = OrderedDict({'type': 'conv_2D', 'stride': [1, 1], 'kernel': [1, 1], 'features': output_channels,
                            'activation': 'relu', 'padding': 'VALID', 'batch_norm': False})
    layers_keys.append('CONV_FIN')
    layers_params.append(conv_1by1)

    if save:
        name ='network_'+ model + '_%s_%s.json' %(str(features), str(n_layers)) 
        io_utils.write_json_network_config(name, layers_keys, layers_params)
    return OrderedDict(zip(layers_keys,layers_params)) 

###############
# FC_DenseNet #
##############

def generate_fcdensenet_json(random=False, conv_type="conv_2D", growth_rate=64, kernel= [5,5], n_pool= 5, 
                                n_layers= 2, model= 'FCDenseNet_custom', output_channels=256, 
                                dropout_prob=0.2, name='', save=True):
    DB_conv_kernel = [3, 3]
    if model == 'FC-DenseNet56':
        n_pool=5
        growth_rate=12
        n_layers_per_block=4
    elif model == 'FC-DenseNet67':
        n_pool=5
        growth_rate=16
        n_layers_per_block=5
    elif model == 'FC-DenseNet103':
        n_pool=5
        growth_rate=16
        n_layers_per_block=[4, 5, 7, 10, 12, 15, 12, 10, 7, 5, 4]
    elif model == 'FC-DenseNet103_custom':
        n_pool=5
        growth_rate=32
        n_layers_per_block=[2, 2, 2, 4, 5, 5, 5, 4, 2, 2, 2]
        # n_layers_per_block=3
        DB_conv_kernel = [5, 5]
    elif model == 'FCDenseNet_custom_18TF':
        n_pool=5
        growth_rate=64
        n_layers_per_block=[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
        # n_layers_per_block=3
        DB_conv_kernel = [7, 7]
    elif model == 'FCDenseNet_custom':
        if random:
            n_pool= 3
            num_db = np.random.randint(1, 6, size=4)
            n_layers = np.append(num_db, num_db[::-1][1:])
            n_layers_per_block = [int(itm) for itm in n_layers]
            kernel_w = np.array([3,5,7])[np.random.randint(3,size=1)[0]]
            DB_conv_kernel = [int(kernel_w), int(kernel_w)]
            growth_rate = int(np.array([64,128,256])[np.random.randint(3,size=1)[0]])
        else:
            n_pool=n_pool
            growth_rate=growth_rate
            n_layers_per_block=n_layers
            # n_layers_per_block=3
            DB_conv_kernel = kernel
            # 25 TFLOPS: kernel=5, growth=128, pool=3, layers=2
    if type(n_layers_per_block) == int:
        n_layers_per_block = [n_layers_per_block] * (2 * n_pool + 1)

    std_conv = OrderedDict({'type': conv_type, 'stride': [1, 1], 'kernel': DB_conv_kernel, 'features': growth_rate,
                            'activation': 'relu', 'padding': 'SAME', 'batch_norm': False})
    deconv = OrderedDict({'type': ''})
    layer = OrderedDict({'type': conv_type, 'stride': [1, 1], 'kernel': DB_conv_kernel, 'features': growth_rate,
                            'activation': 'relu', 'padding': 'SAME', 'batch_norm': True, 'dropout':dropout_prob})
    pool = OrderedDict({'type': 'pooling', 'stride': [2, 2], 'kernel': [2, 2], 'pool_type': 'max','padding':'SAME'})


    layers_params_list = []
    layers_keys_list = []

    # 3x3 conv
    layers_params_list.append(std_conv)
    layers_keys_list.append('CONV_INIT')
    n_filters = std_conv['features']

    # Transition Down
    for i in range(n_pool):
        # Dense Block
        conv_layers = []
        for j in range(n_layers_per_block[i]):
            conv_layers.append(('conv_%s'%j, layer))

        conv_layers = OrderedDict(conv_layers)
        DB = OrderedDict({'type': 'dense_block_down', 'conv':conv_layers})
        layers_params_list.append(DB)
        layers_keys_list.append('DB_'+str(i))
        n_filters += growth_rate * n_layers_per_block[i]
        n_filters -= n_filters % 8
        # Transition Down
        TD = OrderedDict({'type': "transition_down", 'conv':
                                {'type': conv_type, 'stride': [1, 1], 'kernel': [1, 1],
                                'features': n_filters,
                                'activation': 'relu', 'padding': 'SAME', 'batch_norm': True, 'dropout':dropout_prob},
                                'pool':pool})
        layers_params_list.append(TD)
        layers_keys_list.append('TD_'+str(i))

    # Bottleneck
    conv_layers = []
    for j in range(n_layers_per_block[n_pool]):
        conv_layers.append(('conv_%s'%j, layer))

    conv_layers = OrderedDict(conv_layers)
    DB = OrderedDict({'type': 'dense_block_bottleneck', 'conv':conv_layers})
    layers_params_list.append(DB)
    layers_keys_list.append('DB_'+str(i+1))
    offset = i+2

    # Transition Up
    for i in range(n_pool):
        n_filters_keep = growth_rate * n_layers_per_block[n_pool + i]
        n_filters_keep -= n_filters % 8
        TU = OrderedDict({'type': "transition_up", 'deconv':
                                {'type': 'deconvolutional', 'stride': [2, 2], 'kernel': [3, 3],
                                'features': n_filters_keep,'padding': 'SAME', 'upsample':pool['kernel'][0]}
                                })
        layers_params_list.append(TU)
        layers_keys_list.append('TU_'+str(i))
        # Dense Block
        conv_layers = []
        for j in range(n_layers_per_block[n_pool + i + 1]):
            conv_layers.append(('conv_%s'%j, layer))

        conv_layers = OrderedDict(conv_layers)
        DB = OrderedDict({'type': 'dense_block_up', 'conv':conv_layers})
        layers_params_list.append(DB)
        layers_keys_list.append('DB_'+str(offset+i))

    # 1x1 conv
    conv_1by1 = OrderedDict({'type': conv_type, 'stride': [1, 1], 'kernel': [1, 1], 'features': output_channels,
                            'activation': 'relu', 'padding': 'SAME', 'batch_norm': False})
    layers_params_list.append(conv_1by1)
    layers_keys_list.append('CONV_FIN')

    # write to json_dict
    if save:
        io_utils.write_json_network_config('network_'+ model + '.json', layers_keys_list, layers_params_list)
        print('growth_rate=%s,kernel=%s,n_layers=%s' %(format(growth_rate),format(DB_conv_kernel),format(n_layers_per_block)))
    else:
        return OrderedDict(zip(layers_keys_list,layers_params_list))

# ########
# ResNet #
# ########

def generate_res_net_json(num_layers=18, output_features=4, conv_type="conv_2D"):

    # ################# ################# ################# ################# ################# ################

    assert num_layers in [18, 34, 50, 101, 152]
    output_features = int(output_features)
    assert output_features > 0

    std_conv = OrderedDict({'type': conv_type, 'stride': [1, 1], 'kernel': [3, 3], 'features': 64,
                            'activation': 'relu', 'padding': 'SAME', 'batch_norm': True})
    conv_0 = modify_layer(std_conv, {'stride': [2, 2], 'kernel': [7, 7]})
    pool_0 = OrderedDict({'type': 'pooling', 'stride': [2, 2], 'kernel': [3, 3], 'pool_type': 'max', 'padding': 'SAME'})
    max_pool_2 = OrderedDict({'type': 'pooling', 'stride': [2, 2], 'kernel': [2, 2], 'pool_type': 'max',
                              'padding': 'SAME'})
    avg_pool_7 = OrderedDict({'type': 'pooling', 'stride': [7, 7], 'kernel': [7, 7], 'pool_type': 'avg',
                              'padding': 'SAME'})
    # fully_connected = OrderedDict({'type': 'fully_connected', 'weights': 1000, 'bias': 1000, 'activation': 'relu',
    #                                'regularize': True})
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
                        # ('linear_output', linear_output, 1)]
                        ('linear_output', linear_output, 1)]
        else:  # 34
            sequence = [('conv0', conv_0, 1), ('pool0', pool_0, 1),
                        ('res1', res_1, 3), ('pool1', max_pool_2, 1),
                        ('res2', res_2, 4), ('pool2', max_pool_2, 1),
                        ('res3', res_3, 6), ('pool3', max_pool_2, 1),
                        ('res4', res_4, 3), ('pool4', avg_pool_7, 1),
                        # ('linear_output', linear_output, 1)]
                        ('linear_output', linear_output, 1)]
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
                        ('linear_output', linear_output, 1)]

        elif num_layers == 101:
            sequence = [('conv0', conv_0, 1), ('pool0', pool_0, 1),
                        ('res1', bn_res_1, 3), ('pool1', max_pool_2, 1),
                        ('res2', bn_res_2, 4), ('pool2', max_pool_2, 1),
                        ('res3', bn_res_3, 23), ('pool3', max_pool_2, 1),
                        ('res4', bn_res_4, 3), ('pool4', avg_pool_7, 1),
                        ('linear_output', linear_output, 1)]

        else:  # 152
            sequence = [('conv0', conv_0, 1), ('pool0', pool_0, 1),
                        ('res1', bn_res_1, 3), ('pool1', max_pool_2, 1),
                        ('res2', bn_res_2, 8), ('pool2', max_pool_2, 1),
                        ('res3', bn_res_3, 36), ('pool3', max_pool_2, 1),
                        ('res4', bn_res_4, 3), ('pool4', avg_pool_7, 1),
                        ('linear_output', linear_output, 1)]

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
