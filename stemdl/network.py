"""
Created on 10/8/17.
@author: Numan Laanait
email: laanaitn@ornl.gov
misc: ResNet subclass added by Suhas Somnath
"""

from collections import OrderedDict, deque
from itertools import chain
import re
import math
from copy import deepcopy
import tensorflow as tf
import numpy as np
import horovod.tensorflow as hvd
from .mp_wrapper import mp_regularizer_wrapper

worker_name='model'
tf.logging.set_verbosity(tf.logging.ERROR)

class ConvNet:
    """
    Vanilla Convolutional Neural Network (Feed-Forward).
    """
    def __init__(self, scope, params, hyper_params, network, images, labels, operation='train',
                 summary=False, verbose=True):
        """
        :param params: dict
        :param global_step: as it says
        :param hyper_params: dictionary, hyper-parameters
        :param network: collections.OrderedDict, specifies ConvNet layers
        :param images: batch of images
        :param labels: batch of labels
        :param operation: string, 'train' or 'eval'
        :param summary: bool, flag to write tensorboard summaries
        :param verbose: bool, flag to print shapes of outputs
        :return:
        """
        self.scope = scope
        self.params = params
        self.global_step = 0
        self.hyper_params = hyper_params
        self.network = network
        self.images = images
        self.images = self.image_scaling(self.images)
        if self.params['IMAGE_FP16']: #and self.images.dtype is not tf.float16 and operation == 'train':
            self.images = tf.cast(self.images, tf.float16)
        else:
            self.images = tf.cast(self.images, tf.float32)
        image_shape = images.get_shape().as_list()
        if self.params['TENSOR_FORMAT'] != 'NCHW' :
            # change from NHWC to NCHW format
            # TODO: add flag to swith between 2 ....
            self.images = tf.transpose(self.images, perm=[0, 3, 1, 2])
        # self.images = self.get_glimpses(self.images)
        self.labels = labels
        self.net_type = self.hyper_params['network_type']
        self.operation = operation
        self.summary = summary
        self.verbose = verbose
        self.num_weights = 0
        self.misc_ops = []
        self.reuse = tf.AUTO_REUSE
        # self.reuse = None
        if self.operation == 'eval_run' or self.operation == 'eval_ckpt':
            # self.reuse = True
            self.operation == 'eval'
        self.bytesize = 2
        if not self.params['IMAGE_FP16']: self.bytesize = 4
        self.mem = np.prod(self.images.get_shape().as_list()) * self.bytesize/1024  # (in KB)
        self.ops = 0
        if "batch_norm" in self.hyper_params:
            self.hyper_params["batch_norm"]["decay"] = self.hyper_params["batch_norm"].get("decay", 0.995)
            self.hyper_params["batch_norm"]["epsilon"] = self.hyper_params["batch_norm"].get("epsilon", 1E-5)
        else:
            # default params
            self.hyper_params["batch_norm"] = {"epsilon": 1E-5, "decay": 0.995}
        self.model_output = None
        self.scopes = []

        # self.initializer = self._get_initializer(hyper_params.get('initializer', None))

    def print_rank(self, *args, **kwargs):
        if hvd.rank() == 0 and self.operation == 'train':
            print(*args, **kwargs)

    def print_verbose(self, *args, **kwargs):
        if self.verbose:
            self.print_rank(*args, **kwargs)

    def build_model(self, summaries=False):
        """
        Here we build the model.
        :param summaries: bool, flag to print out summaries.
        """
        self.initializer = self._get_initializer(self.hyper_params.get('initializer', None))
        # Initiate 1st layer
        self.print_rank('Building Neural Net ...')
        self.print_rank('input: ---, dim: %s memory: %s MB' %(format(self.images.get_shape().as_list()), format(self.mem/1024)))
        layer_name, layer_params = list(self.network.items())[0]
        with tf.variable_scope(layer_name, reuse=self.reuse) as scope:
            if layer_params['type'] == 'coord_conv':
                out, kernel = self._coord_conv(input=self.images, params=layer_params)
            else:
                out, kernel = self._conv(input=self.images, params=layer_params)
            do_bn = layer_params.get('batch_norm', False)
            if do_bn:
                out = self._batch_norm(input=out)
            else:
                out = self._add_bias(input=out, params=layer_params)
            out = self._activate(input=out, name=scope.name, params=layer_params)
            in_shape = self.images.get_shape().as_list()
            # Tensorboard Summaries
            if self.summary:
                self._activation_summary(out)
                self._activation_image_summary(out)
                self._kernel_image_summary(kernel)

            self._print_layer_specs(layer_params, scope, in_shape, out.get_shape().as_list())
            self.scopes.append(scope)

        # Initiate the remaining layers
        for layer_name, layer_params in list(self.network.items())[1:]:
            with tf.variable_scope(layer_name, reuse=self.reuse) as scope:
                in_shape = out.get_shape().as_list()
                if layer_params['type'] == 'conv_2D':
                    out, _ = self._conv(input=out, params=layer_params)
                    do_bn = layer_params.get('batch_norm', False)
                    if do_bn:
                        out = self._batch_norm(input=out)
                    else:
                        out = self._add_bias(input=out, params=layer_params)
                    out = self._activate(input=out, name=scope.name, params=layer_params)
                    if self.summary: self._activation_summary(out)
                
                if layer_params['type'] == 'coord_conv':
                    self.print_verbose(">>> Adding Coord Conv Layer: %s" % layer_name)
                    self.print_verbose('    input: %s' %format(out.get_shape().as_list()))
                    out, _ = self._coord_conv(input=out, params=layer_params)
                    self.print_verbose('    output: %s' %format(out.get_shape().as_list()))
                    if self.summary: self._activation_summary(out)

                if layer_params['type'] == 'pooling':
                    out = self._pool(input=out, name=scope.name, params=layer_params)

                if layer_params['type'] not in ['conv_2D', 'coord_conv', 'pooling', 'fully_connected', 'linear_output']:
                    out = self._compound_layer(out, layer_params, scope)
                    # Continue any summary
                    if self.summary: self._activation_summary(out)

                # print layer specs and generate Tensorboard summaries
                if out is None:
                    raise NotImplementedError('Layer type: ' + layer_params['type'] + ' was not implemented!')
                out_shape = out.get_shape().as_list()
                self._print_layer_specs(layer_params, scope, in_shape, out_shape)
            self.scopes.append(scope)
        self.print_rank('Total # of blocks: %d,  weights: %2.1e, memory: %s MB, ops: %3.2e \n' % (len(self.network),
                                                                                        self.num_weights,
                                                                                        format(self.mem / 1024),
                                                                                        self.get_ops()))
        self.model_output = out

    def _compound_layer(self, out, layer_params, scope):
        """
        Handles the computation of more complex layer types such as Residual blocks, Inception, etc.

        :param out: 4D tensor, Input to the layer
        :param layer_params: OrderedDictionary, Parameters for the layer
        :param scope: str, name of the layer
        :return: 4D tensor, output of the layer
        """
        pass

    # @staticmethod
    def _get_initializer(self, params):
        """
        Returns an Initializer object for initializing weights

        Note - good reference:
         - https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/python/keras/_impl/keras/initializers.py
        :param params: dictionary
        :return: initializer object
        """
        if params is not None:
            if isinstance(params, dict):
                params_copy = params.copy()
                name = str(params_copy.pop('type').lower())
                if name == 'uniform_unit_scaling':
                    # self.print_verbose('using ' + name + ' initializer')
                    # Random walk initialization (currently in the code).
                    return tf.initializers.variance_scaling(distribution="uniform")
                elif name == 'truncated_normal':
                    # self.print_verbose('using ' + name + ' initializer')
                    return tf.truncated_normal_initializer(**params_copy)
                elif name == 'glorot_uniform':
                    # self.print_verbose('using ' + name + ' initializer')
                    return tf.glorot_uniform_initializer()
                elif name == 'variance_scaling':
                    # self.print_verbose('using ' + name + ' initializer')
                    return tf.initializers.variance_scaling(distribution="normal")
                elif name == 'random_normal':
                    # self.print_verbose('using ' + name + ' initializer')
                    # Normalized Initialization ( eq. 16 in Glorot et al.).
                    return tf.random_normal_initializer(**params_copy)
                elif name == 'random_uniform':
                    # self.print_verbose('using ' + name + ' initializer')
                    return tf.random_uniform_initializer(**params_copy)
                elif name == 'xavier':  # Glorot uniform initializer, also called Xavier
                    # http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
                    # self.print_verbose('using ' + name + ' initializer')
                    return tf.contrib.layers.xavier_initializer(**params_copy)
                elif name in ['he', 'lecun']:
                    """
                    Note that tf.variance_scaling_initializer and tf.contrib.layers.variance_scaling_initializer
                    take the same kinds of parameters with different names and formats.

                    However, tf.variance_scaling_initializer doesn't seem to be available on TF 1.2.1 on the DGX1
                    """
                    params_copy['factor'] = params_copy.pop('scale', 1)
                    params_copy['uniform'] = params_copy.pop('distribution', True)
                    if 'uniform' in params_copy:
                        if isinstance(params_copy['uniform'], str):
                            if params_copy['uniform'].lower() == 'uniform':
                                params_copy['uniform'] = True
                            else:
                                params_copy['uniform'] = False
                    if name == 'he':
                        # He et al., http://arxiv.org/abs/1502.01852
                        _ = params_copy.pop('factor', None)  # force it to be 2.0 (default anyway)
                        _ = params_copy.pop('mode', None)  # force it to be 'FAN_IN' (default anyway)
                        # uniform parameter is False by default -> normal distribution
                        # self.print_verbose('using ' + name + ' initializer')
                        return tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', **params_copy)
                    elif name == 'lecun':
                        _ = params_copy.pop('factor', None)  # force it to be 1.0
                        _ = params_copy.pop('mode', None)  # force it to be 'FAN_IN' (default anyway)
                        # uniform parameter is False by default -> normal distribution
                        # self.print_verbose('using ' + name + ' initializer')
                        return tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN', **params_copy)
                    self.print_verbose('Requested initializer: ' + name + ' has not yet been implemented.')
        # default = Xavier:
        # self.print_verbose('Using default Xavier instead')
        return tf.contrib.layers.xavier_initializer()


    def get_loss(self):
        # with tf.variable_scope(self.scope, reuse=self.reuse) as scope:
        if self.net_type == 'hybrid': self._calculate_loss_hybrid()
        if self.net_type == 'regressor': self._calculate_loss_regressor()
        if self.net_type == 'classifier' : self._calculate_loss_classifier()
        if self.hyper_params['langevin'] : self.add_stochastic_layer()

    def get_output(self):
        layer_params={'bias':self.labels.get_shape().as_list()[-1], 'weights':self.labels.get_shape().as_list()[-1],
            'regularize':True}
        with tf.variable_scope('linear_output', reuse=self.reuse) as scope:
            output = self._linear(input=self.model_output, name=scope.name, params=layer_params)
            print(output.name)
        if self.params['IMAGE_FP16']:
            output = tf.cast(output, tf.float32)
            return output
        return output

    #TODO: return ops per type of layer
    def get_ops(self):
        return 3*self.ops # 3 is for derivate w/t kernel + derivative w/t data + conv (*ignoring everything else eventhough they're calculated)

    def get_misc_ops(self):
        ops = tf.group(*self.misc_ops)
        return ops

    # Loss calculation and regularization helper methods

    def _calculate_loss_hybrid(self):
        dim = self.labels.get_shape().as_list()[-1]
        num_classes = self.params['NUM_CLASSES']
        if self.hyper_params['langevin']:
            class_labels = self.labels
            if class_labels.dtype is not tf.int64:
                class_labels = tf.cast(class_labels, tf.int64)
            regress_labels = tf.random_normal(class_labels.get_shape().as_list(), stddev=0.01, dtype=tf.float64)
        else:
            regress_labels, class_labels = tf.split(self.labels,[dim-num_classes, num_classes],1)
        outputs = []
        for layer_name, labels in zip(['linear_output', 'stochastic'],
                                            [class_labels, regress_labels]):
            layer_params={'bias':labels.get_shape().as_list()[-1], 'weights':labels.get_shape().as_list()[-1],
                'regularize':True}
            with tf.variable_scope(layer_name, reuse=self.reuse) as scope:
                out = tf.cast(self._linear(input=self.model_output, name=scope.name, params=layer_params), tf.float32)
                print(out.name)
            self.print_rank('Output Layer : %s' %format(out.get_shape().as_list()))
            outputs.append(out)
        mixing = self.hyper_params['mixing']
        cost = (1-mixing)*self._calculate_loss_classifier(net_output=outputs[0], labels=class_labels) + \
                        mixing*self._calculate_loss_regressor(net_output=outputs[1],
                        labels=regress_labels, weight=mixing)
        return cost

    def _calculate_loss_regressor(self, net_output=None, labels=None, weight=None):
        """
        Calculate the loss objective for regression
        :param params: dictionary, specifies the objective to use
        :return: cost
        """
        if net_output is None:
            net_output = self.get_output()
        if labels is None:
            labels = self.labels
        if weight is None:
            weight = 1.0
        params = self.hyper_params['loss_function']
        assert params['type'] == 'Huber' or params['type'] == 'MSE' \
        or params['type'] == 'LOG', "Type of regression loss function must be 'Huber' or 'MSE'"
        if params['type'] == 'Huber':
            # decay the residual cutoff exponentially
            decay_steps = int(self.params['NUM_EXAMPLES_PER_EPOCH'] / self.params['batch_size'] \
                              * params['residual_num_epochs_decay'])
            initial_residual = params['residual_initial']
            min_residual = params['residual_minimum']
            decay_residual = params['residual_decay_factor']
            residual_tol = tf.train.exponential_decay(initial_residual, self.global_step, decay_steps,
                                                      decay_residual, staircase=False)
            # cap the residual cutoff to some min value.
            residual_tol = tf.maximum(residual_tol, tf.constant(min_residual))
            if self.summary:
                tf.summary.scalar('residual_cutoff', residual_tol)
            # calculate the cost
            cost = tf.losses.huber_loss(labels, weights=weight, predictions=net_output, delta=residual_tol,
                                        reduction=tf.losses.Reduction.MEAN)
        if params['type'] == 'MSE':
            cost = tf.losses.mean_squared_error(labels, weights=weight, predictions=net_output,
                                                reduction=tf.losses.Reduction.MEAN)
        if params['type'] == 'LOG':
            cost = tf.losses.log_loss(labels, weights=weight, predictions=net_output, reduction=tf.losses.Reduction.MEAN)
        return cost

    def _calculate_loss_classifier(self, net_output=None, labels=None, weight=None):
        """
        Calculate the loss objective for classification
        :param params: dictionary, specifies the objective to use
        :return: cost
        """
        if labels is None:
            labels = self.labels
        if labels.dtype is not tf.int64:
            labels = tf.cast(labels, tf.int64)
        if net_output is None:
            net_output = self.get_output()
        if weight is None:
            weight = 1.0
        labels = tf.argmax(labels, axis=1)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=net_output)
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        precision_1 = tf.scalar_mul(1. / self.params['batch_size'],
                                    tf.reduce_sum(tf.cast(tf.nn.in_top_k(net_output, labels, 1), tf.float32)))
        precision_5 = tf.scalar_mul(1. / self.params['batch_size'],
                                    tf.reduce_sum(tf.cast(tf.nn.in_top_k(net_output, labels, 5), tf.float32)))
        if self.summary :
            tf.summary.scalar('precision@1_train', precision_1)
            tf.summary.scalar('precision@5_train', precision_5)
        tf.add_to_collection(tf.GraphKeys.LOSSES, cross_entropy_mean)
        return cross_entropy_mean

    # Network layers helper methods
    def _conv(self, input=None, params=None):
        """
        Builds 2-D convolutional layer
        :param input: as it says
        :param params: dict, must specify kernel shape, stride, and # of features.
        :return: output of convolutional layer and filters
        """
        stride_shape = [1,1]+list(params['stride'])
        features = params['features']
        kernel_shape = list(params['kernel']) + [input.shape[1].value, features]

        # Fine tunining the initializer:
        conv_initializer = self._get_initializer(self.hyper_params.get('initializer', None))
        if isinstance(conv_initializer, tf.truncated_normal_initializer):
            init_val = np.sqrt(2.0/(kernel_shape[0] * kernel_shape[1] * features))
            conv_initializer.stddev = init_val
        elif isinstance(conv_initializer, tf.uniform_unit_scaling_initializer):
            conv_initializer.factor = 1.43
        elif isinstance(conv_initializer, tf.random_normal_initializer):
            init_val = np.sqrt(2.0/(kernel_shape[0] * kernel_shape[1] * features))

            #self.print_verbose('stddev: %s' % format(init_val))
            conv_initializer.mean = 0.0
            conv_initializer.stddev = init_val
        # TODO: make and modify local copy only

        kernel = self._cpu_variable_init('weights', shape=kernel_shape, initializer=conv_initializer)
        output = tf.nn.conv2d(input, kernel, stride_shape, data_format='NCHW', padding=params['padding'])

        # Keep tabs on the number of weights and memory
        self.num_weights += np.cumprod(kernel_shape)[-1]
        self.mem += np.cumprod(output.get_shape().as_list())[-1]*self.bytesize / 1024
        # batch * width * height * in_channels * kern_h * kern_w * features
        # input = batch_size (ignore), channels, height, width
        # http://imatge-upc.github.io/telecombcn-2016-dlcv/slides/D2L1-memory.pdf
        # this_ops = np.prod(params['kernel'] + input.get_shape().as_list()[1:] + [features])
        # self.print_rank('\tops: %3.2e' % (this_ops))
        """
        # batch * width * height * in_channels * (kern_h * kern_w * channels)
        # at each location in the image:
        ops_per_conv = 2 * np.prod(params['kernel'] + [input.shape[1].value])
        # number of convolutions on the image for a single filter / output channel (stride brings down the number)
        convs_per_filt = np.prod([input.shape[2].value, input.shape[3].value]) // np.prod(params['stride'])
        # final = filters * convs/filter * ops/conv
        this_ops = np.prod([params['features'], convs_per_filt, ops_per_conv])
        if verbose:
            self.print_verbose('\t%d ops/conv, %d convs/filter, %d filters = %3.2e ops' % (ops_per_conv, convs_per_filt,
                                                                              params['features'], this_ops))
        """
        # 2*dim(input=N*H*W)*dim(kernel=H*W*N)
        this_ops = 2 * np.prod(params['kernel']) * features * np.prod(input.get_shape().as_list()[1:])
        self.ops += this_ops

        return output, kernel

    def _depth_conv(self, input=None, params=None):
        """
        Builds 2-D depthwise convolutional layer
        :param input: as it says
        :param params: dict, must specify kernel shape, stride, and # of features.
        :return: output of convolutional layer and filters
        """
        stride_shape = [1,1]+list(params['stride'])
        features = params['features']
        kernel_shape = list(params['kernel']) + [input.shape[1].value, features]

        # Fine tunining the initializer:
        conv_initializer = self._get_initializer(self.hyper_params.get('initializer', None))
        if isinstance(conv_initializer, tf.truncated_normal_initializer):
            init_val = np.sqrt(2.0/(kernel_shape[0] * kernel_shape[1] * features))
            conv_initializer.stddev = init_val
        elif isinstance(conv_initializer, tf.uniform_unit_scaling_initializer):
            conv_initializer.factor = 1.43
        elif isinstance(conv_initializer, tf.random_normal_initializer):
            init_val = np.sqrt(2.0/(kernel_shape[0] * kernel_shape[1] * features))

            # self.print_verbose('stddev: %s' % format(init_val))
            conv_initializer.mean = 0.0
            conv_initializer.stddev = init_val
        # TODO: make and modify local copy only

        kernel = self._cpu_variable_init('weights', shape=kernel_shape, initializer=conv_initializer)
        output = tf.nn.depthwise_conv2d(input, kernel, stride_shape, data_format='NCHW', padding=params['padding'])

        # Keep tabs on the number of weights and memory
        self.num_weights += np.cumprod(kernel_shape)[-1]
        self.mem += np.cumprod(output.get_shape().as_list())[-1]*self.bytesize / 1024
        # batch * width * height * in_channels * kern_h * kern_w * features
        # input = batch_size (ignore), channels, height, width
        # http://imatge-upc.github.io/telecombcn-2016-dlcv/slides/D2L1-memory.pdf
        # this_ops = np.prod(params['kernel'] + input.get_shape().as_list()[1:] + [features])
        # self.print_rank('\tops: %3.2e' % (this_ops))
        """
        # batch * width * height * in_channels * (kern_h * kern_w * channels)
        # at each location in the image:
        ops_per_conv = 2 * np.prod(params['kernel'] + [input.shape[1].value])
        # number of convolutions on the image for a single filter / output channel (stride brings down the number)
        convs_per_filt = np.prod([input.shape[2].value, input.shape[3].value]) // np.prod(params['stride'])
        # final = filters * convs/filter * ops/conv
        this_ops = np.prod([params['features'], convs_per_filt, ops_per_conv])
        if verbose:
            self.print_verbose('\t%d ops/conv, %d convs/filter, %d filters = %3.2e ops' % (ops_per_conv, convs_per_filt,
                                                                              params['features'], this_ops))
        """
        # 2*dim(input=N*H*W)*dim(kernel=H*W*N)
        this_ops = 2 * np.prod(params['kernel']) * features * np.prod(input.get_shape().as_list()[1:])
        self.ops += this_ops

        return output, kernel

    def _deconv(self, input=None, params=None, verbose=True):
        """
        Builds 2-D deconvolutional layer
        :param input: as it says
        :param params: dict, must specify kernel shape, stride, and # of features.
        :return: output of deconvolutional layer and filters
        """
        stride_shape = [1,1]+list(params['stride'])
        features = params['features']
        kernel_shape = list(params['kernel']) + [features, input.shape[1].value]

        # Fine tunining the initializer:
        conv_initializer = self._get_initializer(self.hyper_params.get('initializer', None))
        if isinstance(conv_initializer, tf.truncated_normal_initializer):
            init_val = np.sqrt(2.0/(kernel_shape[0] * kernel_shape[1] * features))
            conv_initializer.stddev = init_val
        elif isinstance(conv_initializer, tf.uniform_unit_scaling_initializer):
            conv_initializer.factor = 1.43
        elif isinstance(conv_initializer, tf.random_normal_initializer):
            init_val = np.sqrt(2.0/(kernel_shape[0] * kernel_shape[1] * features))
            # if verbose:
                # self.print_verbose('stddev: %s' % format(init_val))
            conv_initializer.mean = 0.0
            conv_initializer.stddev = init_val
        # TODO: make and modify local copy only
        upsample = params['upsample']
        output_shape = [input.shape[0].value, features, input.shape[2].value*upsample, input.shape[2].value*upsample]
        kernel = self._cpu_variable_init('weights', shape=kernel_shape, initializer=conv_initializer)
        output = tf.nn.conv2d_transpose(input, kernel, output_shape, stride_shape, data_format='NCHW', padding=params['padding'])

        # Keep tabs on the number of weights, memory, and flops
        self.num_weights += np.cumprod(kernel_shape)[-1]
        self.mem += np.cumprod(output.get_shape().as_list())[-1]*self.bytesize / 1024
        # 2*dim(input=N*H*W)*dim(kernel=H*W*N)
        this_ops = 2 * np.prod(params['kernel']) * features * np.prod(input.get_shape().as_list()[1:])
        self.ops += this_ops

        return output, kernel

    def _coord_conv(self, input=None, params=None):
        """
        Builds 2-D coord convolutional layer
        :param input: as it says
        :param params: dict, must specify kernel shape, stride, and # of features.
        :return: output of convolutional layer and filters
        """
        stride_shape = [1,1]+list(params['stride'])
        features = params['features'] - 2
        kernel_shape = list(params['kernel']) + [input.shape[1].value, features]
        kernel_shape[-2] += 2 
        kernel_shape[-1] += 2

        # Fine tunining the initializer:
        conv_initializer = self._get_initializer(self.hyper_params.get('initializer', None))
        if isinstance(conv_initializer, tf.truncated_normal_initializer):
            init_val = np.sqrt(2.0/(kernel_shape[0] * kernel_shape[1] * features))
            conv_initializer.stddev = init_val
        elif isinstance(conv_initializer, tf.uniform_unit_scaling_initializer):
            conv_initializer.factor = 1.43
        elif isinstance(conv_initializer, tf.random_normal_initializer):
            init_val = np.sqrt(2.0/(kernel_shape[0] * kernel_shape[1] * features))

            # self.print_verbose('stddev: %s' % format(init_val))
            conv_initializer.mean = 0.0
            conv_initializer.stddev = init_val
        # TODO: make and modify local copy only

        kernel = self._cpu_variable_init('weights', shape=kernel_shape, initializer=conv_initializer)

        batch_size = tf.shape(input)[0]
        x_dim = tf.shape(input)[-1]
        y_dim  = tf.shape(input)[-2]

        xx_indices = tf.tile(
            tf.expand_dims(tf.expand_dims(tf.range(x_dim), 0), 0),
            [batch_size, y_dim, 1])
        xx_indices = tf.expand_dims(xx_indices, -1)
        xx_indices = tf.transpose(xx_indices, (0,3,1,2))
        
        yy_indices = tf.tile(
            tf.expand_dims(tf.reshape(tf.range(y_dim), (y_dim, 1)), 0),
            [batch_size, 1, x_dim])
        yy_indices = tf.expand_dims(yy_indices, -1)
        yy_indices = tf.transpose(yy_indices, (0,3,1,2))

        xx_indices = tf.divide(xx_indices, x_dim - 1)
        yy_indices = tf.divide(yy_indices, y_dim - 1)

        xx_indices = tf.cast(tf.subtract(tf.multiply(xx_indices, 2.), 1.),
                                dtype=input.dtype)
        yy_indices = tf.cast(tf.subtract(tf.multiply(yy_indices, 2.), 1.),
                                dtype=input.dtype)

        output = tf.concat([input, xx_indices, yy_indices], axis=1)

        output = tf.nn.conv2d(output, kernel, stride_shape, data_format='NCHW', padding=params['padding'])

        # Keep tabs on the number of weights and memory
        self.num_weights += np.cumprod(kernel_shape)[-1]
        self.mem += np.cumprod(output.get_shape().as_list())[-1]*self.bytesize / 1024

        # 2*dim(input=N*H*W)*dim(kernel=H*W*N)
        this_ops = 2 * np.prod(params['kernel']) * features * np.prod(input.get_shape().as_list()[1:])
        self.ops += this_ops

        return output, kernel

    def _add_bias(self, input=None, params=None):
        """
        Adds bias to a convolutional layer.
        :param input:
        :param params:
        :return:
        """
        bias_shape = input.shape[-1].value
        bias = self._cpu_variable_init('bias', shape=bias_shape, initializer=tf.zeros_initializer())
        output = tf.nn.bias_add(input, bias)

        # Keep tabs on the number of bias parameters and memory
        self.num_weights += bias_shape
        self.mem += bias_shape*self.bytesize / 1024
        # self.ops += bias_shape
        return output

    def _batch_norm(self, input=None, reuse=None):
        """
        Batch normalization
        :param input: as it says
        :return:
        """
        # Initializing hyper_parameters
        shape = [input.shape[1].value]
        epsilon = self.hyper_params["batch_norm"]["epsilon"]
        decay = self.hyper_params["batch_norm"]["decay"]
        decay = 0.9
        is_training = 'train' == self.operation
        # TODO: scaling and centering during normalization need to be hyperparams. Now hardwired.
        param_initializers={
              'beta': tf.constant_initializer(0.0),
              'gamma': tf.constant_initializer(0.1),
        }
        output = tf.contrib.layers.batch_norm(input, decay=decay, scale=True, epsilon=epsilon,zero_debias_moving_mean=False,is_training=is_training,fused=True,data_format='NCHW',renorm=False,param_initializers=param_initializers)
        #output = tf.contrib.layers.batch_norm(input, decay=decay, scale=True, epsilon=epsilon,zero_debias_moving_mean=False,is_training=is_training,fused=True,data_format='NCHW',renorm=False)
        # output = input
        # Keep tabs on the number of weights
        self.num_weights += 2 * shape[0]  # scale and offset (beta, gamma)
        # consistently ignored by most papers / websites for ops calculation
        return output

    def _linear(self, input=None, params=None, name=None, verbose=True):
        """
        Linear layer
        :param input:
        :param params:
        :return:
        """
        #assert params['weights'] == params['bias'], " weights and bias outer dimensions do not match"
        #input_reshape = tf.reshape(input,[self.params['batch_size'], -1])
        input_shape  = input.shape.as_list()
        if len(input_shape) > 2:
            input_reshape = tf.reshape(input,[self.params['batch_size'], -1])
        else:
            input_reshape = input 
        dim_input = input_reshape.shape[1].value
        if params['weights'] is not None:
            weights_shape = [dim_input, params['weights']]
            init_val = max(np.sqrt(2.0/params['weights']), 0.01)
        # self.print_verbose('stddev: %s' % format(init_val))
            bias_shape = [params['bias']]
        else:
            weights_shape = [dim_input, dim_input]
            init_val = max(np.sqrt(2.0/dim_input), 0.01)
        # self.print_verbose('stddev: %s' % format(init_val))
            bias_shape = [dim_input]
            params['weights'] = dim_input
            params['bias'] = dim_input
            
        # Fine tuning the initializer:
        lin_initializer = self._get_initializer(self.hyper_params.get('initializer', None))
        # lin_initializer = tf.glorot_uniform_initializer 
        if isinstance(lin_initializer, tf.uniform_unit_scaling_initializer):
            if params['type'] == 'fully_connected':
                if params['activation'] == 'tanh':
                    lin_initializer.factor = 1.15
                elif params['activation'] == 'relu':
                    lin_initializer.factor = 1.43
            elif params['type'] == 'linear_output':
                lin_initializer.factor = 1.0
        elif isinstance(lin_initializer, tf.random_normal_initializer):
            init_val = max(np.sqrt(2.0 / params['weights']), 0.01)
            if verbose:
                print('stddev: %s' % format(init_val))
            lin_initializer.mean = 0.0
            lin_initializer.stddev = init_val

        weights = self._cpu_variable_init('weights', shape=weights_shape, initializer=lin_initializer,
                                          regularize=params['regularize'])
        bias = self._cpu_variable_init('bias', bias_shape, initializer=tf.zeros_initializer)
        output = tf.nn.bias_add(tf.matmul(input_reshape, weights), bias, name=name)

        # Keep tabs on the number of weights and memory
        self.num_weights += bias_shape[0] + np.cumprod(weights_shape)[-1]
        self.mem += np.cumprod(output.get_shape().as_list())[-1] * self.bytesize / 1024
        # this_ops = 2 * params['weights'] + params['bias']
        # self.ops += this_ops
        return output

    def _dropout(self, input=None, keep_prob=0.5, params=None, name=None):
        """
        Performs dropout
        :param input:
        :param params:
        :param name:
        :return:
        """
        return tf.nn.dropout(input, keep_prob=tf.constant(keep_prob, dtype=input.dtype))

    def _activate(self, input=None, params=None, name=None, verbose=False):
        """
        Activation
        :param input: as it says
        :param params: dict, must specify activation type
        :param name: scope.name
        :return:
        """
        # should ignore the batch size in the calculation!
        # this_ops = 2 * np.prod(input.get_shape().as_list()[1:])
        # if verbose:
            # self.print_verbose('\tactivation = %3.2e ops' % this_ops)
        # self.ops += this_ops

        if params is not None:
            if params['activation'] == 'tanh':
                return tf.nn.tanh(input, name=name)
            else:
                return tf.nn.relu(input, name=name)
        else:
            return input

    def _pool(self, input=None, params=None, name=None, verbose=True):
        """
        Pooling
        :param params: dict, must specify type of pooling (max, average), stride, and kernel size
        :return:
        """
        stride_shape = [1,1]+params['stride']
        kernel_shape = [1,1]+params['kernel']
        if params['pool_type'] == 'max':
            output = tf.nn.max_pool(input, kernel_shape, stride_shape, params['padding'], name=name, data_format='NCHW')
        if params['pool_type'] == 'avg':
            output = tf.nn.avg_pool(input, kernel_shape, stride_shape, params['padding'], name=name, data_format='NCHW')

        # Keep tabs on memory
        self.mem += np.cumprod(output.get_shape().as_list())[-1] * self.bytesize / 1024

        # at each location in the image:
        # avg: 1 to sum each of the N element, 1 op for avg
        # max: N max() operations
        ops_per_pool = 1 * np.prod(params['kernel'] + [input.shape[1].value])
        # number of pools on the image for a single filter / output channel (stride brings down the number)
        num_pools = np.prod([input.shape[2].value, input.shape[3].value]) // np.prod(params['stride'])
        # final = num images * filters * convs/filter * ops/conv
        # self.print_verbose('\t%d ops/pool, %d pools = %3.2e ops' % (ops_per_pool, num_pools,
        #                                                    num_pools * ops_per_pool))

        self.ops += num_pools * ops_per_pool

        return output

    # Summary helper methods
    @staticmethod
    def _activation_summary(x):
        """Helper to create summaries for activations.

         Creates a summary that provides a histogram of activations.
         Creates a summary that measures the sparsity of activations.

         Args:
           x: Tensor
         Returns:
           nothing
        """
        # Remove 'worker_[0-9]/' from the name in Tensorboard.
        tensor_name = re.sub('%s_[0-9]*/' % worker_name, '', x.name)
        tf.summary.histogram(tensor_name + '/activations', x)
        tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

    @staticmethod
    def _activation_image_summary(image_stack, n_features=None):
        """ Helper to show images of activation maps in summary.

        Args:
            image_stack: Tensor, 4-d output of conv/pool layer
            n_features: int, # of featuers to display, Optional, default is half of features depth.
        Returns:
            Nothing
        """

        # Transpose to NHWC
        image_stack = tf.transpose(image_stack, perm=[0, 2, 3, 1])
        #
        tensor_name = re.sub('%s_[0-9]*/' % worker_name, '', image_stack.name)
        # taking only first 3 images from batch
        if n_features is None:
            # nFeatures = int(pool.shape[-1].value /2)
            n_features = -1
        for ind in range(1):
            map = tf.slice(image_stack, (ind, 0, 0, 0), (1, -1, -1, n_features))
            map = tf.reshape(map, (map.shape[1].value, map.shape[2].value, map.shape[-1].value))
            map = tf.transpose(map, (2, 0 , 1))
            map = tf.reshape(map, (-1, map.shape[1].value, map.shape[2].value, 1))

            # Tiling
            nOfSlices = map.shape[0].value
            n = int(np.ceil(np.sqrt(nOfSlices)))
            # padding by 4 pixels
            padding = [[0, n ** 2 - nOfSlices], [0, 4], [0, 4], [0, 0]]
            map_padded = tf.pad(map, paddings=padding)
            # reshaping and transposing gymnastics ...
            new_shape = (n, n) + (map_padded.shape[1].value, map_padded.shape[2].value, 1)
            map_padded = tf.reshape(map_padded, new_shape)
            map_padded = tf.transpose(map_padded, perm=(0, 2, 1, 3, 4))
            new_shape = (n * map_padded.shape[1].value, n * map_padded.shape[3].value, 1)
            map_tile = tf.reshape(map_padded, new_shape)
            # Convert to 4-d
            map_tile = tf.expand_dims(map_tile,0)
            map_tile = tf.log1p(map_tile)
            # Display feature maps
            tf.summary.image(tensor_name + '/activation'+ str(ind), map_tile)

    @staticmethod
    def _kernel_image_summary(image_stack, n_features=None):
        """ Helper to show images of activation maps in summary.

        Args:
            image_stack: Tensor, 4-d output of conv/pool layer
            n_features: int, # of featuers to display, Optional, default is half of features depth.
        Returns:
            Nothing
        """
        # Remove 'worker_[0-9]/' from the name in Tensorboard.
        tensor_name = re.sub('%s_[0-9]*/' % worker_name, '', image_stack.name)
        if n_features is None:
            n_features = -1
        map = tf.slice(image_stack, (0, 0, 0, 0), (-1, -1, -1, n_features))
        # self.print_rank('activation map shape: %s' %(format(map.shape)))
        map = tf.reshape(map, (map.shape[0].value, map.shape[1].value, map.shape[-2].value*map.shape[-1].value))
        map = tf.transpose(map, (2, 0, 1))
        map = tf.reshape(map, (-1, map.shape[1].value, map.shape[2].value, 1))
        # color_maps = tf.image.grayscale_to_rgb(map)
        # Tiling
        nOfSlices = map.shape[0].value
        n = int(np.ceil(np.sqrt(nOfSlices)))
        # padding by 4 pixels
        padding = [[0, n ** 2 - nOfSlices], [0, 4], [0, 4], [0, 0]]
        map_padded = tf.pad(map, paddings=padding)
        # reshaping and transposing gymnastics ...
        new_shape = (n, n) + (map_padded.shape[1].value, map_padded.shape[2].value, 1)
        map_padded = tf.reshape(map_padded, new_shape)
        map_padded = tf.transpose(map_padded, perm=(0, 2, 1, 3, 4))
        new_shape = (n * map_padded.shape[1].value, n * map_padded.shape[3].value, 1)
        map_tile = tf.reshape(map_padded, new_shape)
        # Convert to 4-d
        map_tile = tf.expand_dims(map_tile, 0)
        map_tile = tf.log1p(map_tile)
        # Display feature maps
        tf.summary.image(tensor_name + '/kernels' , map_tile)

    def _print_layer_specs(self, params, scope, input_shape, output_shape):
        mem_in_MB = np.cumprod(output_shape)[-1] * self.bytesize / 1024**2
        if 'conv' in params['type'] :
            self.print_verbose('%s --- output: %s, kernel: %s, stride: %s, # of weights: %s,  memory: %s MB' %
                  (scope.name, format(output_shape), format(params['kernel']),
                   format(params['stride']), format(self.num_weights), format(mem_in_MB)))
        if params['type'] == 'pooling':
            self.print_verbose('%s --- output: %s, kernel: %s, stride: %s, memory: %s MB' %
                  (scope.name, format(output_shape), format(params['kernel']),
                   format(params['stride']), format(mem_in_MB)))
        if params['type'] == 'fully_connected' or params['type'] == 'linear_output':
            self.print_verbose('%s --- output: %s, weights: %s, bias: %s, # of weights: %s,  memory: %s MB' %
                   (scope.name, format(output_shape), format(params['weights']),
                     format(params['bias']), format(self.num_weights), format(mem_in_MB)))

    def _add_loss_summaries(self, total_loss, losses):
        """
        Add summaries for losses in model.
        Generates moving average for all losses and associated summaries for
        visualizing the performance of the network.
        :param total_loss:
        :param losses:
        :return: loss_averages_op
        """
        # Compute the moving average of all individual losses and the total loss.
        loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
        loss_averages_op = loss_averages.apply(losses + [total_loss])

        # Attach a scalar summary to all individual losses and the total loss;
        if self.summary:
            for l in losses + [total_loss]:
                # Name each loss as '(raw)' and name the moving average version of the loss
                # as the original loss name.
                loss_name = re.sub('%s_[0-9]*/' % worker_name, '', l.op.name)
                tf.summary.scalar(loss_name + ' (raw)', l)
                tf.summary.scalar(loss_name, loss_averages.average(l))

        return loss_averages_op

    def _json_summary(self):
        """
        Generate text summary out of *.json file input
        :return: None
        """
        net_list = [[key, str([self.network[key]])] for key in self.network.iterkeys()]
        hyp_list = [[key, str([self.hyper_params[key]])] for key in self.hyper_params.iterkeys()]
        net_config = tf.constant(net_list, name='network_config')
        hyp_params = tf.constant(hyp_list, name='hyper_params')
        tf.summary.text(net_config.op.name, net_config)
        tf.summary.text(hyp_params.op.name, hyp_params)
        return None

    # Variable placement, initialization, regularization
    def _cpu_variable_init(self, name, shape, initializer, trainable=True, regularize=True):
        """Helper to create a Variable stored on CPU memory.

        Args:
          name: name of the variable
          shape: list of ints
          initializer: initializer for Variable

        Returns:
          Variable Tensor
        """
        # if self.params['IMAGE_FP16'] and self.operation == 'train':
        if self.params['IMAGE_FP16']:
            dtype = tf.float16
        else:
            dtype = tf.float32

        if regularize:
            var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype, trainable=trainable,
                                  regularizer=self._weight_decay)
        else:
            var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype, trainable=trainable)

        return var


    def _weight_decay(self, tensor):
        return tf.multiply(tf.nn.l2_loss(tf.cast(tensor, tf.float32)), self.hyper_params['weight_decay'])

    def get_glimpses(self, batch_images):
        """
        Get bounded glimpses from images, corresponding to ~ 2x1 supercell
        :param batch_images: batch of training images
        :return: batch of glimpses
        """
        if self.params['glimpse_mode'] not in ['uniform', 'normal', 'fixed']:
            """
            print('No image glimpsing will be performed since mode: "{}" is not'
                   'among "uniform", "normal", "fixed"'
                   '.'.format(self.params['glimpse_mode']))
            """
            return batch_images

        # set size of glimpses
        #TODO: change calls to image specs from self.params to self.features
        y_size, x_size = self.params['IMAGE_HEIGHT'], self.params['IMAGE_WIDTH']
        crop_y_size, crop_x_size = self.params['CROP_HEIGHT'], self.params['CROP_WIDTH']
        size = tf.constant(value=[crop_y_size, crop_x_size],
                           dtype=tf.int32)

        if self.params['glimpse_mode'] == 'uniform':
            # generate uniform random window centers for the batch with overlap with input
            y_low, y_high = int(crop_y_size / 2), int(y_size - crop_y_size // 2)
            x_low, x_high = int(crop_x_size / 2), int(x_size - crop_x_size // 2)
            cen_y = tf.random_uniform([self.params['batch_size']], minval=y_low, maxval=y_high)
            cen_x = tf.random_uniform([self.params['batch_size']], minval=x_low, maxval=x_high)
            offsets = tf.stack([cen_y, cen_x], axis=1)

        elif self.params['glimpse_mode'] == 'normal':
            # generate normal random window centers for the batch with overlap with input
            cen_y = tf.random_normal([self.params['batch_size']], mean=y_size // 2, stddev=self.params['glimpse_normal_off_stdev'])
            cen_x = tf.random_normal([self.params['batch_size']], mean=x_size // 2, stddev=self.params['glimpse_normal_off_stdev'])
            offsets = tf.stack([cen_y, cen_x], axis=1)

        elif self.params['glimpse_mode'] == 'fixed':
            # fixed crop
            cen_y = np.ones((self.params['batch_size'],), dtype=np.int32) * self.params['glimpse_height_off']
            cen_x = np.ones((self.params['batch_size'],), dtype=np.int32) * self.params['glimpse_width_off']
            offsets = np.vstack([cen_y, cen_x]).T
            offsets = tf.constant(value=offsets, dtype=tf.float32)

        else:
            # should not come here:
            return batch_images

        # extract glimpses
        glimpse_batch = tf.image.extract_glimpse(batch_images, size, offsets, centered=False, normalized=False,
                                                 uniform_noise=False, name='batch_glimpses')
        return glimpse_batch

    @staticmethod
    def image_scaling(image_batch):
        image_batch -= tf.reduce_min(image_batch, axis=[2,3], keepdims=True)
        image_batch = tf.sqrt(image_batch)
        #image_batch = tf.log1p(image_batch)
        return image_batch

    @staticmethod
    def image_minmax_scaling(image_batch, scale=[0,1]):
        """
        :param label: tensor
        :param min_vals: list, minimum value for each label dimension
        :param max_vals: list, maximum value for each label dimension
        :param range: list, range of label, default [0,1]
        :return:
        scaled label tensor
        """
        min_val = tf.reduce_min(image_batch, keepdims=True) 
        max_val = tf.reduce_max(image_batch, keepdims=True) 
        scaled = (image_batch - min_val)/( max_val - min_val)
        scaled = scaled * (scale[-1] - scale[0]) + scale[0]
        return scaled


class ResNet(ConvNet):

    # def __init__(self, *args, **kwargs):
    #     super(ResNet, self).__init__(*args, **kwargs)

    def _add_branches(self, hidden, out, verbose=True):
        """
        Adds two 4D tensors ensuring that the number of channels is consistent

        :param hidden: 4D tensor, one branch of inputs in the final step of a ResNet (more number of channels)
        :param out: 4D tensor, another branch of inputs in the final step of a ResNet (fewer number of channels)
        :param verbose: bool, (Optional) - whether or not to print statements.
        :return: 4D tensor, output of the addition
        """
        if out.get_shape().as_list()[1] != hidden.get_shape().as_list()[1]:
            # Need to do a 1x1 conv layer on the input to increase the number of channels:
            shortcut_parms = {"kernel": [1, 1], "stride": [1, 1], "padding": "SAME",
                              "features": hidden.get_shape().as_list()[1], "batch_norm": True}
            if verbose:
                self.print_verbose('Doing 1x1 conv on output to bring channels from %d to %d' % (out.get_shape().as_list()[1],
                                                                                    hidden.get_shape().as_list()[1]))
            with tf.variable_scope("shortcut", reuse=self.reuse) as scope:
                out, _ = self._conv(input=out, params=shortcut_parms)
        # ops just for the addition operation - ignore the batch size
        # this_ops = np.prod(out.get_shape().as_list()[:1])
        # if verbose:
        #     self.print_verbose('\tops for adding shortcut: %3.2e' % this_ops)
        # self.ops += this_ops
        # Now add the hidden with input
        return tf.add(out, hidden)

    def _compound_layer(self, out, layer_params, scope_name):
        if layer_params['type'] == 'residual':
            return self._residual_block(out, layer_params)

    def _residual_block(self, out, res_block_params, verbose=True):
        """
        Unit residual block consisting of arbitrary number of convolutional layers, each specified by its own
        OrderedDictionary in the parameters.
        Implementation here based on: https://arxiv.org/pdf/1603.05027.pdf
        Input >> BN >> Relu >> weight >> BN >> ReLU >> Weight >> Add Input

        :param out: 4D tensor, Input to the residual block
        :param res_block_params: OrderedDictionary, Parameters for the residual block
        :param verbose: bool, (Optional) - whether or not to print statements.
        :return: 4D tensor, output of the residual block
        """
        ops_in = self.ops

        with tf.variable_scope("pre_conv1", reuse=self.reuse) as sub_scope:
            hidden = self._batch_norm(input=out)
            hidden = self._activate(input=hidden, name=sub_scope.name)

        # First find the names of all conv layers inside
        layer_ids = []
        for parm_name in res_block_params.keys():
            if isinstance(res_block_params[parm_name], OrderedDict):
                if res_block_params[parm_name]['type'] == 'conv_2D':
                    layer_ids.append(parm_name)
        """
        if verbose:
            print('internal layers:' + str(layer_ids))
            print('Working on the first N-1 layers')
        """
        # Up to N-1th layer: weight >> BN >> ReLU
        for layer_name in layer_ids[:-1]:
            if verbose:
                self.print_verbose('weight >> BN >> ReLU on layer: ' + layer_name)
            with tf.variable_scope(layer_name, reuse=self.reuse) as sub_scope:
                layer_params = res_block_params[layer_name]
                hidden, _ = self._conv(input=hidden, params=layer_params)
                hidden = self._batch_norm(input=hidden)
                hidden = self._activate(input=hidden, name=sub_scope.name, params=layer_params)

        if verbose:
            self.print_verbose('weight ONLY on layer: ' + layer_ids[-1])
        # last layer: Weight ONLY
        with tf.variable_scope(layer_ids[-1], reuse=self.reuse) as sub_scope:
            hidden, _ = self._conv(input=hidden, params=res_block_params[layer_ids[-1]])

        # Now add the two branches
        ret_val = self._add_branches(hidden, out)

        if verbose:
            ops_out = self.ops
            self.print_verbose('\tresnet ops = %3.2e' % (ops_out - ops_in))

        return ret_val

    def _print_layer_specs(self, params, scope, input_shape, output_shape):
        if params['type'] == 'pooling':
            self.print_verbose('%s --- output: %s, kernel: %s, stride: %s' %
                  (scope.name, format(output_shape), format(params['kernel']),
                   format(params['stride'])))
        elif params['type'] == 'residual':
            mem_in_MB = np.cumprod(output_shape)[-1] * self.bytesize / 1024**2
            self.print_verbose('Residual Layer: ' + scope.name)
            for parm_name in params.keys():
                if isinstance(params[parm_name], OrderedDict):
                    if 'conv' in params[parm_name]['type'] :
                        conv_parms = params[parm_name]
                        self.print_verbose('\t%s --- output: %s, kernel: %s, stride: %s, # of weights: %s,  memory: %s MB' %
                              (parm_name, format(output_shape), format(conv_parms['kernel']),
                               format(conv_parms['stride']), format(self.num_weights), format(0)))


        else:
            super(ResNet, self)._print_layer_specs(params, scope, input_shape, output_shape)


class FCDenseNet(ConvNet):
    """
    Fully Convolutional Dense Neural Network
    """

    def build_model(self, summaries=False):
        """
        Here we build the model.
        :param summaries: bool, flag to print out summaries.
        """
        # Initiate 1st layer
        self.print_rank('Building Neural Net ...')
        self.print_rank('input: ---, dim: %s memory: %s MB' %(format(self.images.get_shape().as_list()), format(self.mem/1024)))
        layer_name, layer_params = list(self.network.items())[0]

        with tf.variable_scope(layer_name, reuse=self.reuse) as scope:
            if layer_params['type'] == 'coord_conv':
                out, kernel = self._coord_conv(input=self.images, params=layer_params)
            else:
                out, kernel = self._conv(input=self.images, params=layer_params)
            do_bn = layer_params.get('batch_norm', False)
            if do_bn:
                out = self._batch_norm(input=out)
            else:
                out = self._add_bias(input=out, params=layer_params)
            out = self._activate(input=out, name=scope.name, params=layer_params)
            in_shape = self.images.get_shape().as_list()
            # Tensorboard Summaries
            if self.summary:
                self._activation_summary(out)
                self._activation_image_summary(out)
                self._kernel_image_summary(kernel)

            self._print_layer_specs(layer_params, scope, in_shape, out.get_shape().as_list())
            self.scopes.append(scope)

        # Initiate the remaining layers
        skip_connection_list = list()
        #block_upsample_list = list()
        skip_hub = -1
        for layer_name, layer_params in list(self.network.items())[1:]:
            with tf.variable_scope(layer_name, reuse=self.reuse) as scope:
                in_shape = out.get_shape().as_list()
                if layer_params['type'] == 'conv_2D':
                    self.print_verbose(">>> Adding Conv Layer: %s" % layer_name)
                    self.print_verbose('    input: %s' %format(out.get_shape().as_list()))
                    out, _ = self._conv(input=out, params=layer_params)
                    self.print_verbose('    output: %s' %format(out.get_shape().as_list()))
                    if self.summary: 
                        self._activation_summary(out)
                        self._activation_image_summary(out)

                if layer_params['type'] == 'coord_conv':
                    self.print_verbose(">>> Adding Coord Conv Layer: %s" % layer_name)
                    self.print_verbose('    input: %s' %format(out.get_shape().as_list()))
                    out, _ = self._coord_conv(input=out, params=layer_params)
                    self.print_verbose('    output: %s' %format(out.get_shape().as_list()))
                    if self.summary: self._activation_summary(out)

                if layer_params['type'] == 'pooling':
                    self.print_verbose(">>> Adding Pooling Layer: %s" % layer_name)
                    self.print_verbose('    input: %s' %format(out.get_shape().as_list()))
                    out = self._pool(input=out, name=scope.name, params=layer_params)
                    self.print_verbose('    output: %s' %format(out.get_shape().as_list()))

                if layer_params['type'] == 'linear_output':
                    in_shape = out.get_shape().as_list()
                    # sometimes the same network json file is used for regression and classification.
                    # Taking the number of classes from the parameters / flags instead of the network json
                    if layer_params['bias'] != self.params['NUM_CLASSES']:
                        self.print_verbose("Overriding the size of the bias ({}) and weights ({}) with the 'NUM_CLASSES' parm ({})"
                              "".format(layer_params['bias'], layer_params['weights'], self.params['NUM_CLASSES']))
                        layer_params['bias'] = self.params['NUM_CLASSES']
                        layer_params['weights'] = self.params['NUM_CLASSES']
                    out = self._linear(input=out, name=scope.name, params=layer_params)
                    assert out.get_shape().as_list()[-1] == self.params['NUM_CLASSES'], 'Dimensions of the linear output layer' + \
                                                                         'do not match the expected output set in the params'
                    if self.summary: self._activation_summary(out)

                if layer_params['type'] == 'dense_block_down':
                    self.print_verbose(">>> Adding Dense Block Down: %s" % layer_name)
                    self.print_verbose('    input: %s' %format(out.get_shape().as_list()))
                    out, _ = self._dense_block(out, layer_params, scope)
                    out_freq = self._freq2space(inputs=out) 
                    skip_connection_list.append(out_freq)
                    skip_hub += 1
                    self.print_verbose('    output: %s' %format(out.get_shape().as_list()))
                    if self.summary: 
                        self._activation_summary(out_freq)
                        self._activation_image_summary(out_freq)

                if layer_params['type'] == 'dense_block_bottleneck':
                    self.print_verbose(">>> Adding Dense Block Bottleneck: %s" % layer_name)
                    self.print_verbose('    input: %s' %format(out.get_shape().as_list()))
                    out, block_features = self._dense_block(out, layer_params, scope)
                    # out = self._freq2space(inputs=out)
                    block_features = self._freq2space(inputs=block_features)
                    self.print_verbose('    output: %s' %format(block_features.get_shape().as_list()))
                    if self.summary: 
                        self._activation_summary(block_features)
                        self._activation_image_summary(block_features)

                if layer_params['type'] == 'dense_block_up':
                    self.print_verbose(">>> Adding Dense Block Up: %s" % layer_name)
                    self.print_verbose('    input: %s' %format(out.get_shape().as_list()))
                    out, block_features = self._dense_block(out, layer_params, scope)
                    # block_features = self._freq2space(inputs=block_features)
                    self.print_verbose('    output: %s' %format(out.get_shape().as_list()))
                    if self.summary: 
                        self._activation_summary(out)
                        self._activation_image_summary(out)

                if layer_params['type'] == 'transition_up':
                    self.print_verbose(">>> Adding Transition Up: %s" % layer_name)
                    self.print_verbose('    input: %s' %format(out.get_shape().as_list()))
                    out = self._transition_up(block_features, skip_connection_list[skip_hub], layer_params, scope)
                    self.print_verbose('    output: %s' %format(out.get_shape().as_list()))
                    skip_hub -= 1
                    if self.summary: 
                        self._activation_summary(out)
                        self._activation_image_summary(out)

                if layer_params['type'] == 'transition_down':
                    self.print_verbose(">>> Adding Transition Down: %s" % layer_name)
                    self.print_verbose('    input: %s' %format(out.get_shape().as_list()))
                    out = self._transition_down(out, layer_params, scope)
                    self.print_verbose('    output: %s' %format(out.get_shape().as_list()))
                    if self.summary: 
                        self._activation_summary(out)
                        self._activation_image_summary(out)

                # print layer specs and generate Tensorboard summaries
                if out is None:
                    raise NotImplementedError('Layer type: ' + layer_params['type'] + 'is not implemented!')
                out_shape = out.get_shape().as_list()
                self._print_layer_specs(layer_params, scope, in_shape, out_shape)
            self.scopes.append(scope)
        self.print_rank('Total # of blocks: %d,  weights: %2.1e, memory: %s MB, ops: %3.2e \n' % (len(self.network),
                                                                                        self.num_weights,
                                                                                        format(self.mem / 1024),
                                                                                        self.get_ops()))
        conv_1by1 = OrderedDict({'type': "conv_2D", 'stride': [1, 1], 'kernel': [3,3], 'features': 1, 
                                'padding': 'SAME', 'activation': None}) 
        for i in range(1):
            with tf.variable_scope('FINAL_conv_1by1_%d' %i, reuse=self.reuse) as _ :
                if i == 2: conv_1by1['features'] = 1
                out, _ = self._conv(input=out, params=conv_1by1)
                # out = self._activate(input=out, params=conv_1by1)
        if self.summary: 
            self._activation_image_summary(out)
        # self.print_rank("final shape", out.shape)
        self.model_output = tf.saturate_cast(out, tf.float32)
    
    def _freq2space(self, inputs=None):
        shape = inputs.shape
        weights_dim = 512
        num_fc = 2
        # if weights_dim < 4096 :
        fully_connected = OrderedDict({'type': 'fully_connected','weights': weights_dim,'bias': weights_dim, 'activation': 'relu',
                                'regularize': True})
        deconv = OrderedDict({'type': "deconv_2D", 'stride': [2, 2], 'kernel': [3,3], 'features': 8, 'padding': 'SAME', 'upsample': 2})
        conv = OrderedDict({'type': "conv_2D", 'stride': [1, 1], 'kernel': [3,3], 'features': inputs.shape.as_list()[1],
        'padding': 'SAME', 'activation': 'relu', 'dropout':0.5})
        conv_1by1 = OrderedDict({'type': "conv_2D", 'stride': [1, 1], 'kernel': [3,3], 'features': 1, 'padding': 'SAME', 
        'activation': 'relu', 'dropout':0.5})
        with tf.variable_scope('Freq2Space', reuse=tf.AUTO_REUSE) as _ :
            with tf.variable_scope('conv_1by1', reuse=tf.AUTO_REUSE) as _ :
                conv_params = deepcopy(conv_1by1)
                stride = shape.as_list()[-1] // (int(math.sqrt(weights_dim)))
                conv_params['stride'] = [stride, stride]
                out, _ = self._conv(input=inputs, params=conv_params)
                out = tf.reshape(out, [shape[0], -1])
            for i in range(num_fc):
                if i > 0:
                    weights_dim = min(4096, int(shape.as_list()[-2]**2))
                    fully_connected['weights'] = weights_dim
                    fully_connected['bias'] = weights_dim
                with tf.variable_scope('FC_%d' %i, reuse=self.reuse) as _ :
                    out = self._linear(input=out, params=fully_connected)
                    out = self._activate(input=out, params=fully_connected)
            new_dim = int(math.sqrt(weights_dim))
            out = tf.reshape(out, [shape[0], 1, new_dim, new_dim])
            # print('shape after fc', out.shape)
            num_upsamp = int(np.log2(shape.as_list()[-1] / out.shape.as_list()[-1]))
            conv_1by1_n = deepcopy(conv_1by1)
            conv_1by1_n['stride'] = [1,1]
            conv_1by1_n['features'] = 16
            # self.print_rank("num_upsamp=", num_upsamp)
            if num_upsamp >= 0:
                for up in range(num_upsamp):
                    with tf.variable_scope('deconv_upscale_%d' % up, reuse=self.reuse) as _:
                        out, _ = self._deconv(input=out, params=deconv)
                    with tf.variable_scope('conv_upscale_%d' % up, reuse=self.reuse) as _:
                        out, _ = self._conv(input=out, params=conv_1by1_n)
                        out = self._batch_norm(input=out)
                        out = self._activate(input=out, params=conv_1by1_n)
                        rate = conv_1by1_n.get('dropout', 0) 
                        out = tf.keras.layers.SpatialDropout2D(rate, data_format='channels_first')(inputs=out, training= self.operation == 'train')
                    # self.print_rank(" out shape after upscale+conv", out.shape)
            else:
                out = tf.transpose(out, perm=[0,2,3,1])
                out = tf.image.resize_images(out, [shape[2], shape[3]])
                out = tf.cast(tf.transpose(out, perm=[0,3,1,2]), tf.float16)

            with tf.variable_scope('conv_restore', reuse=tf.AUTO_REUSE) as _ :
                out, _ = self._conv(input=out, params=conv)
                out = self._batch_norm(input=out)
                out = self._activate(input=out, params=conv)
                rate = conv.get('dropout', 0) 
                out = tf.keras.layers.SpatialDropout2D(rate, data_format='channels_first')(inputs=out, training= self.operation == 'train')
        self.print_rank("freq2space: input shape %s, output shape %s" %(format(shape), format(out.shape)))
        return out

    def _upscale(self, inputs=None, params=None, scale=2):
        # conv_params = OrderedDict({'type': 'conv_2D', 'stride': [1, 1], 'kernel': [1, 1], 
        #                         'features': inputs.shape.as_list()[1],
        #                         'activation': 'relu', 
        #                         'padding': 'VALID', 
        #                         'batch_norm': False, 'dropout':0.0})
        with tf.variable_scope('upscale', reuse=self.reuse) as _:
            shape = inputs.shape
            out = tf.reshape(inputs, [-1, shape[1], shape[2], 1, shape[3], 1])
            out = tf.tile(out, [1, 1, 1, scale, 1, scale])
            out = tf.reshape(out, [-1, shape[1], shape[2] * scale, shape[3] * scale])
            # out, _ = self._conv(input=out, params=conv_params)
            # out = self._batch_norm(out)
            return out

    # def _batch_norm(self, input=None):
    #     # if self.operation == 'train':
    #     #     training = True
    #     # else:
    #     #     training = tf.constant(False, dtype=tf.bool)
    #     # out = tf.keras.layers.BatchNormalization(axis=1)(inputs=input, training=training)
    #     # return out
    #     return input
 
    def get_loss(self):
        with tf.variable_scope(self.scope, reuse=self.reuse) as scope:
            self._calculate_loss_regressor()

    def skip_strength(self):
        coeff = self._cpu_variable_init('skip_strength', regularize=False, shape=[], initializer=tf.zeros_initializer())
        strength = tf.math.erf(tf.nn.relu(coeff))
        return strength 

    def _transition_up(self, input, block_connect, layer_params, scope):
        """
        Transition up block : transposed deconvolution.
        Also add skip connection from skip hub to current output
        """
        # strength = self.skip_strength()
        # shape = input.shape
        # out = tf.transpose(input, perm=[0,2,3,1])
        # out = tf.image.resize_images(out, [shape[2]*2, shape[3]*2])
        # out = tf.cast(tf.transpose(out, perm=[0,3,1,2]), tf.float16)
        out, _ = self._deconv(input, layer_params['deconv'], scope)
        if block_connect is not None:
            out = tf.concat([out,block_connect], axis=1)
        return out

    def _transition_down(self, input, layer_params, scope):
        """
        TransitionDown Unit for FCDenseNet
        BN >> ReLU >> 1x1 Convolution >> Dropout >> Pooling
        """
        # BN >> RELU >> 1x1 conv >> Dropout
        conv_layer_params = layer_params['conv']
        if layer_params.get('batch_norm', False):
            out = self._batch_norm(input=input)
        else:
            out = input
        out = self._activate(input=out, params=conv_layer_params)
        if conv_layer_params['type'] == 'coord_conv':
            out, _ = self._coord_conv(input=out, params=conv_layer_params)
        else:
            out, _ = self._conv(input=out, params=conv_layer_params)
        rate = layer_params.get('dropout', 0)
        out = tf.keras.layers.SpatialDropout2D(rate, data_format='channels_first')(inputs=out, training= self.operation == 'train')
        # Pooling
        pool_layer_params = layer_params['pool']
        out = self._pool(input=out, params=pool_layer_params)
        in_shape = input.get_shape().as_list()
        out_shape = out.get_shape().as_list()
        self._print_layer_specs(pool_layer_params, scope, in_shape, out_shape)
        return out

    def _db_layer(self, input, layer_params, scope):
        """
        Dense Block unit for DenseNets
        BN >> Nonlinear Activation >> Convolution >> Dropout
        """
        #out = self._batch_norm(input=input)
        if layer_params.get('batch_norm', False):
            out = self._batch_norm(input=input)
        else:
            out = input
        out = self._activate(input=out, params=layer_params)
        if layer_params['type'] == 'coord_conv':
           out, _ = self._coord_conv(input=out, params=layer_params)
        else:
            out, _ = self._conv(input=out, params=layer_params)
        rate = layer_params.get('dropout', 0) 
        out = tf.keras.layers.SpatialDropout2D(rate, data_format='channels_first')(inputs=out, training= self.operation == 'train')
        in_shape = input.get_shape().as_list()
        out_shape = out.get_shape().as_list()
        self._print_layer_specs(layer_params, scope, in_shape, out_shape)
        return out

    def _dense_block(self, input, layer_params, scope):
        """
        Returns output, block_features (feature maps created)
        """
        # First find the names of all conv layers inside
        layer_params = layer_params['conv']
        layer_ids = []
        for parm_name in layer_params.keys():
            if isinstance(layer_params[parm_name], OrderedDict):
                if 'conv' in layer_params[parm_name]['type']: 
                    layer_ids.append(parm_name)
        # Build layer by layer
        block_features = []
        for layer_name in layer_ids:
            with tf.variable_scope(layer_name, reuse = self.reuse) as scope:
                # build layer
                layer = self._db_layer(input, layer_params[layer_name], scope)
                # layer = self._freq2space(inputs=layer)
                # append to list of features
                block_features.append(layer)
                #stack new layer
                input = tf.concat([input, layer], axis=1)
        block_features = tf.concat(block_features, axis=1)
        return input, block_features


class FCNet(ConvNet):
    """
    Fully (vanilla) convolutional neural net
    """
    def build_model(self, summaries=False):
        """
        Here we build the model.
        :param summaries: bool, flag to print out summaries.
        """
        # Initiate 1st layer
        self.print_rank('Building Neural Net ...')
        self.print_rank('input: ---, dim: %s memory: %s MB' %(format(self.images.get_shape().as_list()), format(self.mem/1024)))
        for layer_num, (layer_name, layer_params) in enumerate(list(self.network.items())):
            if layer_num == 0:
                out = self.images
            else:
                out = out
            with tf.variable_scope(layer_name, reuse=self.reuse) as scope:
                in_shape = out.get_shape().as_list()
                if layer_params['type'] == 'conv_2D':
                    self.print_verbose(">>> Adding Conv Layer: %s" % layer_name)
                    self.print_verbose('    input: %s' %format(out.get_shape().as_list()))
                    out, _ = self._conv(input=out, params=layer_params)
                    do_bn = layer_params.get('batch_norm', False)
                    if do_bn:
                        out = self._batch_norm(input=out)
                    else:
                        out = self._add_bias(input=out, params=layer_params)
                        # dropout
                    if self.operation == 'train' and layer_params.get('dropout', None) is not None:
                        rate = layer_params['dropout']
                    else:
                        rate = 0
                    out = tf.keras.layers.SpatialDropout2D(rate, data_format='channels_first')(inputs=out, training= self.operation == 'train')
                    # out = tf.nn.dropout(out, rate=tf.constant(rate, dtype=out.dtype))
                    out_shape = out.get_shape().as_list()
                    out = self._activate(input=out, name=scope.name, params=layer_params)
                    self.print_verbose('    output: %s' %format(out.get_shape().as_list()))
                    if self.summary: 
                        self._activation_summary(out)
                        self._activation_image_summary(out)

                if layer_params['type'] == 'depth_conv':
                    self.print_verbose(">>> Adding depthwise Conv Layer: %s" % layer_name)
                    self.print_verbose('    input: %s' %format(out.get_shape().as_list()))
                    out, _ = self._depth_conv(input=out, params=layer_params)
                    do_bn = layer_params.get('batch_norm', False)
                    if do_bn:
                        out = self._batch_norm(input=out)
                    else:
                        out = self._add_bias(input=out, params=layer_params)
                    out = self._activate(input=out, name=scope.name, params=layer_params)
                    self.print_verbose('    output: %s' %format(out.get_shape().as_list()))
                    if self.summary: 
                        self._activation_summary(out)
                        self._activation_image_summary(out)

                if layer_params['type'] == 'coord_conv':
                    self.print_verbose(">>> Adding Coord Conv Layer: %s" % layer_name)
                    self.print_verbose('    input: %s' %format(out.get_shape().as_list()))
                    out, _ = self._coord_conv(input=out, params=layer_params)
                    do_bn = layer_params.get('batch_norm', False)
                    if do_bn:
                        out = self._batch_norm(input=out)
                    else:
                        out = self._add_bias(input=out, params=layer_params)
                    out = self._activate(input=out, name=scope.name, params=layer_params)
                    self.print_verbose('    output: %s' %format(out.get_shape().as_list()))
                    if self.summary: self._activation_summary(out)

                if layer_params['type'] == 'pooling':
                    self.print_verbose(">>> Adding Pooling Layer: %s" % layer_name)
                    self.print_verbose('    input: %s' %format(out.get_shape().as_list()))
                    out = self._pool(input=out, name=scope.name, params=layer_params)
                    self.print_verbose('    output: %s' %format(out.get_shape().as_list()))

                if layer_params['type'] == 'deconv_2D':
                    self.print_verbose(">>> Adding de-Conv Layer: %s" % layer_name)
                    self.print_verbose('    input: %s' %format(out.get_shape().as_list()))
                    out, _ = self._deconv(input=out, params=layer_params)
                    # new_shape = out.shape.as_list()[-2:]
                    # out = tf.transpose(out, perm=[0, 2, 3, 1])
                    # # out = tf.cast(out, tf.float32)
                    # out = tf.image.resize(out, [new_shape[0] * 2, new_shape[1] * 2], 
                    #             method=tf.image.ResizeMethod.BILINEAR, align_corners=True)
                    # if self.params['IMAGE_FP16']:
                    #     out = tf.cast(out, tf.float16)
                    # out = tf.transpose(out, perm=[0, 3, 1, 2])
                    self.print_verbose('    output: %s' %format(out.get_shape().as_list()))

                    if self.summary: 
                        self._activation_summary(out) 
                        self._activation_image_summary(out)

                if layer_params['type'] == 'dense_layers_block':
                    self.print_verbose(">>> Adding Dense Layers Block: %s" % layer_name)
                    self.print_verbose('    input: %s' %format(out.get_shape().as_list()))
                    if layer_params['conv_type'] == 'conv_2D':
                        # out, _ = self._dense_layers_block_multi_head_loop(input=out, params=layer_params)
                        out, _ = self._freq_to_space(input=out, params=layer_params)
                        self.print_verbose(">>> Using Single head" )
                    else:
                        out, _ = self._dense_layers_block_multi_head_loop(input=out, params=layer_params)
                        # out, _ = self._dense_layers_block_multi_head(input=out, params=layer_params)
                        self.print_verbose(">>> Using Multi head" )
                    self.print_verbose('    output: %s' %format(out.get_shape().as_list()))
                    if self.summary: self._activation_summary(out) 

                if layer_params['type'] == 'freq2space':
                    self.print_verbose(">>> Adding freq2space layer: %s" % layer_name)
                    self.print_verbose('    input: %s' %format(out.get_shape().as_list())) 
                    out, _ = self._freq_to_space(input=out, params=layer_params) 
                    self.print_verbose('    output: %s' %format(out.get_shape().as_list()))
                    if self.summary: 
                        self._activation_summary(out)
                        self._activation_image_summary(out)

                if layer_params['type'] == 'freq2space_attention':
                    self.print_verbose(">>> Adding freq2space layer: %s" % layer_name)
                    self.print_verbose('    input: %s' %format(out.get_shape().as_list())) 
                    out, _ = self._freq_to_space_attention(input=out, params=layer_params) 
                    self.print_verbose('    output: %s' %format(out.get_shape().as_list()))
                    if self.summary: 
                        self._activation_summary(out)
                        self._activation_image_summary(out)

                if layer_params['type'] == 'freq2space_CVAE':
                    self.print_verbose(">>> Adding freq2space layer: %s" % layer_name)
                    self.print_verbose('    input: %s' %format(out.get_shape().as_list())) 
                    out, _ = self._freq_to_space_attention(input=out, params=layer_params, attention=False) 
                    self.print_verbose('    output: %s' %format(out.get_shape().as_list()))
                    if self.summary: 
                        self._activation_summary(out)
                        self._activation_image_summary(out)

                # print layer specs and generate Tensorboard summaries
                if out is None:
                    raise NotImplementedError('Layer type: ' + layer_params['type'] + 'is not implemented!')
                
                out_shape = out.get_shape().as_list()
                self._print_layer_specs(layer_params, scope, in_shape, out_shape)
            self.scopes.append(scope)
        # final 1x1 conv
        with tf.variable_scope('CONV_FIN', reuse=self.reuse) as scope:
            conv_1by1 = OrderedDict({'type': 'conv_2D', 'stride': [1, 1], 'kernel': [3, 3], 'padding': 'SAME', 'features': 1})
            self.print_verbose(">>> Adding CONV_FIN layer: ")
            self.print_verbose('    input: %s' %format(out.get_shape().as_list()))
            out, _ = self._conv(input=out, params=conv_1by1) 
            self.print_verbose('    output: %s' %format(out.get_shape().as_list()))
            out_shape = out.get_shape().as_list()
            self._print_layer_specs(layer_params, scope, in_shape, out_shape)
            self.scopes.append(scope) 
            if self.summary: 
                self._activation_summary(out)
                self._activation_image_summary(out)

        if self.labels.shape != out.shape:
            out = tf.transpose(out, perm = [0, 2, 3, 1])
            out = tf.image.resize(out, self.labels.shape[-2:], method=tf.image.ResizeMethod.BILINEAR)
            out = tf.transpose(out, perm=[0, 3, 1, 2])
        self.print_rank('Total # of blocks: %d,  weights: %2.1e, memory: %s MB, ops: %3.2e \n' % (len(self.network),
                                                                                        self.num_weights,
                                                                                        format(self.mem / 1024),
                                                                                        self.get_ops()))
        self.model_output = tf.saturate_cast(out, tf.float32)
 
    def _dense_layers_block_multi_head(self, input=None, params=None):
        conv_1by1_base = OrderedDict({'type': 'conv_2D', 'stride': [1, 1], 'kernel': [1, 1], 
                                'features': 1,
                                'activation': 'relu', 'padding': 'VALID', 'batch_norm': False, 'dropout': None})
        fully_connected = OrderedDict({'type': 'fully_connected','weights': None,'bias': None, 'activation': 'relu',
                                   'regularize': True})
        num_slices = self.images.get_shape().as_list()[1]
        if params['conv_type'] == "conv_2D":
            with tf.variable_scope('restore_channels', reuse=self.reuse) as scope:
                conv_1by1 = deepcopy(conv_1by1_base)
                conv_1by1['features'] = num_slices 
                input, _ = self._conv(input=input, params=conv_1by1) 
        new_shape = [input.shape.as_list()[0], num_slices, -1, 
                     input.shape.as_list()[-2], input.shape.as_list()[-1]] 
        tensor_slices = tf.reshape(input, new_shape)
        
        tensor_slices = tf.transpose(tensor_slices, perm=[1, 0, 2, 3, 4])

        num_pattern = '[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?'
        re_vals = re.compile(num_pattern, re.VERBOSE)
        #TODO: This generates a shared conv1by1 for all slices!!
        def freq_to_space(tens):
            # name = re_vals.findall(tens.name)[-1]
            # tf.get_default_graph().get_tensor_by_name
            with tf.variable_scope('freq_to_space', reuse=self.reuse) as scope:
                conv_1by1 = deepcopy(conv_1by1_base)
                out, _ = self._conv(input=tens, params=conv_1by1) 
                do_bn = conv_1by1.get('batch_norm', False)
                if do_bn:
                    out = self._batch_norm(input=out)
                else:
                    out = self._add_bias(input=out, params=conv_1by1)
                out = self._activate(input=out, name=scope.name, params=conv_1by1)
                out = tf.reshape(out, [self.params['batch_size'], -1])
                fully_connected['weights']= out.shape.as_list()[-1] 
                fully_connected['bias'] = out.shape.as_list()[-1] 
                # fully_connected['weights']= 128 
                # fully_connected['bias'] = 128
            if params['n_layers'] == 0:
                out = tf.reshape(out, [self.params['batch_size'], -1])
            for n_layer in range(params['n_layers']):
                with tf.variable_scope('fully_connnected_%d' % n_layer, reuse=self.reuse) as _ :
                    out = self._linear(input=out, params=fully_connected)
                    out = self._activate(input=out, params=fully_connected)
            if self.operation == 'train' and conv_1by1['dropout'] is not None:
                rate = conv_1by1['dropout']
            else:
                rate = 0
            out = tf.nn.dropout(out, rate=tf.constant(rate, dtype=tens.dtype))
            return out

        ops_map_fn = 2 * np.prod(conv_1by1_base['kernel']) * conv_1by1_base['features'] * np.prod(tensor_slices.shape.as_list()[2:])
        self.ops += num_slices * ops_map_fn
        out_slices = tf.map_fn(freq_to_space, tensor_slices, back_prop=True)
        out_slices = tf.transpose(out_slices, perm=[1, 0, 2])
        out_slices = tf.reshape(out_slices, [self.images.shape.as_list()[0], -1])
        # expand dims
        out_slices = tf.expand_dims(tf.expand_dims(out_slices, -1), -1)
        # layout spatially
        out = tf.nn.depth_to_space(out_slices, int(np.sqrt(self.images.shape.as_list()[1])), data_format=self.params['TENSOR_FORMAT'])
        return out, None

    def _freq_to_space_conv1by1(self, input=None, params=None):
        # input = self._multi_attention_head(input)
        freq_dim = int(np.sqrt(input.shape[1].value))
        slices = tf.reshape(input, [input.shape[0].value, freq_dim, freq_dim, -1])
        slices = tf.transpose(slices, perm=[0, 3, 1, 2])
        conv_1by1 = OrderedDict({'type': 'conv_2D', 'stride': [1, 1], 'kernel': [1, 1], 
                                'features': params['init_features'], 'activation': params['activation'], 
                                'padding': 'VALID', 
                                'batch_norm': params['batch_norm'], 'dropout':params['dropout']})
        fully_connected = OrderedDict({'type': 'fully_connected','weights': None,'bias': None, 
                                        'activation': params['activation'], 'regularize': True})
        # with tf.variable_scope('freq_to_space', reuse=self.reuse) as scope:
        out, _ = self._conv(input=slices, params=conv_1by1) 
        do_bn = conv_1by1.get('batch_norm', False)
        if do_bn:
            out = self._batch_norm(input=out)
        else:
            out = self._add_bias(input=out, params=conv_1by1)
        out = self._activate(input=out, params=conv_1by1)
        if self.operation == 'train' and conv_1by1['dropout'] is not None:
            rate = conv_1by1['dropout']
        else:
            rate = 0
        out = tf.nn.dropout(out, rate=tf.constant(rate, dtype=out.dtype))
        return out, None

    def _freq_to_space(self, input=None, params=None, fc_cond=False):
        # input = self._multi_attention_head(input)
        conv_1by1 = OrderedDict({'type': 'conv_2D', 'stride': [1, 1], 'kernel': [1, 1], 
                                'features': params['init_features'],
                                'activation': params['activation'], 
                                # 'activation': 're',
                                'padding': 'SAME', 
                                'batch_norm': params['batch_norm'], 'dropout':params['dropout']})
        freq_dim =  input.shape[-1].value
        space_dim = int(np.sqrt(input.shape[1].value))
        batch_dim = self.params['batch_size']
        slices = tf.reshape(input, [input.shape[0].value, space_dim, space_dim, -1])
        slices = tf.transpose(slices, perm=[0, 3, 1, 2])
        slices, _ = self._conv(input=slices, params=conv_1by1) 
        do_bn = conv_1by1.get('batch_norm', False)
        if do_bn:
            slices = self._batch_norm(input=slices)
        else:
            slices = self._add_bias(input=slices, params=conv_1by1)
        slices = self._activate(input=slices, params=conv_1by1)
        if fc_cond:
            freq_dim =  slices.shape[1].value
            slices = tf.transpose(slices, perm=[0, 2, 3, 1])
            slices = tf.reshape(slices, [batch_dim, space_dim * space_dim, freq_dim])
            fully_connected = OrderedDict({'type': 'fully_connected','weights': 512,'bias': 512, 
                                            'activation': params['activation'], 'regularize': True}) 
            weights_shape = [batch_dim, freq_dim, fully_connected['weights']]
            bias_shape = [batch_dim, space_dim * space_dim, fully_connected['bias']]
            lin_initializer= tf.random_normal_initializer
            lin_initializer = tf.uniform_unit_scaling_initializer
            lin_initializer.factor = 1.43 
            weights = self._cpu_variable_init('fc_weights', shape=weights_shape, initializer=lin_initializer,
                                            regularize=True)
            bias = self._cpu_variable_init('fc_bias', bias_shape, initializer=tf.zeros_initializer)
            slices = tf.add( tf.matmul(slices, weights), bias)
            conv_1by1['activation'] = 'tanh'
            slices = self._activate(input=slices, params=conv_1by1) 
            out = tf.transpose(slices, perm=[0,2,1])
            out = tf.reshape(slices, [batch_dim, -1, space_dim, space_dim])
            return out, None
        return slices, None

    def _multi_attention_head(self, inputs=None, params=None):
        try:
            from tensor2tensor.layers import common_image_attention as cia
            from tensor2tensor.layers import common_layers
            from tensor2tensor.layers import common_hparams
            self.print_rank('Adding Multi-Attention Head')
        except Exception as e:
            self.print_rank('Tensor2Tensor could not be imported. Skipping attention module.')
            self.print_rank('%s' % format(e))
            return input

        dim_batch = self.params['batch_size']
        dim_q_x = inputs.shape[-1].value
        dim_q_y = inputs.shape[-2].value
        dim_x = int(np.sqrt(inputs.shape[1].value))
        dim_y = dim_x
        hidden_size = params['init_features']
        hidden_size = 256

        hparams= common_hparams.basic_params1()
        hparams.add_hparam("num_heads", 1)
        hparams.add_hparam("pos", "timing")
        hparams.add_hparam("attention_key_channels", dim_q_x * dim_q_y)
        hparams.add_hparam("attention_value_channels", dim_q_x * dim_q_y)
        hparams.add_hparam("query_shape",(1,1))
        hparams.add_hparam("memory_flange", (1,1))
        hparams.set_hparam("hidden_size", hidden_size)
        hparams.set_hparam("max_length", 2**16)
        hparams.set_hparam("dropout", params['dropout'])
        hparams.add_hparam("ffn_layer", "conv_hidden_relu")

        out = tf.transpose(inputs, perm=[0, 2, 3, 1])
        out = tf.image.resize_images(out, [16,16], method=tf.image.ResizeMethod.AREA)
        out = tf.transpose(out, perm=[0, 3, 1, 2])    
        out = tf.reshape(out, [dim_batch, dim_y, dim_x, -1])
        out = tf.cast(out, tf.float16)
        # embedding and position encodings
        # out = self.image_minmax_scaling(out, scale=[0, 2**32])
        # out = tf.cast(out, tf.int32)
        # out = self._embedding_layer(out, shape=[256,1])
        # out = tf.reshape(out, [dim_batch, dim_y, dim_x, -1])
        # out = cia.get_channel_embeddings(dim_q_x * dim_q_y, out, hidden_size)
        # out = cia.add_pos_signals(out, hparams)
        # out = self.add_pos_signals(out, hparams)
        # out = tf.cast(out, tf.float16)

        # # 2d local attention
        with tf.variable_scope('multihead_attention', reuse=self.reuse) as scope:
            out = cia.local_attention_2d(common_layers.layer_preprocess(out, hparams), 
                        hparams, attention_type="local_attention_2d")
        out = tf.transpose(out, perm = [0, 3, 1, 2])
        print('attention output:', out.get_shape().as_list())
        return out

    def _cvae_slices(self, inputs, params=None):
        new_shape = [inputs.shape.as_list()[0], inputs.shape.as_list()[1], -1, 
                     inputs.shape.as_list()[-2], inputs.shape.as_list()[-1]] 
        tensor_slices = tf.reshape(inputs, new_shape)
        tensor_slices = tf.transpose(tensor_slices, perm=[1, 0, 2, 3, 4])
        conv_1by1 = params['conv_params']
        fully_connected = params['fc_params']
        num_conv = params['n_conv_layers']
        num_fc = params['n_fc_layers']
        # conv_1by1 = OrderedDict({'type': 'conv_2D', 'stride': [2, 2], 'kernel': [4, 4], 
        #                         'features': 16,
        #                         'activation': 'relu', 'padding': 'SAME', 'batch_norm': True, 'dropout': 0.0})
        # pool = OrderedDict({'type': 'pooling', 'stride': [2, 2], 'kernel': [2, 2], 'pool_type': 'max','padding':'SAME'})
        # fully_connected = OrderedDict({'type': 'fully_connected','weights': 1024,'bias': 1024, 'activation': 'relu',
        #                            'regularize': True})
        self.print_verbose("\t>>> Adding CVAE: " )
        self.print_verbose('\t\t    input: %s' %format(inputs.get_shape().as_list()))
        # pre_ops = deepcopy(self.ops)
        def CVAE(tens):
            #TODO, turn this into a full denoising VAE
            for i in range(num_conv):
                with tf.variable_scope('CVAE_block_%d' % i , reuse=self.reuse) as scope:
                    tens , _ = self._conv(input=tens, params=conv_1by1)
                    # tens = self._pool(input=tens, params=pool)
                    tens = self._activate(input=tens, params=conv_1by1)
            if tens.shape[-2:] != [32, 32]:
                tens = tf.transpose(tens, perm=[0, 2, 3, 1])
                tens = tf.image.resize(tens, [32, 32], method=tf.image.ResizeMethod.BILINEAR)
                if self.params['IMAGE_FP16']:
                    tens = tf.saturate_cast(tens, tf.float16)
                tens = tf.transpose(tens, perm=[0, 3, 1, 2])
            # self.print_rank('shape inside CVAE', tens.get_shape())
            for i in range(num_fc):
                with tf.variable_scope('CVAE_fc_%d' %i, reuse=self.reuse) as _ :
                    tens = self._linear(input=tens, params=fully_connected)
                    tens = self._activate(input=tens, params=fully_connected)
            # tens = tf.reshape(tens, [new_shape[0], -1])
            return tens

        # post_ops = deepcopy(self.ops)
        # self.print_rank("post pre, cvae ops: ", pre_ops - post_ops)
        out = tf.map_fn(CVAE, tensor_slices, back_prop=True, swap_memory=True, parallel_iterations=256)
        # self.print_rank('output of CVAE', out.get_shape())
        out = tf.transpose(out, perm= [1, 2, 0])
        out = tf.reshape(out, [new_shape[0], -1, int(math.sqrt(new_shape[1])), int(math.sqrt(new_shape[1]))])
        if out.shape[-2:] != [16, 16]:
            new_shape = out.shape.as_list()[-2:]
            out = tf.transpose(out, perm=[0, 2, 3, 1])
            out = tf.image.resize(out, [16, 16], method=tf.image.ResizeMethod.BILINEAR)
            out = tf.transpose(out, perm=[0, 3, 1, 2])
            if self.params['IMAGE_FP16']:
                out = tf.saturate_cast(out, tf.float16)
        self.print_verbose('\t\t    output: %s' %format(out.get_shape().as_list()))
        return out
    
    def _cvae_slices_batched(self, inputs, params=None):
        new_shape = [inputs.shape.as_list()[0], inputs.shape.as_list()[1], -1, 
                     inputs.shape.as_list()[-2], inputs.shape.as_list()[-1]]
        re_shape = [inputs.shape.as_list()[0] * inputs.shape.as_list()[1], 1, 
                     inputs.shape.as_list()[-2], inputs.shape.as_list()[-1]]  
        inputs = tf.reshape(inputs, re_shape)

        conv_1by1 = params['conv_params']
        fc_params = params['fc_params']
        num_conv = params['n_conv_layers']
        num_fc = params['n_fc_layers']
        # conv_1by1 = OrderedDict({'type': 'conv_2D', 'stride': [2, 2], 'kernel': [4, 4], 
        #                         'features': 16,
        #                         'activation': 'relu', 'padding': 'SAME', 'batch_norm': True, 'dropout': 0.0})
        # pool = OrderedDict({'type': 'pooling', 'stride': [2, 2], 'kernel': [2, 2], 'pool_type': 'max','padding':'SAME'})
        # fully_connected = OrderedDict({'type': 'fully_connected','weights': 1024,'bias': 1024, 'activation': 'relu',
        #                            'regularize': True})
        self.print_verbose("\t>>> Adding CVAE: " )
        self.print_verbose('\t\t    input: %s' %format(inputs.get_shape().as_list()))
        # pre_ops = deepcopy(self.ops)
        def CVAE(tens):
            #TODO, turn this into a full denoising VAE
            for i in range(num_conv):
                with tf.variable_scope('CVAE_block_%d' % i , reuse=self.reuse) as scope:
                    tens , _ = self._conv(input=tens, params=conv_1by1)
                    # tens = self._pool(input=tens, params=pool)
                    tens = self._activate(input=tens, params=conv_1by1)
            if tens.shape[-2:] != [32, 32]:
                tens = tf.transpose(tens, perm=[0, 2, 3, 1])
                tens = tf.image.resize(tens, [32, 32], method=tf.image.ResizeMethod.BILINEAR)
                if self.params['IMAGE_FP16']:
                    tens = tf.saturate_cast(tens, tf.float16)
                tens = tf.transpose(tens, perm=[0, 3, 1, 2])
            # self.print_rank('shape inside CVAE', tens.get_shape())
            return tens

        def fc_block(inputs, layer_params, scope_name):
            out = inputs
            for i in range(num_fc):
                with tf.variable_scope('%s_fc_%d' %(scope_name, i), reuse=self.reuse) as _ :
                    lin_initializer = tf.glorot_uniform_initializer 
                    # lin_initializer = tf.uniform_unit_scaling_initializer
                    if isinstance(lin_initializer, tf.uniform_unit_scaling_initializer):
                        if fc_params['type'] == 'fully_connected':
                            if fc_params['activation'] == 'tanh':
                                lin_initializer.factor = 1.15
                            elif fc_params['activation'] == 'relu':
                                lin_initializer.factor = 1.43
                        elif fc_params['type'] == 'linear_output':
                            lin_initializer.factor = 1.0
                    elif isinstance(lin_initializer, tf.random_normal_initializer):
                        init_val = max(np.sqrt(2.0 / fc_params['weights']), 0.01)
                        lin_initializer.mean = 0.0
                        lin_initializer.stddev = init_val
                    weights_shape = [1] + [out.shape.as_list()[-1]] + [fc_params['weights']]
                    bias_shape = [fc_params['bias']]
                    weights = self._cpu_variable_init('weights', shape=weights_shape, initializer=lin_initializer,
                                                    regularize=fc_params['regularize'])
                    bias = self._cpu_variable_init('bias', bias_shape, initializer=tf.zeros_initializer)
                    weights_mat = tf.tile(weights, [self.params['batch_size'], 1, 1])
                    # bias_mat = tf.tile(bias, [self.params['batch_size'], 1])
                    out = tf.matmul(out, weights_mat)
                    out = out + bias
                    out = self._activate(input=out, params=fc_params) 
            return out

        out = CVAE(inputs)
        out = tf.reshape(out, [self.params['batch_size'], new_shape[1], -1])
        # out = tf.transpose(out, perm=[1,0,2])
        self.print_rank('output of CVAE- before fc block ', out.get_shape()) 
        
        out = fc_block(out, None, 'CVAE')
       
        self.print_rank('output of CVAE- after fc block ', out.get_shape()) 
        out = tf.transpose(out, perm=[0, 2, 1])
        out = tf.reshape(out, [new_shape[0], -1, int(math.sqrt(new_shape[1])), int(math.sqrt(new_shape[1]))])
        if out.shape[-2:] != [16, 16]:
            new_shape = out.shape.as_list()[-2:]
            out = tf.transpose(out, perm=[0, 2, 3, 1])
            out = tf.image.resize(out, [16, 16], method=tf.image.ResizeMethod.BILINEAR)
            out = tf.transpose(out, perm=[0, 3, 1, 2])
            if self.params['IMAGE_FP16']:
                out = tf.saturate_cast(out, tf.float16)
        self.print_verbose('\t\t    output: %s' %format(out.get_shape().as_list()))
        return out
        
    def _embedding_layer(self, inputs, shape=[8,256]):
        lin_initializer = self._get_initializer(self.hyper_params.get('initializer', None))
        embed_matrix = self._cpu_variable_init('embedding_matrix', shape=shape, initializer=lin_initializer,
                                          regularize=False)
        out = tf.nn.embedding_lookup(embed_matrix, inputs)
        # out = tf.cast(out, tf.float32)
        return out

    def add_pos_signals(self, x, hparams, name="pos_emb"):
        with tf.variable_scope(name, reuse=False):
            if hparams.pos == "timing":
                x = self.add_timing_signal_nd(x)
            else:
                return x
        return x

    @staticmethod
    def add_timing_signal_nd(x, min_timescale=1.0, max_timescale=1.0e4):
        """Adds a bunch of sinusoids of different frequencies to a Tensor.

        Each channel of the input Tensor is incremented by a sinusoid of a different
        frequency and phase in one of the positional dimensions.

        This allows attention to learn to use absolute and relative positions.
        Timing signals should be added to some precursors of both the query and the
        memory inputs to attention.

        The use of relative position is possible because sin(a+b) and cos(a+b) can be
        experessed in terms of b, sin(a) and cos(a).

        x is a Tensor with n "positional" dimensions, e.g. one dimension for a
        sequence or two dimensions for an image

        We use a geometric sequence of timescales starting with
        min_timescale and ending with max_timescale.  The number of different
        timescales is equal to channels // (n * 2). For each timescale, we
        generate the two sinusoidal signals sin(timestep/timescale) and
        cos(timestep/timescale).  All of these sinusoids are concatenated in
        the channels dimension.

        Args:
            x: a Tensor with shape [batch, d1 ... dn, channels]
            min_timescale: a float
            max_timescale: a float

        Returns:
            a Tensor the same shape as x.
        """
        from tensor2tensor.layers import common_layers
        num_dims = len(x.get_shape().as_list()) - 2
        channels = common_layers.shape_list(x)[-1]
        num_timescales = channels // (num_dims * 2)
        log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) /
            (tf.to_float(num_timescales) - 1))
        inv_timescales = min_timescale * tf.exp(
            tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
        for dim in range(num_dims):
            length = common_layers.shape_list(x)[dim + 1]
            position = tf.to_float(tf.range(length))
            scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(
                inv_timescales, 0)
            signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
            prepad = dim * 2 * num_timescales
            postpad = channels - (dim + 1) * 2 * num_timescales
            signal = tf.pad(signal, [[0, 0], [prepad, postpad]])
            for _ in range(1 + dim):
                signal = tf.expand_dims(signal, 0)
            for _ in range(num_dims - 1 - dim):
                signal = tf.expand_dims(signal, -2)
            x += tf.cast(signal, x.dtype)
        return x

    @staticmethod
    def image_minmax_scaling(image_batch, scale=[0,1]):
        """
        :param label: tensor
        :param min_vals: list, minimum value for each label dimension
        :param max_vals: list, maximum value for each label dimension
        :param range: list, range of label, default [0,1]
        :return:
        scaled label tensor
        """
        min_val = tf.reduce_min(image_batch, keepdims=True) 
        max_val = tf.reduce_max(image_batch, keepdims=True) 
        scaled = (image_batch - min_val)/( max_val - min_val)
        scaled = scaled * (scale[-1] - scale[0]) + scale[0]
        return scaled

    def _freq_to_space_attention(self, input=None, params=None, attention=True):
        conv_1by1 = OrderedDict({'type': 'conv_2D', 'stride': [1, 1], 'kernel': [1, 1], 
                                'features': params['init_features'],
                                'activation': params['activation'], 
                                'padding': 'VALID', 
                                'batch_norm': params['batch_norm'], 'dropout':params['dropout']})
        out = self._cvae_slices(input, params=params['cvae_params'])
        if attention:
            out = self._multi_attention_head(inputs=input, params=params)
        out, _ = self._conv(input=out, params=conv_1by1) 
        do_bn = conv_1by1.get('batch_norm', False)
        if do_bn:
            out = self._batch_norm(input=out)
        else:
            out = self._add_bias(input=out, params=conv_1by1)
        out = self._activate(input=out, params=conv_1by1)
        return out, None

    def _dense_layers_block_multi_head_loop(self, input=None, params=None):
        conv_1by1_base = OrderedDict({'type': 'conv_2D', 'stride': [1, 1], 'kernel': [1, 1], 
                                'features': 1,
                                'activation': 'relu', 'padding': 'VALID', 'batch_norm': True, 'dropout': 0.25})
        fully_connected = OrderedDict({'type': 'fully_connected','weights': None,'bias': None, 'activation': 'relu',
                                   'regularize': True})
        num_slices = self.images.get_shape().as_list()[1]
        # tensor_slices = tf.split(input, num_slices, axis=1)
        new_shape = [input.shape.as_list()[0], num_slices, -1, 
                     input.shape.as_list()[-2], input.shape.as_list()[-1]] 
        tensor_slices = tf.reshape(input, new_shape)
        tensor_slices = tf.transpose(tensor_slices, perm=[1, 0, 2, 3, 4])
        # print(tensor_slices.shape)
        def freq_to_space(args):
            conv_kernel, tens = args[:]
            # conv_kernel = conv_kernel[0]
            print("conv_kernel:", conv_kernel)
            # 1by1 conv
            stride_shape = [1,1]+list(conv_1by1_base['stride'])
            out = tf.nn.conv2d(tens, conv_kernel, stride_shape, data_format='NCHW', padding=conv_1by1_base['padding'])
            out = self._activate(input=out, params=conv_1by1)
            out = tf.reshape(out, [self.params['batch_size'], -1])
            # dense layer
            # output = tf.nn.bias_add(tf.matmul(out, fc_weights), fc_bias)
            # dropout
            if self.operation == 'train' and conv_1by1['dropout'] is not None:
                rate = conv_1by1['dropout']
            else:
                rate = 0
            out = tf.nn.dropout(out, rate=tf.constant(rate, dtype=tens.dtype))
            return vars, out

        # get variables
        vars = []
        for idx in range(num_slices):
            # 1by1 conv to collapse channels
            with tf.variable_scope('freq_to_space_slices/conv_1by1_%d' %idx, reuse=self.reuse) as scope:
                conv_1by1 = deepcopy(conv_1by1_base)
                conv_kernel = self._get_vars_conv(input_shape=tensor_slices.shape.as_list(), params=conv_1by1) 
                vars.append(conv_kernel)
            #     params['n_layers'] = 1
            #     # dense layers to transform
            # with tf.variable_scope('freq_to_space_slices/fully_conn_%d' %idx , reuse=self.reuse) as scope:
            #     fc_weights, fc_bias = self._get_vars_linear(input=input, params=fully_connected)
            # vars.append([conv_kernel, fc_weights, fc_bias])
        # print(len(vars))
        # print(vars[0])
        vars = tf.stack(vars)
        print("vars shape: ", vars.shape)
        _, out_slices = tf.map_fn(freq_to_space, (vars, tensor_slices), back_prop=True, swap_memory=True, parallel_iterations=num_slices)
        out_slices = tf.transpose(out_slices, perm=[1, 0, 2])
        out_slices = tf.reshape(out_slices, [self.images.shape.as_list()[0], -1])

        # concatenate along depth
        # out = tf.concat(tensor_slices, 1)
        out = tf.expand_dims(tf.expand_dims(out_slices, -1), -1)
        # self.print_rank(out.shape.as_list())
        # layout spatially
        out = tf.nn.depth_to_space(out, int(np.sqrt(self.images.shape.as_list()[1])), data_format=self.params['TENSOR_FORMAT'])
        return out, None

    def _dense_layers_block_single_head(self, input=None, params=None):
        conv_1by1_base = OrderedDict({'type': 'conv_2D', 'stride': [1, 1], 'kernel': [1, 1], 
                                'features': 1,
                                'activation': 'relu', 'padding': 'VALID', 'batch_norm': False, 'dropout': 0.5})
        fully_connected = OrderedDict({'type': 'fully_connected','weights': None,'bias': None, 'activation': 'relu',
                                   'regularize': True})
        num_slices = self.images.get_shape().as_list()[1]
        with tf.variable_scope('restore_channels', reuse=self.reuse) as scope:
            conv_1by1 = deepcopy(conv_1by1_base)
            # conv_1by1['features'] = num_slices 
            out, _ = self._conv(input=input, params=conv_1by1) 
        fully_connected['weights']= 1024
        fully_connected['bias'] = 1024
        if params['n_layers'] == 0:
            out = tf.reshape(out, [self.params['batch_size'], -1])
        for n_layer in range(params['n_layers']):
            with tf.variable_scope('fully_connnected_%d' % n_layer, reuse=self.reuse) as _ :
                out = self._linear(input=out, params=fully_connected)
                out = self._activate(input=out, params=fully_connected)
        if self.operation == 'train' and conv_1by1['dropout'] is not None:
            rate = conv_1by1['dropout']
        else:
            rate = 0
        out = tf.nn.dropout(out, rate=tf.constant(rate, dtype=out.dtype))
        # out = tf.transpose(out, perm=[1, 0])
        # expand dims
        out_slices = tf.expand_dims(tf.expand_dims(out, -1), -1)
        # layout spatially
        out = tf.nn.depth_to_space(out_slices, int(np.sqrt(self.images.shape.as_list()[1])), data_format=self.params['TENSOR_FORMAT'])
        return out, None

    def _get_vars_linear(self, input=None, params=None, name=None, verbose=True):
        """
        Linear layer
        :param input:
        :param params:
        :return:
        """
        dim_input = input.shape[2].value * input.shape[3].value
        if params['weights'] is not None:
            weights_shape = [dim_input, params['weights']]
            init_val = max(np.sqrt(2.0/params['weights']), 0.01)
        # self.print_verbose('stddev: %s' % format(init_val))
            bias_shape = [params['bias']]
        else:
            weights_shape = [dim_input, dim_input]
            init_val = max(np.sqrt(2.0/dim_input), 0.01)
        # self.print_verbose('stddev: %s' % format(init_val))
            bias_shape = [dim_input]
            params['weights'] = dim_input
            params['bias'] = dim_input
            
        # Fine tuning the initializer:
        #lin_initializer = self._get_initializer(self.hyper_params.get('initializer', None))
        lin_initializer = tf.glorot_uniform_initializer 
        if isinstance(lin_initializer, tf.uniform_unit_scaling_initializer):
            if params['type'] == 'fully_connected':
                if params['activation'] == 'tanh':
                    lin_initializer.factor = 1.15
                elif params['activation'] == 'relu':
                    lin_initializer.factor = 1.43
            elif params['type'] == 'linear_output':
                lin_initializer.factor = 1.0
        elif isinstance(lin_initializer, tf.random_normal_initializer):
            init_val = max(np.sqrt(2.0 / params['weights']), 0.01)
            if verbose:
                print('stddev: %s' % format(init_val))
            lin_initializer.mean = 0.0
            lin_initializer.stddev = init_val

        weights = self._cpu_variable_init('weights', shape=weights_shape, initializer=lin_initializer,
                                          regularize=params['regularize'])
        bias = self._cpu_variable_init('bias', bias_shape, initializer=tf.zeros_initializer)

        # Keep tabs on the number of weights and memory
        self.num_weights += bias_shape[0] + np.cumprod(weights_shape)[-1]
        # self.mem += np.cumprod(output.get_shape().as_list())[-1] * self.bytesize / 1024
        this_ops = 2 * params['weights'] + params['bias']
        self.ops += this_ops
        return weights, bias

    def _get_vars_conv(self, input_shape=None, params=None):
        """
        Builds 2-D convolutional layer
        :param input: as it says
        :param params: dict, must specify kernel shape, stride, and # of features.
        :return: output of convolutional layer and filters
        """
        stride_shape = [1,1]+list(params['stride'])
        features = params['features']
        kernel_shape = [1,1] + [input_shape[2], features]


        # Fine tunining the initializer:
        conv_initializer = self._get_initializer(self.hyper_params.get('initializer', None))
        if isinstance(conv_initializer, tf.truncated_normal_initializer):
            init_val = np.sqrt(2.0/(kernel_shape[0] * kernel_shape[1] * features))
            conv_initializer.stddev = init_val
        elif isinstance(conv_initializer, tf.uniform_unit_scaling_initializer):
            conv_initializer.factor = 1.43
        elif isinstance(conv_initializer, tf.random_normal_initializer):
            init_val = np.sqrt(2.0/(kernel_shape[0] * kernel_shape[1] * features))

            #self.print_verbose('stddev: %s' % format(init_val))
            conv_initializer.mean = 0.0
            conv_initializer.stddev = init_val
        # TODO: make and modify local copy only

        kernel = self._cpu_variable_init('weights', shape=kernel_shape, initializer=conv_initializer)
        # output = tf.nn.conv2d(input, kernel, stride_shape, data_format='NCHW', padding=params['padding'])

        # Keep tabs on the number of weights and memory
        self.num_weights += np.cumprod(kernel_shape)[-1]
        # self.mem += np.cumprod(output.get_shape().as_list())[-1]*self.bytesize / 1024
        # batch * width * height * in_channels * kern_h * kern_w * features
        # input = batch_size (ignore), channels, height, width
        # http://imatge-upc.github.io/telecombcn-2016-dlcv/slides/D2L1-memory.pdf
        # this_ops = np.prod(params['kernel'] + input.get_shape().as_list()[1:] + [features])
        # self.print_rank('\tops: %3.2e' % (this_ops))
        """
        # batch * width * height * in_channels * (kern_h * kern_w * channels)
        # at each location in the image:
        ops_per_conv = 2 * np.prod(params['kernel'] + [input.shape[1].value])
        # number of convolutions on the image for a single filter / output channel (stride brings down the number)
        convs_per_filt = np.prod([input.shape[2].value, input.shape[3].value]) // np.prod(params['stride'])
        # final = filters * convs/filter * ops/conv
        this_ops = np.prod([params['features'], convs_per_filt, ops_per_conv])
        if verbose:
            self.print_verbose('\t%d ops/conv, %d convs/filter, %d filters = %3.2e ops' % (ops_per_conv, convs_per_filt,
                                                                              params['features'], this_ops))
        """
        # 2*dim(input=N*H*W)*dim(kernel=H*W*N)
        this_ops = 2 * np.prod(params['kernel']) * features * np.prod(input_shape[2:])
        self.ops += this_ops

        return kernel


class YNet(FCDenseNet, FCNet):
    """
    An inverter model
    """
    def __init__(self, *args, **kwargs):
        super(YNet, self).__init__(*args, **kwargs)
        self.all_scopes = {"encoder": None, "decoder_RE": None, "decoder_IM": None, "inverter": None}
        self.all_ops = {"encoder": 0., "decoder": 0., "inverter": 0.}
        self.all_weights = {"encoder": 0., "decoder": 0., "inverter": 0.}
        self.all_mem = {"encoder": 0., "decoder": 0., "inverter": 0.}
        self.skip_connection_list = []
        self.skip_hub = -1
        self.model_output = {"encoder": None, "decoder": None, "inverter": None}
        self.network = dict([(key, itm) for key,itm in self.network.items()])

    def _batch_norm(self, input=None):
        out = tf.keras.layers.BatchNormalization(axis=1)(inputs=input, training= self.operation == 'train')
        return out

    def get_all_ops(self, subnet=None):
        if subnet is None:
            return 3 * np.sum([op for _, op in self.all_ops.items()])
        else:
            return 3 * self.all_ops[subnet]

    def update_all_attrs(self, subnet='encoder'):
        self.all_ops[subnet] = self.ops
        self.ops = 0
        self.all_weights[subnet] = self.num_weights
        self.num_weights = 0
        self.all_mem[subnet] = self.mem 
        self.mem = 0
        self.all_scopes[subnet] = self.scopes
        # self.scopes = []

    def build_encoder_DenseNet(self):
        """
        Here we build the model.
        :param summaries: bool, flag to print out summaries.
        """
        self.scopes = []
        self.print_rank('Building Neural Net ...')
        self.print_rank('input: ---, dim: %s memory: %s MB' %(format(self.images.get_shape().as_list()), format(self.mem/1024)))
        self.print_rank('***** Encoder Branch ******')
        network = self.network['encoder']

        # Initiate the remaining layers
        for layer_num, (layer_name, layer_params) in enumerate(list(network.items())):
            if layer_num == 0:
                out = self.images
            else:
                out = out
            with tf.variable_scope('encoder'+'_'+layer_name, reuse=self.reuse) as scope:
                in_shape = out.get_shape().as_list()
             
                if layer_params['type'] == 'conv_2D':
                    self.print_verbose(">>> Adding Conv Layer: %s" % layer_name)
                    self.print_verbose('    input: %s' %format(out.get_shape().as_list()))
                    out, _ = self._conv(input=out, params=layer_params)
                    do_bn = layer_params.get('batch_norm', False)
                    if do_bn:
                        out = self._batch_norm(input=out)
                    else:
                        out = self._add_bias(input=out, params=layer_params)
                        # dropout
                    if self.operation == 'train' and layer_params.get('dropout', None) is not None:
                        rate = layer_params['dropout']
                    else:
                        rate = 0
                    out = tf.keras.layers.SpatialDropout2D(rate, data_format='channels_first')(inputs=out, training= self.operation == 'train')
                    # out = tf.nn.dropout(out, rate=tf.constant(rate, dtype=out.dtype))
                    out_shape = out.get_shape().as_list()
                    out = self._activate(input=out, name=scope.name, params=layer_params)
                    self.print_verbose('    output: %s' %format(out.get_shape().as_list()))
                    if self.summary: 
                        self._activation_summary(out)
                        self._activation_image_summary(out)

                if layer_params['type'] == 'depth_conv':
                    self.print_verbose(">>> Adding depthwise Conv Layer: %s" % layer_name)
                    self.print_verbose('    input: %s' %format(out.get_shape().as_list()))
                    out, _ = self._depth_conv(input=out, params=layer_params)
                    do_bn = layer_params.get('batch_norm', False)
                    if do_bn:
                        out = self._batch_norm(input=out)
                    else:
                        out = self._add_bias(input=out, params=layer_params)
                    out = self._activate(input=out, name=scope.name, params=layer_params)
                    self.print_verbose('    output: %s' %format(out.get_shape().as_list()))
                    if self.summary: 
                        self._activation_summary(out)
                        self._activation_image_summary(out)

                if layer_params['type'] == 'coord_conv':
                    self.print_verbose(">>> Adding Coord Conv Layer: %s" % layer_name)
                    self.print_verbose('    input: %s' %format(out.get_shape().as_list()))
                    out, _ = self._coord_conv(input=out, params=layer_params)
                    do_bn = layer_params.get('batch_norm', False)
                    if do_bn:
                        out = self._batch_norm(input=out)
                    else:
                        out = self._add_bias(input=out, params=layer_params)
                    out = self._activate(input=out, name=scope.name, params=layer_params)
                    self.print_verbose('    output: %s' %format(out.get_shape().as_list()))
                    if self.summary: self._activation_summary(out)

                if layer_params['type'] == 'pooling':
                    self.print_verbose(">>> Adding Pooling Layer: %s" % layer_name)
                    self.print_verbose('    input: %s' %format(out.get_shape().as_list()))
                    out = self._pool(input=out, name=scope.name, params=layer_params)
                    self.print_verbose('    output: %s' %format(out.get_shape().as_list()))

                if layer_params['type'] == 'deconv_2D':
                    self.print_verbose(">>> Adding de-Conv Layer: %s" % layer_name)
                    self.print_verbose('    input: %s' %format(out.get_shape().as_list()))
                    out, _ = self._deconv(input=out, params=layer_params)
                    self.print_verbose('    output: %s' %format(out.get_shape().as_list()))
                    if self.summary: 
                        self._activation_summary(out) 
                        self._activation_image_summary(out)

                if layer_params['type'] == 'dense_layers_block':
                    self.print_verbose(">>> Adding Dense Layers Block: %s" % layer_name)
                    self.print_verbose('    input: %s' %format(out.get_shape().as_list()))
                    if layer_params['conv_type'] == 'conv_2D':
                        # out, _ = self._dense_layers_block_multi_head_loop(input=out, params=layer_params)
                        out, _ = self._freq_to_space(input=out, params=layer_params)
                        self.print_verbose(">>> Using Single head" )
                    else:
                        out, _ = self._dense_layers_block_multi_head_loop(input=out, params=layer_params)
                        # out, _ = self._dense_layers_block_multi_head(input=out, params=layer_params)
                        self.print_verbose(">>> Using Multi head" )
                    self.print_verbose('    output: %s' %format(out.get_shape().as_list()))
                    if self.summary: self._activation_summary(out) 
         
                if layer_params['type'] == 'dense_block_down':
                    self.print_verbose(">>> Adding Dense Block Down: %s" % layer_name)
                    self.print_verbose('    input: %s' %format(out.get_shape().as_list()))
                    out, _ = self._dense_block(out, layer_params, scope)
                    self.skip_connection_list.append(out)
                    self.skip_hub += 1
                    self.print_verbose('    output: %s' %format(out.get_shape().as_list()))

                if layer_params['type'] == 'transition_down':
                    self.print_verbose(">>> Adding Transition Down: %s" % layer_name)
                    self.print_verbose('    input: %s' %format(out.get_shape().as_list()))
                    out = self._transition_down(out, layer_params, scope)
                    self.print_verbose('    output: %s' %format(out.get_shape().as_list()))

                if layer_params['type'] == 'dense_block_bottleneck':
                    self.print_verbose(">>> Adding Dense Block Bottleneck: %s" % layer_name)
                    self.print_verbose('    input: %s' %format(out.get_shape().as_list()))
                    out, self.block_features = self._dense_block(out, layer_params, scope)
                    self.print_verbose('    output: %s' %format(out.get_shape().as_list()))
             # print layer specs and generate Tensorboard summaries
                if out is None:
                    raise NotImplementedError('Layer type: ' + layer_params['type'] + 'is not implemented!')

                out_shape = out.get_shape().as_list()
                self._print_layer_specs(layer_params, scope, in_shape, out_shape)
                self.scopes.append(scope)

                # if self.summary: 
                #     self._activation_summary(out)
                #     self._activation_image_summary(out)

        # Book-keeping...
        self.print_rank('Total # of blocks: %d,  weights: %2.1e, memory: %2.f MB, ops: %3.3e \n' % (len(network),
                                                                                        self.num_weights,
                                                                                        self.mem / 1024,
                                                                                        self.get_ops()))
        self.model_output['encoder'] = out 
        self.update_all_attrs(subnet='encoder')

    def build_encoder_Batched(self):
        """
        Here we build the model.
        :param summaries: bool, flag to print out summaries.
        """
        self.scopes = []
        self.print_rank('Building Neural Net ...')
        self.print_rank('input: ---, dim: %s memory: %s MB' %(format(self.images.get_shape().as_list()), format(self.mem/1024)))
        self.print_rank('***** Encoder Branch ******')
        network = self.network['encoder']

        def Encode(tens):
            # Initiate the remaining layers
            for layer_num, (layer_name, layer_params) in enumerate(list(network.items())):
                if layer_num == 0:
                    out = tens
                else:
                    out = out
                with tf.variable_scope('encoder'+'_'+layer_name, reuse=self.reuse) as scope:
                    in_shape = out.get_shape().as_list()
                
                    if layer_params['type'] == 'conv_2D':
                        self.print_verbose(">>> Adding Conv Layer: %s" % layer_name)
                        self.print_verbose('    input: %s' %format(out.get_shape().as_list()))
                        out, _ = self._conv(input=out, params=layer_params)
                        do_bn = layer_params.get('batch_norm', False)
                        if do_bn:
                            out = self._batch_norm(input=out)
                        else:
                            out = self._add_bias(input=out, params=layer_params)
                            # dropout
                        if self.operation == 'train' and layer_params.get('dropout', None) is not None:
                            rate = layer_params['dropout']
                        else:
                            rate = 0
                        out = tf.keras.layers.SpatialDropout2D(rate, data_format='channels_first')(inputs=out, training= self.operation == 'train')
                        # out = tf.nn.dropout(out, rate=tf.constant(rate, dtype=out.dtype))
                        out_shape = out.get_shape().as_list()
                        out = self._activate(input=out, name=scope.name, params=layer_params)
                        self.print_verbose('    output: %s' %format(out.get_shape().as_list()))
                        if self.summary: 
                            self._activation_summary(out)
                            self._activation_image_summary(out)

                    if layer_params['type'] == 'depth_conv':
                        self.print_verbose(">>> Adding depthwise Conv Layer: %s" % layer_name)
                        self.print_verbose('    input: %s' %format(out.get_shape().as_list()))
                        out, _ = self._depth_conv(input=out, params=layer_params)
                        do_bn = layer_params.get('batch_norm', False)
                        if do_bn:
                            out = self._batch_norm(input=out)
                        else:
                            out = self._add_bias(input=out, params=layer_params)
                        out = self._activate(input=out, name=scope.name, params=layer_params)
                        self.print_verbose('    output: %s' %format(out.get_shape().as_list()))
                        if self.summary: 
                            self._activation_summary(out)
                            self._activation_image_summary(out)

                    if layer_params['type'] == 'coord_conv':
                        self.print_verbose(">>> Adding Coord Conv Layer: %s" % layer_name)
                        self.print_verbose('    input: %s' %format(out.get_shape().as_list()))
                        out, _ = self._coord_conv(input=out, params=layer_params)
                        do_bn = layer_params.get('batch_norm', False)
                        if do_bn:
                            out = self._batch_norm(input=out)
                        else:
                            out = self._add_bias(input=out, params=layer_params)
                        out = self._activate(input=out, name=scope.name, params=layer_params)
                        self.print_verbose('    output: %s' %format(out.get_shape().as_list()))
                        if self.summary: self._activation_summary(out)

                    if layer_params['type'] == 'pooling':
                        self.print_verbose(">>> Adding Pooling Layer: %s" % layer_name)
                        self.print_verbose('    input: %s' %format(out.get_shape().as_list()))
                        out = self._pool(input=out, name=scope.name, params=layer_params)
                        self.print_verbose('    output: %s' %format(out.get_shape().as_list()))

                    if layer_params['type'] == 'deconv_2D':
                        self.print_verbose(">>> Adding de-Conv Layer: %s" % layer_name)
                        self.print_verbose('    input: %s' %format(out.get_shape().as_list()))
                        out, _ = self._deconv(input=out, params=layer_params)
                        self.print_verbose('    output: %s' %format(out.get_shape().as_list()))
                        if self.summary: 
                            self._activation_summary(out) 
                            self._activation_image_summary(out)

                    if layer_params['type'] == 'dense_layers_block':
                        self.print_verbose(">>> Adding Dense Layers Block: %s" % layer_name)
                        self.print_verbose('    input: %s' %format(out.get_shape().as_list()))
                        if layer_params['conv_type'] == 'conv_2D':
                            # out, _ = self._dense_layers_block_multi_head_loop(input=out, params=layer_params)
                            out, _ = self._freq_to_space(input=out, params=layer_params)
                            self.print_verbose(">>> Using Single head" )
                        else:
                            out, _ = self._dense_layers_block_multi_head_loop(input=out, params=layer_params)
                            # out, _ = self._dense_layers_block_multi_head(input=out, params=layer_params)
                            self.print_verbose(">>> Using Multi head" )
                        self.print_verbose('    output: %s' %format(out.get_shape().as_list()))
                        if self.summary: self._activation_summary(out) 
            
                    if layer_params['type'] == 'dense_block_down':
                        self.print_verbose(">>> Adding Dense Block Down: %s" % layer_name)
                        self.print_verbose('    input: %s' %format(out.get_shape().as_list()))
                        out, _ = self._dense_block(out, layer_params, scope)
                        self.skip_connection_list.append(out)
                        self.skip_hub += 1
                        self.print_verbose('    output: %s' %format(out.get_shape().as_list()))

                    if layer_params['type'] == 'transition_down':
                        self.print_verbose(">>> Adding Transition Down: %s" % layer_name)
                        self.print_verbose('    input: %s' %format(out.get_shape().as_list()))
                        out = self._transition_down(out, layer_params, scope)
                        self.print_verbose('    output: %s' %format(out.get_shape().as_list()))

                    if layer_params['type'] == 'dense_block_bottleneck':
                        self.print_verbose(">>> Adding Dense Block Bottleneck: %s" % layer_name)
                        self.print_verbose('    input: %s' %format(out.get_shape().as_list()))
                        out, self.block_features = self._dense_block(out, layer_params, scope)
                        self.print_verbose('    output: %s' %format(out.get_shape().as_list()))

                    # if layer_params['type'] == 'fully_connected_block':
                    #     self.print_verbose(">>> Adding Dense Block Bottleneck: %s" % layer_name)
                    #     self.print_verbose('    input: %s' %format(out.get_shape().as_list()))
                    #     out = self._fully_connected_block(out, layer_params, scope)
                    #     self.print_verbose('    output: %s' %format(out.get_shape().as_list()))

                # print layer specs and generate Tensorboard summaries
                    if out is None:
                        raise NotImplementedError('Layer type: ' + layer_params['type'] + 'is not implemented!')

                    out_shape = out.get_shape().as_list()
                    self._print_layer_specs(layer_params, scope, in_shape, out_shape)
                    self.scopes.append(scope)

                    # if self.summary: 
                    #     self._activation_summary(out)
                    #     self._activation_image_summary(out)

            # Book-keeping...
            self.print_rank('Total # of blocks: %d,  weights: %2.1e, memory: %s MB, ops: %3.2e \n' % (len(network),
                                                                                            self.num_weights,
                                                                                            format(self.mem / 1024),
                                                                                            self.get_ops()))
            return out

        # # Prepare input
        # inputs = self.images
        # new_shape = [inputs.shape.as_list()[0], inputs.shape.as_list()[1], -1, 
        #              inputs.shape.as_list()[-2], inputs.shape.as_list()[-1]] 
        # tensor_slices = tf.reshape(inputs, new_shape)
        # tensor_slices = tf.transpose(tensor_slices, perm=[1, 0, 2, 3, 4])

        # # Apply encoder
        # out = tf.map_fn(Encode, tensor_slices, back_prop=True, swap_memory=True, parallel_iterations=16)
        # self.print_rank('output of Encoder', out.get_shape())
        # out = tf.transpose(out, perm= [1, 2, 0])
        inputs = self.images
        new_shape = [inputs.shape.as_list()[0], inputs.shape.as_list()[1], -1, 
                     inputs.shape.as_list()[-2], inputs.shape.as_list()[-1]]
        re_shape = [inputs.shape.as_list()[0] * inputs.shape.as_list()[1], 1, 
                     inputs.shape.as_list()[-2], inputs.shape.as_list()[-1]]  
        inputs = tf.reshape(self.images, re_shape)
        out = Encode(inputs)
        out = tf.reshape(out, [self.params['batch_size'], new_shape[1], -1])
        # out = tf.transpose(out, perm=[1,0,2])
        self.print_rank('output of Encoder- before fc block ', out.get_shape()) 
        
        out = self.fully_connected_block(out, self.network['encoder']['fully_connected_block'], 'encoder')
       
        self.print_rank('output of Encoder- after fc block ', out.get_shape()) 
        # out = tf.transpose(out, perm=[0, 2, 1])
        # dim = int(math.sqrt(self.images.shape.as_list()[1]))
        # out = tf.reshape(out, [self.params['batch_size'], -1, dim, dim])
        self.print_rank('output of Encoder', out.get_shape())
        self.model_output['encoder'] = out 
        self.update_all_attrs(subnet='encoder')

    def build_encoder(self):
        """
        Here we build the model.
        :param summaries: bool, flag to print out summaries.
        """
        self.scopes = []
        self.print_rank('Building Neural Net ...')
        self.print_rank('input: ---, dim: %s memory: %s MB' %(format(self.images.get_shape().as_list()), format(self.mem/1024)))
        self.print_rank('***** Encoder Branch ******')
        network = self.network['encoder']
        params = network['freq2space']['cvae_params']
        inputs = self.images
        new_shape = [inputs.shape.as_list()[0], inputs.shape.as_list()[1], -1, 
                     inputs.shape.as_list()[-2], inputs.shape.as_list()[-1]] 
        tensor_slices = tf.reshape(inputs, new_shape)
        tensor_slices = tf.transpose(tensor_slices, perm=[1, 0, 2, 3, 4])
        conv_1by1 = params['conv_params']
        fully_connected = params['fc_params']
        num_conv = params['n_conv_layers']
        num_fc = params['n_fc_layers']
        # conv_1by1 = OrderedDict({'type': 'conv_2D', 'stride': [2, 2], 'kernel': [4, 4], 
        #                         'features': 16,
        #                         'activation': 'relu', 'padding': 'SAME', 'batch_norm': True, 'dropout': 0.0})
        # pool = OrderedDict({'type': 'pooling', 'stride': [2, 2], 'kernel': [2, 2], 'pool_type': 'max','padding':'SAME'})
        # fully_connected = OrderedDict({'type': 'fully_connected','weights': 1024,'bias': 1024, 'activation': 'relu',
        #                            'regularize': True})
        self.print_verbose("\t>>> Adding CVAE: " )
        self.print_verbose('\t\t    input: %s' %format(inputs.get_shape().as_list()))
        # pre_ops = deepcopy(self.ops)
        def CVAE(tens):
            #TODO, turn this into a full denoising VAE
            for i in range(num_conv):
                with tf.variable_scope('CVAE_block_%d' % i , reuse=self.reuse) as scope:
                    tens , _ = self._conv(input=tens, params=conv_1by1)
                    # tens = self._pool(input=tens, params=pool)
                    tens = self._activate(input=tens, params=conv_1by1)
                    tens = self._batch_norm(input=tens)
            # if tens.shape[-2:] != [32, 32]:
            #     tens = tf.transpose(tens, perm=[0, 2, 3, 1])
            #     tens = tf.image.resize(tens, [32, 32], method=tf.image.ResizeMethod.BILINEAR)
            #     if self.params['IMAGE_FP16']:
            #         tens = tf.saturate_cast(tens, tf.float16)
            #     tens = tf.transpose(tens, perm=[0, 3, 1, 2])
            # self.print_rank('shape inside CVAE', tens.get_shape())
            # for i in range(num_fc):
            #     with tf.variable_scope('CVAE_fc_%d' %i, reuse=self.reuse) as _ :
            #         tens = self._linear(input=tens, params=fully_connected)
            #         tens = self._activate(input=tens, params=fully_connected)
            # # tens = tf.reshape(tens, [new_shape[0], -1])
            return tens

        # post_ops = deepcopy(self.ops)
        # self.print_rank("post pre, cvae ops: ", pre_ops - post_ops)
        out = tf.map_fn(CVAE, tensor_slices, back_prop=True, swap_memory=True, parallel_iterations=256)
        # self.print_rank('output of CVAE', out.get_shape())
        # out = tf.transpose(out, perm= [1, 2, 0])
        # dim = int(math.sqrt(self.images.shape.as_list()[1]))
        # out = tf.reshape(out, [self.params['batch_size'], -1, dim, dim])
        # out = tf.transpose(out, perm=[1,0,2,3])
        self.print_rank('output of Encoder', out.get_shape())
        self.model_output['encoder'] = out 
        self.update_all_attrs(subnet='encoder')

    def fully_connected_block(self, inputs, layer_params, scope_name):
        fc_params = OrderedDict({'type': 'fully_connected','weights': 1024,'bias': 1024, 'activation': layer_params['activation'],
                                   'regularize': True})
        num_fc = layer_params['n_fc_layers']
        out = inputs
        for i in range(num_fc):
            with tf.variable_scope('%s_fc_%d' %(scope_name, i), reuse=self.reuse) as _ :
                lin_initializer = tf.uniform_unit_scaling_initializer
                if isinstance(lin_initializer, tf.uniform_unit_scaling_initializer):
                    if fc_params['type'] == 'fully_connected':
                        if fc_params['activation'] == 'tanh':
                            lin_initializer.factor = 1.15
                        elif fc_params['activation'] == 'relu':
                            lin_initializer.factor = 1.43
                    elif fc_params['type'] == 'linear_output':
                        lin_initializer.factor = 1.0
                elif isinstance(lin_initializer, tf.random_normal_initializer):
                    init_val = max(np.sqrt(2.0 / fc_params['weights']), 0.01)
                    lin_initializer.mean = 0.0
                    lin_initializer.stddev = init_val
                weights_shape = [1] + [out.shape.as_list()[-1]] + [fc_params['weights']]
                bias_shape = [fc_params['bias']]
                weights = self._cpu_variable_init('weights', shape=weights_shape, initializer=lin_initializer,
                                                regularize=fc_params['regularize'])
                bias = self._cpu_variable_init('bias', bias_shape, initializer=tf.zeros_initializer)
                weights_mat = tf.tile(weights, [self.params['batch_size'], 1, 1])
                # bias_mat = tf.tile(bias, [self.params['batch_size'], 1])
                out = tf.matmul(out, weights_mat)
                out = out + bias
                out = self._activate(input=out, params=fc_params) 
        return out

    def _build_branch(self, subnet='decoder', scope= None, inputs=None):
        self.scopes = []
        self.print_rank('***** %s Branch ******' % subnet)
        network = self.network[subnet]
        subnet_scope = subnet if scope is None else scope
        # out = self.model_output['encoder']
        # if inputs is not None:
        #     out = inputs
        # out = self.model_output['encoder'] if inputs is None else inputs
        local_skip_hub = self.skip_hub
        block_features = self.model_output['encoder'] if inputs is None else inputs
        out = inputs

        for layer_num, (layer_name, layer_params) in enumerate(list(network.items())):
            with tf.variable_scope(subnet_scope+'_'+layer_name, reuse=self.reuse) as scope:
                in_shape = out.get_shape().as_list()
             
                if layer_params['type'] == 'conv_2D':
                    self.print_verbose(">>> Adding Conv Layer: %s" % layer_name)
                    self.print_verbose('    input: %s' %format(out.get_shape().as_list()))
                    out, _ = self._conv(input=out, params=layer_params)
                    out = self._activate(input=out, name=scope.name, params=layer_params)
                    do_bn = layer_params.get('batch_norm', False)
                    if do_bn:
                        out = self._batch_norm(input=out)
                    else:
                        out = self._add_bias(input=out, params=layer_params)
                    rate = layer_params.get('dropout', 0) 
                    out = tf.keras.layers.SpatialDropout2D(rate, data_format='channels_first')(inputs=out, training= self.operation == 'train')
                    out_shape = out.get_shape().as_list()
                    self.print_verbose('    output: %s' %format(out.get_shape().as_list()))
                    if self.summary: 
                        self._activation_summary(out)
                        self._activation_image_summary(out)
                
                if layer_params['type'] == 'residual':
                    self.print_verbose(">>> Adding Residual Block: %s" % layer_name)
                    out, _ = self._residual_unit(inputs=out, params=layer_params)
                    out_shape = out.get_shape().as_list()
                    self.print_verbose('    output: %s' %format(out.get_shape().as_list()))
                    if self.summary: 
                        self._activation_summary(out)
                        self._activation_image_summary(out)

                if layer_params['type'] == 'depth_conv':
                    self.print_verbose(">>> Adding depthwise Conv Layer: %s" % layer_name)
                    self.print_verbose('    input: %s' %format(out.get_shape().as_list()))
                    out, _ = self._depth_conv(input=out, params=layer_params)
                    do_bn = layer_params.get('batch_norm', False)
                    if do_bn:
                        out = self._batch_norm(input=out)
                    else:
                        out = self._add_bias(input=out, params=layer_params)
                    out = self._activate(input=out, name=scope.name, params=layer_params)
                    self.print_verbose('    output: %s' %format(out.get_shape().as_list()))
                    if self.summary: 
                        self._activation_summary(out)
                        self._activation_image_summary(out)

                if layer_params['type'] == 'coord_conv':
                    self.print_verbose(">>> Adding Coord Conv Layer: %s" % layer_name)
                    self.print_verbose('    input: %s' %format(out.get_shape().as_list()))
                    out, _ = self._coord_conv(input=out, params=layer_params)
                    do_bn = layer_params.get('batch_norm', False)
                    if do_bn:
                        out = self._batch_norm(input=out)
                    else:
                        out = self._add_bias(input=out, params=layer_params)
                    out = self._activate(input=out, name=scope.name, params=layer_params)
                    self.print_verbose('    output: %s' %format(out.get_shape().as_list()))
                    if self.summary: self._activation_summary(out)

                if layer_params['type'] == 'pooling':
                    self.print_verbose(">>> Adding Pooling Layer: %s" % layer_name)
                    self.print_verbose('    input: %s' %format(out.get_shape().as_list()))
                    out = self._pool(input=out, name=scope.name, params=layer_params)
                    self.print_verbose('    output: %s' %format(out.get_shape().as_list()))

                if layer_params['type'] == 'deconv_2D':
                    self.print_verbose(">>> Adding de-Conv Layer: %s" % layer_name)
                    self.print_verbose('    input: %s' %format(out.get_shape().as_list()))
                    if subnet == 'inverter':
                        out = self._upscale(inputs=out, params=layer_params)
                    else:
                        out, _ = self._deconv(input=out, params=layer_params)
                    self.print_verbose('    output: %s' %format(out.get_shape().as_list()))
                    if self.summary: 
                        self._activation_summary(out) 
                        self._activation_image_summary(out)

                if layer_params['type'] == 'dense_layers_block':
                    self.print_verbose(">>> Adding Dense Layers Block: %s" % layer_name)
                    self.print_verbose('    input: %s' %format(out.get_shape().as_list()))
                    if layer_params['conv_type'] == 'conv_2D':
                        # out, _ = self._dense_layers_block_multi_head_loop(input=out, params=layer_params)
                        out, _ = self._freq_to_space(input=out, params=layer_params)
                        self.print_verbose(">>> Using Single head" )
                    else:
                        out, _ = self._dense_layers_block_multi_head_loop(input=out, params=layer_params)
                        # out, _ = self._dense_layers_block_multi_head(input=out, params=layer_params)
                        self.print_verbose(">>> Using Multi head" )
                    self.print_verbose('    output: %s' %format(out.get_shape().as_list()))
                    if self.summary: self._activation_summary(out) 
         
                if layer_params['type'] == 'dense_block_up':
                    self.print_verbose(">>> Adding Dense Block Up: %s" % layer_name)
                    self.print_verbose('    input: %s' %format(out.get_shape().as_list()))
                    out, block_features = self._dense_block(out, layer_params, scope)
                    self.print_verbose('    output: %s' %format(out.get_shape().as_list()))

                if layer_params['type'] == 'transition_up':
                    self.print_verbose(">>> Adding Transition Up: %s" % layer_name)
                    self.print_verbose('    input: %s' %format(out.get_shape().as_list()))
                    # out = self._transition_up(self.block_features, self.skip_connection_list[local_skip_hub], layer_params, scope)
                    out = self._transition_up(block_features, None, layer_params, scope)
                    self.print_verbose('    output: %s' %format(out.get_shape().as_list()))
                    local_skip_hub -= 1

                # print layer specs and generate Tensorboard summaries
                if out is None:
                    raise NotImplementedError('Layer type: ' + layer_params['type'] + 'is not implemented!')

                out_shape = out.get_shape().as_list()
                self._print_layer_specs(layer_params, scope, in_shape, out_shape)
                self.scopes.append(scope)
                # if self.summary: 
                #     self._activation_summary(out)
                #     self._activation_image_summary(out)
        self.print_rank('Total # of blocks: %d,  weights: %2.1e, memory: %s MB, ops: %3.2e \n' % (len(network),
                                                                                    self.num_weights,
                                                                                    format(self.mem / 1024),
                                                                                    self.get_ops()))
        # if subnet == 'inverter':
        #     out = tf.reduce_mean(out, axis=1, keepdims=True)
        self.model_output[subnet] = out
        self.update_all_attrs(subnet=subnet)
    
    def _upscale(self, inputs=None, params=None, scale=2):
        conv_params = OrderedDict({'type': 'conv_2D', 'stride': [1, 1], 'kernel': [1, 1], 
                                'features': inputs.shape.as_list()[1]//2,
                                'activation': 'relu', 
                                'padding': 'VALID', 
                                'batch_norm': True, 'dropout':0.0})
        with tf.variable_scope('upscale', reuse=self.reuse) as _:
            shape = inputs.shape
            out = tf.reshape(inputs, [-1, shape[1], shape[2], 1, shape[3], 1])
            out = tf.tile(out, [1, 1, 1, scale, 1, scale])
            out = tf.reshape(out, [-1, shape[1], shape[2] * scale, shape[3] * scale])
            out, _ = self._conv(input=out, params=conv_params)
            return out

    def _residual_unit(self, inputs=None, params=None):
        conv_params = OrderedDict({'type': 'conv_2D', 'stride': [1, 1], 'kernel': [1, 1], 
                                'features': inputs.shape.as_list()[1],
                                'activation': 'relu', 
                                'padding': 'VALID', 
                                'batch_norm': True, 'dropout':0.0})
        out = self._batch_norm(input=inputs)
        out = self._activate(input=out, params=conv_params)
        with tf.variable_scope('residual_conv_1', reuse=self.reuse) as scope:
            out, _ = self._conv(input=out, params=conv_params)
        out = self._batch_norm(input=out)
        out = self._activate(input=out, params=conv_params)
        with tf.variable_scope('residual_conv_2', reuse=self.reuse) as scope:
            out, _ = self._conv(input=out, params=conv_params)
        out = tf.add(inputs, out)
        return out, None

    def build_decoder(self, subnet='decoder_IM'):
        scopes_list = []
        out = self.model_output['encoder']
        out_shape = out.shape.as_list()
        params = self.network['encoder']['freq2space']['cvae_params']
        fully_connected = params['fc_params']
        num_fc = params['n_fc_layers']
        conv_1by1_1 = OrderedDict({'type': 'conv_2D', 'stride': [1, 1], 'kernel': [1, 1], 
                                'features': 1,
                                'activation': 'relu', 
                                'padding': 'VALID', 
                                'batch_norm': True, 'dropout':0.0})
        conv_1by1_1024 = OrderedDict({'type': 'conv_2D', 'stride': [1, 1], 'kernel': [1, 1], 
        'features': 1024,
        'activation': 'relu', 
        'padding': 'VALID', 
        'batch_norm': True, 'dropout':0.0})
        if False:
            def fc_map(tens):
                for i in range(num_fc):
                    with tf.variable_scope('%s_fc_%d' %(subnet, i), reuse=self.reuse) as scope :
                        tens = self._linear(input=tens, params=fully_connected)
                        tens = self._activate(input=tens, params=fully_connected)
                        # scopes_list.append(scope)
                return tens
            out = tf.map_fn(fc_map, out, back_prop=True, swap_memory=True, parallel_iterations=256)
            out = tf.transpose(out, perm= [1, 2, 0])
            dim = int(math.sqrt(self.images.shape.as_list()[1]))
            out = tf.reshape(out, [self.params['batch_size'], -1, dim, dim])
        else:
            out = tf.reshape(out, [out_shape[0]*out_shape[1], out_shape[2], out_shape[3], out_shape[4]])
            with tf.variable_scope('%s_conv_1by1_1' % subnet, reuse=self.reuse) as scope:
                out, _ = self._conv(input=out, params=conv_1by1_1)
                out = tf.reshape(out, [out_shape[0], out_shape[1], out_shape[3], out_shape[4]])
                out = tf.transpose(out, perm=[1,0,2,3])

                # scopes_list.append(scope)
                print('conv1by1_decoder shape', out.shape.as_list())
        with tf.variable_scope('%s_conv_1by1_1024' % subnet, reuse=self.reuse) as scope:
            out, _ = self._conv(input=out, params=conv_1by1_1024) 
            out = self._activate(input=out, params=conv_1by1_1024)
            do_bn = conv_1by1_1024.get('batch_norm', False)
            if do_bn:
                out = self._batch_norm(input=out)
            else:
                out = self._add_bias(input=out, params=conv_1by1_1024)
            # scopes_list.append(scope)

        self._build_branch(subnet=subnet, inputs=out)

        # conv_1by1 = OrderedDict({'type': 'conv_2D', 'stride': [1, 1], 'kernel': [1, 1], 'features': 1,
        #                     'activation': 'relu', 'padding': 'SAME', 'batch_norm': False})
        # with tf.variable_scope('%s_CONV_FIN' % subnet, reuse=self.reuse) as scope:
        #     out, _ = self._conv(input=self.model_output[subnet], params=conv_1by1)
        #     do_bn = conv_1by1.get('batch_norm', False)
        #     if do_bn:
        #         out = self._batch_norm(input=out)
        #     else:
        #         out = self._add_bias(input=out, params=conv_1by1)
        #     out = self._activate(input=out, params=conv_1by1)
        #     scopes_list.append(scope)
        # self.model_output[subnet] = out
        # self.all_scopes[subnet] += scopes_list

    def build_inverter(self):
        out = self.model_output['encoder']
        params = self.network['encoder']['freq2space']['cvae_params']
        fully_connected = params['fc_params']
        num_fc = params['n_fc_layers']
        scopes_list = []
        if True:
            def fc_map(tens):
                for i in range(num_fc):
                    with tf.variable_scope('Inverter_fc_%d' %i, reuse=self.reuse) as scope :
                        tens = self._linear(input=tens, params=fully_connected)
                        tens = self._activate(input=tens, params=fully_connected)
                        # scopes_list.append(scope)
                return tens
            out = tf.map_fn(fc_map, out, back_prop=True, swap_memory=True, parallel_iterations=256)
            out = tf.transpose(out, perm= [1, 2, 0])
            dim = int(math.sqrt(self.images.shape.as_list()[1]))
            out = tf.reshape(out, [self.params['batch_size'], -1, dim, dim])
        else:
            conv_1by1_1 = OrderedDict({'type': 'conv_2D', 'stride': [1, 1], 'kernel': [1, 1], 
                                'features': 1,
                                'activation': 'relu', 
                                'padding': 'VALID', 
                                'batch_norm': True, 'dropout':0.0})
            out_shape = out.shape.as_list()
            out = tf.reshape(out, [out_shape[0]*out_shape[1], out_shape[2], out_shape[3], out_shape[4]])
            with tf.variable_scope('%s_conv_1by1_1' % 'inverter', reuse=self.reuse) as scope:
                out, _ = self._conv(input=out, params=conv_1by1_1)
                out = tf.reshape(out, [out_shape[0], out_shape[1], out_shape[3], out_shape[4]])
                out = tf.transpose(out, perm=[1,0,2,3])
                scopes_list.append(scope)
                print('conv1by1_inverter shape', out.shape.as_list())
        conv_1by1_1024 = OrderedDict({'type': 'conv_2D', 'stride': [1, 1], 'kernel': [1, 1], 
            'features': 1024,
            'activation': 'relu', 
            'padding': 'VALID', 
            'batch_norm': True, 'dropout':0.0})
        with tf.variable_scope('inverter_conv_1by1_1024', reuse=self.reuse) as scope:
            out, _ = self._conv(input=out, params=conv_1by1_1024) 
            do_bn = conv_1by1_1024.get('batch_norm', False)
            if do_bn:
                out = self._batch_norm(input=out)
            else:
                out = self._add_bias(input=out, params=conv_1by1_1024)
            out = self._activate(input=out, params=conv_1by1_1024)
            # scopes_list.append(scope)
        self._build_branch(subnet='inverter', inputs=out)
        self.all_scopes['inverter'] += scopes_list

    def build_model(self):
        self.build_encoder()
        self.build_decoder(subnet='decoder_IM')
        self.build_decoder(subnet='decoder_RE')
        self.build_inverter()
        self.scopes = list(chain.from_iterable([scope for _, scope in self.all_scopes.items()]))
        # self.scopes = None
        # for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        #     self.print_rank("var:%s , dtype:%s" % (var.name, var.dtype))
        # ##TODO: check why one of the GRADIENTS COMES OUT AS INT32 WHEN DOING BATCHNORM

    
