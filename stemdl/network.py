"""
Created on 10/8/17.
@author: Numan Laanait, Suhas Somnath
email: laanaitn@ornl.gov, somnaths@ornl.gov
"""

from collections import OrderedDict
import re
import tensorflow as tf
import numpy as np
from tensorflow.python.training import moving_averages


# If a model is trained with multiple GPUs, prefix all Op names with worker_name
# to differentiate the operations. But then remove from the summaries
worker_name = 'worker'


class ConvNet(object):
    """
    Vanilla Convolutional Neural Network (Feed-Forward).
    """
    def __init__(self, scope, flags, global_step, hyper_params, network, images, labels, operation='train',
                 summary=False):
        """
        :param flags: tf.app.flags
        :param global_step: as it says
        :param hyper_params: dictionary, hyper-parameters
        :param network: collections.OrderedDict, specifies ConvNet layers
        :param images: batch of images
        :param labels: batch of labels
        :param operation: string, 'train' or 'eval'
        :param summary: bool, flag to write tensorboard summaries
        :return:
        """
        self.scope = scope
        self.flags = flags
        self.global_step = global_step
        self.hyper_params = hyper_params
        self.network = network
        self.images = images
        self.labels = labels
        self.net_type = self.hyper_params['network_type']
        assert self.net_type == 'regressor' or self.net_type == 'classifier',\
            "'net_type' must be 'regressor' or 'classifier'"
        self.operation = operation
        assert self.operation == 'train' or self.operation == 'eval',\
            "'operation' must be 'train' or 'eval'"
        self.summary = summary
        self.num_weights = 0
        self.misc_ops = []
        if self.scope == self.flags.worker_name+'_0/':
            self.reuse = None
        else:
            self.reuse = True
        self.bytesize = 2
        if not self.flags.IMAGE_FP16: self.bytesize = 4
        self.mem = np.cumprod(self.images.get_shape())[-1]*self.bytesize/1024  # (in KB)
        self.ops = 0
        self.initializer = self._get_initializer(hyper_params.get('initializer', None))
        if "batch_norm" in self.hyper_params:
            self.hyper_params["batch_norm"]["decay"] = self.hyper_params["batch_norm"].get("decay", 0.995)
            self.hyper_params["batch_norm"]["epsilon"] = self.hyper_params["batch_norm"].get("epsilon", 1E-5)
        else:
            # default params
            self.hyper_params["batch_norm"] = {"epsilon": 1E-5, "decay": 0.995}
        self.model_output = None

    def build_model(self, summaries=False):
        """
        Here we build the model.
        :param summaries: bool, flag to print out summaries.
        """
        # Initiate 1st layer
        print('Building Neural Net on %s...' % self.scope)
        print('input: ---, dim: %s memory: %s MB' %(format(self.images.get_shape()), format(self.mem/1024)))
        layer_name, layer_params = list(self.network.items())[0]
        with tf.variable_scope(layer_name, reuse=self.reuse) as scope:
            out, kernel = self._conv(input=self.images, params=layer_params)
            if layer_params['batch_norm']:
                out = self._batch_norm(input=out)
            else:
                out = self._add_bias(input=out, params=layer_params)
            out = self._activate(input=out, name=scope.name, params=layer_params)
            in_shape = self.images.get_shape()
            # Tensorboard Summaries
            if self.summary:
                self._activation_summary(out)
                self._activation_image_summary(out)
                self._kernel_image_summary(kernel)

            self._print_layer_specs(layer_params, scope, in_shape, out.get_shape())

        # Initiate the remaining layers
        for layer_name, layer_params in list(self.network.items())[1:]:
            with tf.variable_scope(layer_name, reuse=self.reuse) as scope:
                in_shape = out.get_shape()
                if layer_params['type'] == 'convolutional':
                    out, _ = self._conv(input=out, params=layer_params)
                    if layer_params['batch_norm']:
                        out = self._batch_norm(input=out)
                    else:
                        out = self._add_bias(input=out, params=layer_params)
                    out = self._activate(input=out, name=scope.name, params=layer_params)
                    if self.summary: self._activation_summary(out)

                if layer_params['type'] == 'pooling':
                    out = self._pool(input=out, name=scope.name, params=layer_params)

                if layer_params['type'] == 'fully_connected':
                    out = self._linear(input=out, name=scope.name+'_preactiv', params=layer_params)
                    out = self._activate(input=out, name=scope.name, params=layer_params)
                    keep_prob = layer_params.get('dropout', None)
                    if keep_prob is not None:
                        out = self._dropout(input=out, name=scope.name+ '_dropout')
                    if self.summary: self._activation_summary(out)

                if layer_params['type'] == 'linear_output':
                    in_shape = out.get_shape()
                    out = self._linear(input=out, name=scope.name, params=layer_params)
                    assert out.get_shape()[-1] == self.flags.OUTPUT_DIM, 'Dimensions of the linear output layer' + \
                                                                         'do not match the expected output set in' + \
                                                                         'tf.app.flags. Check flags or network_config.json'
                    if self.summary: self._activation_summary(out)

                if layer_params['type'] not in ['convolutional', 'pooling', 'fully_connected', 'linear_output']:
                    out = self._compound_layer(out, layer_params, scope.name)
                    # Continue any summary
                    if self.summary: self._activation_summary(out)

                # print layer specs and generate Tensorboard summaries
                out_shape = out.get_shape()
                self._print_layer_specs(layer_params, scope, in_shape, out_shape)

        print('Total # of layers: %d,  weights: %2.1e, memory: %s MB, ops: %3.2e \n' % (len(self.network),
                                                                                        self.num_weights,
                                                                                        format(self.mem / 1024),
                                                                                        self.ops))

        # reference the output
        self.model_output = out

    def _compound_layer(self, out, layer_params, scope_name):
        """
        Handles the computation of more complex layer types such as Residual blocks, Inception, etc.

        :param out: 4D tensor, Input to the layer
        :param layer_params: OrderedDictionary, Parameters for the layer
        :param scope_name: str, name of the layer
        :return: 4D tensor, output of the layer
        """
        pass

    @staticmethod
    def _get_initializer(params):
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
                    print('using ' + name + ' initializer')
                    # Random walk initialization (currently in the code).
                    return tf.uniform_unit_scaling_initializer(**params_copy)
                elif name == 'truncated_normal':
                    print('using ' + name + ' initializer')
                    return tf.truncated_normal_initializer(**params_copy)
                elif name == 'variance_scaling':
                    print('using ' + name + ' initializer')
                    return tf.contrib.layers.variance_scaling_initializer(**params_copy)
                elif name == 'random_normal':
                    print('using ' + name + ' initializer')
                    # Normalized Initialization ( eq. 16 in Glorot et al.).
                    return tf.random_normal_initializer(**params_copy)
                elif name == 'random_uniform':
                    print('using ' + name + ' initializer')
                    return tf.random_uniform_initializer(**params_copy)
                elif name == 'xavier':  # Glorot uniform initializer, also called Xavier
                    # http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
                    print('using ' + name + ' initializer')
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
                        print('using ' + name + ' initializer')
                        return tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', **params_copy)
                    elif name == 'lecun':
                        _ = params_copy.pop('factor', None)  # force it to be 1.0
                        _ = params_copy.pop('mode', None)  # force it to be 'FAN_IN' (default anyway)
                        # uniform parameter is False by default -> normal distribution
                        print('using ' + name + ' initializer')
                        return tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN', **params_copy)
                    print('Requested initializer: ' + name + ' has not yet been implemented.')
        # default = Xavier:
        print('Using default Xavier instead')
        return tf.contrib.layers.xavier_initializer()

    def get_loss(self):
        with tf.variable_scope(self.scope, reuse=self.reuse) as scope:
            if self.net_type == 'regressor':
                self._calculate_loss_regressor(self.hyper_params['loss_function'])
            if self.net_type == 'classifier':
                self._calculate_loss_classifier()

    def get_misc_ops(self):
        ops = tf.group(*self.misc_ops)
        return ops

    # Loss calculation and regularization helper methods
    def _calculate_loss_regressor(self, params):
        """
        Calculate the loss objective for regression
        :param params: dictionary, specifies the objective to use
        :return: cost
        """
        labels = tf.cast(self.labels, tf.float64)
        assert params['type'] == 'Huber' or params['type'] == 'MSE',\
            "Type of regression loss function must be 'Huber' or 'MSE'"
        if params['type'] == 'Huber':
            # decay the residual cutoff exponentially
            decay_steps = int(self.flags.NUM_EXAMPLES_PER_EPOCH / self.flags.batch_size \
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
            cost = tf.losses.huber_loss(labels, predictions=self.model_output, delta=residual_tol,
                                        reduction=tf.losses.Reduction.MEAN)
        if params['type'] == 'MSE':
            cost = tf.losses.mean_squared_error(labels, predictions=self.model_output,
                                                reduction=tf.losses.Reduction.MEAN)

        return cost

    def _calculate_loss_classifier(self):
        """
        Calculate the loss objective for classification
        :param params: dictionary, specifies the objective to use
        :return: cost
        """
        labels = self.labels
        labels = tf.argmax(labels, axis=1)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=self.model_output)
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        precision_1 = tf.scalar_mul(1. / self.flags.batch_size,
                                    tf.reduce_sum(tf.cast(tf.nn.in_top_k(self.model_output, labels, 1), tf.float32)))
        precision_5 = tf.scalar_mul(1. / self.flags.batch_size,
                                    tf.reduce_sum(tf.cast(tf.nn.in_top_k(self.model_output, labels, 5), tf.float32)))
        if self.summary :
            tf.summary.scalar('precision@1_train', precision_1)
            tf.summary.scalar('precision@5_train', precision_5)
        tf.add_to_collection(tf.GraphKeys.LOSSES, cross_entropy_mean)
        return cross_entropy_mean

    # Network layers helper methods
    def _conv(self, input=None, params=None, verbose=True):
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
        conv_initializer = self.initializer
        if isinstance(conv_initializer, tf.truncated_normal_initializer):
            init_val = np.sqrt(2.0/(kernel_shape[0] * kernel_shape[1] * features))
            conv_initializer.stddev = init_val
        elif isinstance(conv_initializer, tf.uniform_unit_scaling_initializer):
            conv_initializer.factor = 1.43
        elif isinstance(conv_initializer, tf.random_normal_initializer):
            init_val = np.sqrt(2.0/(kernel_shape[0] * kernel_shape[1] * features))
            if verbose:
                print('stddev: %s' % format(init_val))
            conv_initializer.mean = 0.0
            conv_initializer.stddev = init_val
        # TODO: make and modify local copy only

        kernel = self._cpu_variable_init('weights', shape=kernel_shape, initializer=conv_initializer)
        output = tf.nn.conv2d(input, kernel, stride_shape, data_format='NCHW', padding=params['padding'])

        # Keep tabs on the number of weights and memory
        self.num_weights += np.cumprod(kernel_shape)[-1]
        self.mem += np.cumprod(output.get_shape())[-1]*self.bytesize / 1024
        # batch * width * height * in_channels * kern_h * kern_w * features
        # at each location in the image:
        ops_per_conv = 2 * np.prod(params['kernel'] + [input.shape[1].value])
        # number of convolutions on the image for a single filter / output channel (stride brings down the number)
        convs_per_filt = np.prod([input.shape[2].value, input.shape[3].value]) // np.prod(params['stride'])
        # final = num images * filters * convs/filter * ops/conv
        this_ops = np.prod([params['features'], input.shape[0].value, convs_per_filt, ops_per_conv])
        if verbose:
            print('\t%d ops/conv, %d convs/filter, %d filters, %d examples = %3.2e ops' % (ops_per_conv, convs_per_filt,
                                                                              params['features'], input.shape[0].value,
                                                                              this_ops))
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
        self.ops += bias_shape
        return output

    def _batch_norm(self, input=None):
        """
        Batch normalization
        :param input: as it says
        :return:
        """
        # Initializing hyper_parameters
        shape = [input.shape[1].value]
        beta = self._cpu_variable_init('beta', shape=shape, initializer=tf.zeros_initializer())
        gamma = self._cpu_variable_init('gamma', shape=shape, initializer=tf.ones_initializer())

        epsilon = self.hyper_params["batch_norm"]["epsilon"]
        decay = self.hyper_params["batch_norm"]["decay"]
        if self.operation == 'train':
            output, mean, variance = tf.nn.fused_batch_norm(input, gamma, beta, None, None, epsilon, data_format='NCHW',
                                                            is_training=True)
            moving_mean = self._cpu_variable_init('moving_mean', shape=shape,
                                                  initializer=tf.zeros_initializer(), trainable=False)
            moving_variance = self._cpu_variable_init('moving_variance', shape=shape,
                                                      initializer=tf.ones_initializer(), trainable=False)
            self.misc_ops.append(moving_averages.assign_moving_average(
                moving_mean, mean, decay))
            self.misc_ops.append(moving_averages.assign_moving_average(
                moving_variance, variance, decay))

        if self.operation == 'eval':
            mean = tf.get_variable('moving_mean', shape=shape)
            variance = tf.get_variable('moving_variance', shape=shape)
            output, _, _ = tf.nn.fused_batch_norm(input, gamma, beta, mean, variance, epsilon=1.e-3, data_format='NCHW',
                                                  is_training=False)
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
        assert params['weights'] == params['bias'], " weights and bias outer dimensions do not match"
        input_reshape = tf.reshape(input,[self.flags.batch_size, -1])
        dim_input = input_reshape.shape[1].value
        # print(dim_input,list(params['weights']))
        weights_shape = [dim_input, params['weights']]
        bias_shape = [params['bias']]

        if params['type'] == 'linear_output':
            params['regularize'] = False

        # Fine tuning the initializer:
        lin_initializer = self.initializer
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
        bias = self._cpu_variable_init('bias', bias_shape, initializer=tf.constant_initializer(1.e-3))
        output = tf.nn.bias_add(tf.matmul(input_reshape, weights), bias, name=name)

        # Keep tabs on the number of weights and memory
        self.num_weights += bias_shape[0] + np.cumprod(weights_shape)[-1]
        self.mem += np.cumprod(output.get_shape())[-1] * self.bytesize / 1024
        # equation = W * X + b
        # equation = [batch x features] * [features, nodes] + nodes
        # batch * nodes * 2 * features + nodes <- 2 comes in for the dot prod + sum
        this_ops = np.prod(input.get_shape().as_list() + [2, params['weights']]) + params['weights']
        if verbose:
            print('\t%3.2e ops' % (this_ops))
        self.ops += this_ops
        return output

    def _dropout(self, input=None, params=None, name=None):
        """
        Performs dropout
        :param input:
        :param params:
        :param name:
        :return:
        """
        # TODO: Avoid hardcoding hyperparameters like this.
        keep_prob = tf.constant(0.5)
        return tf.nn.dropout(input, keep_prob=keep_prob)

    def _activate(self, input=None, params=None, name=None, verbose=True):
        """
        Activation
        :param input: as it says
        :param params: dict, must specify activation type
        :param name: scope.name
        :return:
        """
        this_ops = 2 * np.prod(input.get_shape().as_list())
        if verbose:
            print('\tactivation = %3.2e ops' % (this_ops))
        self.ops += this_ops

        if params is not None:
            if params['activation'] == 'tanh':
                return tf.nn.tanh(input, name=name)

        return tf.nn.relu(input, name=name)

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
        self.mem += np.cumprod(output.get_shape())[-1] * self.bytesize / 1024

        # at each location in the image:
        # avg: 1 to sum each of the N element, 1 op for avg
        # max: N max() operations
        ops_per_pool = 1 * np.prod(params['kernel'] + [input.shape[1].value])
        # number of pools on the image for a single filter / output channel (stride brings down the number)
        num_pools = np.prod([input.shape[2].value, input.shape[3].value]) // np.prod(params['stride'])
        # final = num images * filters * convs/filter * ops/conv
        if verbose:
            print('\t%d ops/pool, %d pools = %3.2e ops' % (ops_per_pool, num_pools,
                                                           num_pools * ops_per_pool))

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
            # print('activation map shape: %s' %(format(map.shape)))
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
        # print('activation map shape: %s' %(format(map.shape)))
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
        if params['type'] == 'convolutional':
            print('%s --- output: %s, kernel: %s, stride: %s, # of weights: %s,  memory: %s MB' %
                  (scope.name, format(output_shape), format(params['kernel']),
                   format(params['stride']), format(self.num_weights), format(mem_in_MB)))
        if params['type'] == 'pool':
            print('%s --- output: %s, kernel: %s, stride: %s, memory: %s MB' %
                  (scope.name, format(output_shape), format(params['kernel']),
                   format(params['stride']), format(mem_in_MB)))
        if params['type'] == 'fully_connected' or params['type'] == 'linear_output':
            print('%s --- output: %s, weights: %s, bias: %s, # of weights: %s,  memory: %s MB' %
                   (scope.name, format(output_shape), format(params['weights']),
                     format(params['bias']), format(self.num_weights), format(mem_in_MB)))

    def _add_loss_summaries(self, total_loss, losses):
        """
        Add summaries for losses in model.
        Generates moving average for all losses and associated summaries for
        visualizing the performance of the network.
        :param flags:
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
    def _cpu_variable_init(self, name, shape, initializer, trainable=True, regularize=False):
        """Helper to create a Variable stored on CPU memory.

        Args:
          name: name of the variable
          shape: list of ints
          initializer: initializer for Variable

        Returns:
          Variable Tensor
        """
        with tf.device(self.flags.CPU_ID):
            if regularize:
                var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32, trainable=trainable,
                                      regularizer=self._weight_decay)
                return var

            var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32, trainable=trainable)
        return var

    def _weight_decay(self, tensor):
        return tf.multiply(tf.nn.l2_loss(tensor), self.hyper_params['weight_decay'])


class ResNet(ConvNet):

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
                print('Doing 1x1 conv on output to bring channels from %d to %d' % (out.get_shape().as_list()[1],
                                                                                    hidden.get_shape().as_list()[1]))
            with tf.variable_scope("_shortcut"):
                out, _ = self._conv(input=out, params=shortcut_parms)
        # ops just for the addition operation
        this_ops = np.prod(out.get_shape().as_list())
        if verbose:
            print('\tops for adding shortcut: %3.2e' % (this_ops))
        self.ops += this_ops
        # Now add the hidden with input
        return tf.add(out, hidden)

    def _compound_layer(self, out, layer_params, scope_name):
        if layer_params['type'] == 'residual':
            return self._residual_block(out, layer_params, scope_name)

    def _residual_block(self, out, res_block_params, scope_name, verbose=True):
        """
        Unit residual block consisting of arbitrary number of convolutional layers, each specified by its own
        OrderedDictionary in the parameters.
        Implementation here based on: https://arxiv.org/pdf/1603.05027.pdf
        Input >> BN >> Relu >> weight >> BN >> ReLU >> Weight >> Add Input

        :param out: 4D tensor, Input to the residual block
        :param res_block_params: OrderedDictionary, Parameters for the residual block
        :param scope_name: str, name of the layer
        :param verbose: bool, (Optional) - whether or not to print statements.
        :return: 4D tensor, output of the residual block
        """
        ops_in = self.ops

        with tf.variable_scope("_pre_conv1"):
            hidden = self._batch_norm(input=out)
        hidden = self._activate(input=hidden, name=scope_name + '_pre_conv1')

        # First find the names of all conv layers inside
        layer_ids = []
        for parm_name in res_block_params.keys():
            if isinstance(res_block_params[parm_name], OrderedDict):
                if res_block_params[parm_name]['type'] == 'convolutional':
                    layer_ids.append(parm_name)
        """
        if verbose:
            print('internal layers:' + str(layer_ids))
            print('Working on the first N-1 layers')
        """
        # Up to N-1th layer: weight >> BN >> ReLU
        for layer_name in layer_ids[:-1]:
            if verbose:
                print('weight >> BN >> ReLU on layer: ' + layer_name)
            with tf.variable_scope(layer_name):
                layer_params = res_block_params[layer_name]
                hidden, _ = self._conv(input=hidden, params=layer_params)
                hidden = self._batch_norm(input=hidden)
            hidden = self._activate(input=hidden, name=scope_name + '_' + layer_name, params=layer_params)

        if verbose:
            print('weight ONLY on layer: ' + layer_ids[-1])
        # last layer: Weight ONLY
        with tf.variable_scope(layer_ids[-1]):
            hidden, _ = self._conv(input=hidden, params=res_block_params[layer_ids[-1]])

        # Now add the two branches
        ret_val = self._add_branches(hidden, out)

        if verbose:
            ops_out = self.ops
            print('\tresnet ops = %3.2e' % (ops_out - ops_in))

        return ret_val

    def _print_layer_specs(self, params, scope, input_shape, output_shape):
        if params['type'] == 'pooling':
            print('%s --- output: %s, kernel: %s, stride: %s' %
                  (scope.name, format(output_shape), format(params['kernel']),
                   format(params['stride'])))
        elif params['type'] == 'residual':
            mem_in_MB = np.cumprod(output_shape)[-1] * self.bytesize / 1024**2
            print('Residual Layer: ' + scope.name)
            for parm_name in params.keys():
                if isinstance(params[parm_name], OrderedDict):
                    if params[parm_name]['type'] == 'convolutional':
                        conv_parms = params[parm_name]
                        print('\t%s --- output: %s, kernel: %s, stride: %s, # of weights: %s,  memory: %s MB' %
                              (parm_name, format(output_shape), format(conv_parms['kernel']),
                               format(conv_parms['stride']), format(self.num_weights), format(0)))

        else:
            super(ResNet, self)._print_layer_specs(params, scope, input_shape, output_shape)
