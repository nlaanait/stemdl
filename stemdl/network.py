"""
Created on 10/8/17.
@author: Numan Laanait.
email: laanaitn@ornl.gov
"""

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
        self.mem = np.cumprod(self.images.get_shape())[-1]*self.bytesize/1024 #(in KB)

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
            out = self._batch_norm(input=out)
            out = self._activate(input=out, name=scope.name, params=layer_params)
            in_shape = self.images.get_shape()
            # Tensorboard Summaries
            if self.summary:
                self._activation_summary(out)
                self._activation_image_summary(out)
                self._kernel_image_summary(kernel)
                # def none():
                #     pass
                # tf.cond(self.global_step < 2 * self.flags.save_frequency, true_fn=self._json_summary,
                #         false_fn=lambda: None)
            self._print_layer_specs(layer_params, scope, in_shape, out.get_shape())

        # Initiate the remaining layers
        for layer_name, layer_params in list(self.network.items())[1:]:
            with tf.variable_scope(layer_name, reuse=self.reuse) as scope:
                in_shape = out.get_shape()
                if layer_params['type'] == 'convolutional':
                    out, _ = self._conv(input=out, params=layer_params)
                    out = self._batch_norm(input=out)
                    out = self._activate(input=out, name=scope.name, params=layer_params)
                    if self.summary: self._activation_summary(out)

                if layer_params['type'] == 'pooling':
                    out = self._pool(input=out, name=scope.name, params=layer_params)

                if layer_params['type'] == 'fully_connected':
                    out = self._linear(input=out, name=scope.name+'_preactiv', params=layer_params)
                    out = self._activate(input=out, name=scope.name, params=layer_params)
                    if self.summary: self._activation_summary(out)

                if layer_params['type'] == 'linear_output':
                    in_shape = out.get_shape()
                    out = self._linear(input=out, name=scope.name, params=layer_params)
                    assert out.get_shape()[-1] == self.flags.OUTPUT_DIM, 'Dimensions of the linear output layer' + \
                                                                         'do not match the expected output set in' + \
                                                                         'tf.app.flags. Check flags or network_config.json'
                    if self.summary: self._activation_summary(out)

                # print layer specs and generate Tensorboard summaries
                out_shape = out.get_shape()
                self._print_layer_specs(layer_params, scope, in_shape, out_shape)

        print('Total # of layers: %d,  weights: %2.1e, memory: %s MB \n' % (len(self.network), self.num_weights,
                                                                         format(self.mem/1024)))

        # reference the output
        self.model_output = out

    def get_loss(self):
        with tf.variable_scope(self.scope, reuse=self.reuse) as scope:
            if self.net_type == 'regressor':
                self._calculate_loss_regressor(self.hyper_params['loss_function'])
            if self.net_type == 'classifier':
                self._calculate_loss_classifier(self.hyper_params['loss_function'])
        # # Calculate total loss
        # losses = tf.get_collection(tf.GraphKeys.LOSSES, self.scope)
        # regularization = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        # total_loss = tf.add_n(losses+regularization)
        # # Moving average of loss and summaries
        # loss_ops = self._add_loss_summaries(total_loss,losses)
        # return total_loss, loss_ops

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
            # decay_steps = int(self.flags.NUM_EXAMPLES_PER_EPOCH / self.flags.batch_size \
            #                   * params['residual_num_epochs_decay'])
            # initial_residual = params['residual_initial']
            # min_residual = params['residual_minimum']
            # decay_residual = params['residual_decay_factor']
            # residual_tol = tf.train.exponential_decay(initial_residual, self.global_step, decay_steps,
            #                                           decay_residual,staircase=False)
            # # cap the residual cutoff to some min value.
            # residual_tol = tf.maximum(residual_tol, tf.constant(min_residual))
            # if self.summary:
            #     tf.summary.scalar('residual_cutoff', residual_tol)
            # # calculate the cost
            # cost = tf.losses.huber_loss(labels, predictions=self.model_output, delta=residual_tol,
            #                             reduction=tf.losses.Reduction.MEAN)
            cost = tf.losses.huber_loss(labels, predictions=self.model_output, delta=params['residual_initial'],
                                        reduction=tf.losses.Reduction.MEAN)
        if params['type'] == 'MSE':
            cost = tf.losses.mean_squared_error(labels, predictions=self.model_output,
                                                reduction=tf.losses.Reduction.MEAN)

        return cost

    def _calculate_loss_classifier(self,params):
        """
        Calculate the loss objective for classification
        :param params: dictionary, specifies the objective to use
        :return: cost
        """
        labels = tf.cast(self.labels, tf.int64)
        if params['type'] == 'exclusive':
            labels = tf.argmax(labels, axis=1)
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=self.model_output)
            cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
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
        init_val = np.sqrt(2.0/(kernel_shape[0] * kernel_shape[1] * features))
        kernel = self._cpu_variable_init('weights', shape=kernel_shape,
                                         initializer=tf.truncated_normal_initializer(stddev=init_val))
        output = tf.nn.conv2d(input, kernel, stride_shape, data_format='NCHW', padding=params['padding'])

        # Keep tabs on the number of weights and memory
        self.num_weights += np.cumprod(kernel_shape)[-1]
        self.mem += np.cumprod(output.get_shape())[-1]*self.bytesize / 1024

        return output, kernel

    def _batch_norm(self, input=None):
        """
        Batch normalization
        :param input: as it says
        :return:
        """
        # Initializing hyper_parameters
        shape = [input.shape[1].value]
        beta = self._cpu_variable_init('beta', shape=shape, initializer=tf.zeros_initializer())
        gamma = self._cpu_variable_init('gamma', shape=shape,initializer=tf.ones_initializer())
        if self.operation == 'train':
            output, mean, variance = tf.nn.fused_batch_norm(input, gamma, beta, None, None, 1.e-3, data_format='NCHW',
                                                            is_training=True)
            moving_mean = self._cpu_variable_init('moving_mean', shape=shape,
                                                  initializer=tf.zeros_initializer(), trainable=False)
            moving_variance = self._cpu_variable_init('moving_variance', shape=shape,
                                                      initializer=tf.ones_initializer(), trainable=False)
            self.misc_ops.append(moving_averages.assign_moving_average(
                moving_mean, mean, 0.9))
            self.misc_ops.append(moving_averages.assign_moving_average(
                moving_variance, variance, 0.9))
        if self.operation == 'eval':
            mean = self._cpu_variable_init('moving_mean', shape=shape, \
                                           initializer=tf.zeros_initializer(), trainable=False)
            variance = self._cpu_variable_init('moving_variance', shape=shape, \
                                               initializer=tf.ones_initializer(), trainable=False)
            output, _, _ = tf.nn.fused_batch_norm(input, gamma, beta, mean, variance, epsilon=1.e-3, data_format='NCHW',
                                            is_training=False)
        # Keep tabs on the number of weights
        self.num_weights += beta.shape[0].value + gamma.shape[0].value
        return output

    def _linear(self, input=None, params=None, name=None):
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
        if params['type'] == 'fully_connected' and params['activation'] == 'tanh':
            weights = self._cpu_variable_init('weights', shape=weights_shape,
                                              initializer=tf.uniform_unit_scaling_initializer(factor=1.15),
                                              regularize=params['regularize'])
        if params['type'] == 'fully_connected' and params['activation'] == 'relu':
            weights = self._cpu_variable_init('weights', shape=weights_shape,
                                              initializer=tf.uniform_unit_scaling_initializer(factor=1.43),
                                              regularize=params['regularize'])
        if params['type'] == 'linear_output':
            weights = self._cpu_variable_init('weights', shape=weights_shape,
                                              initializer=tf.uniform_unit_scaling_initializer(factor=1.0))

        bias = self._cpu_variable_init('bias', bias_shape, initializer=tf.constant_initializer(0.0))
        output = tf.nn.bias_add(tf.matmul(input_reshape, weights), bias, name=name)

        # Keep tabs on the number of weights and memory
        self.num_weights += bias_shape[0] + np.cumprod(weights_shape)[-1]
        self.mem += np.cumprod(output.get_shape())[-1] * self.bytesize / 1024
        return output

    @staticmethod
    def _activate(input=None, params=None, name=None):
        """
        Activation
        :param input: as it says
        :param params: dict, must specify activation type
        :param name: scope.name
        :return:
        """
        if params['activation'] == 'tanh':
            return tf.nn.tanh(input, name=name)

        return tf.nn.relu(input, name=name)

    def _pool(self, input=None, params=None, name=None):
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
        with tf.device('/cpu:0'):
            # var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32, trainable=trainable)
            if regularize:
                var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32, trainable=trainable,
                                      regularizer=self._weight_decay)
                return var
                # weight_decay = tf.get_variable(name='weight_decay', shape=[1], initializer=tf.ones_initializer(),
                #                                trainable=False) * self.hyper_params['weight_decay']
                # weight_loss = tf.multiply(tf.nn.l2_loss(var), weight_decay, name='weight_loss')
                # tf.add_to_collection('losses', weight_loss)
            var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32, trainable=trainable)
        return var

    def _weight_decay(self, tensor):
        return tf.multiply(tf.nn.l2_loss(tensor), self.hyper_params['weight_decay'])

# TODO: implement ResNet
# TODO: ResNet should inherit Net
class ResNet(object):
    pass


