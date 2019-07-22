import tensorflow as tf
from .optimizers import get_regularization_loss
import numpy as np

def _add_loss_summaries(total_loss, losses, summaries=False):
    """
    Add summaries for losses in model.
    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.
    :param params:
    :param total_loss:
    :param losses:
    :return: loss_averages_op
    """
    # # Compute the moving average of all individual losses and the total loss.
    # loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    # loss_averages_op = loss_averages.apply(losses + [total_loss])
    # # loss_averages_op = loss_averages.apply([total_loss])

    # Attach a scalar summary to all individual losses and the total loss;
    if summaries:
        for l in losses + [total_loss]:
            tf.summary.scalar(l.op.name + '(raw)', l)
            #tf.summary.scalar(l.op.name, loss_averages.average(l))
    loss_averages_op = tf.no_op(name='no_op')
    return loss_averages_op


def calc_loss(n_net, scope, hyper_params, params, labels, images=None, summary=False):
    labels_shape = labels.get_shape().as_list()
    layer_params={'bias':labels_shape[-1], 'weights':labels_shape[-1],'regularize':True}
    if hyper_params['network_type'] == 'hybrid':
        dim = labels_shape[-1]
        num_classes = params['NUM_CLASSES']
        regress_labels, class_labels = tf.split(labels,[dim-num_classes, num_classes],1)
        if class_labels.dtype is not tf.int64:
            class_labels = tf.cast(class_labels, tf.int64)
        # Build output layer
        class_shape = class_labels.get_shape().as_list()
        regress_shape = regress_labels.get_shape().as_list()
        layer_params_class={'bias':class_shape[-1], 'weights':class_shape[-1],'regularize':True}
        layer_params_regress={'bias':regress_shape[-1], 'weights':regress_shape[-1],'regularize':True}
        output_class = fully_connected(n_net, layer_params_class, params['batch_size'],
                                name='linear', wd=hyper_params['weight_decay'])
        output_regress = fully_connected(n_net, layer_params_regress, params['batch_size'],
                                name='linear', wd=hyper_params['weight_decay'])
        # Calculate loss
        _ = calculate_loss_classifier(output_class, labels, params, hyper_params)
        _ = calculate_loss_regressor(output_regress, labels, params, hyper_params)
    if hyper_params['network_type'] == 'regressor':
        output = fully_connected(n_net, layer_params, params['batch_size'],
                                name='linear', wd=hyper_params['weight_decay'])
        _ = calculate_loss_regressor(output, labels, params, hyper_params)
    if hyper_params['network_type'] == 'inverter':
        #mask = np.ones((512,512), dtype=np.float32)
        #snapshot = slice(512// 4, 3 * 512//4)
        #mask[snapshot, snapshot] = 0.0 
        #mask = np.expand_dims(np.expand_dims(mask, axis=0), axis=0)
        #weight = tf.constant(mask)
        # if labels.shape != n_net.model_output:
        #     labels = tf.transpose(labels, perm=[0, 2, 3, 1])
        #     labels = tf.image.resize(labels, n_net.model_output.shape.as_list()[-2:], method=tf.image.ResizeMethod.BILINEAR)
        #     labels = tf.transpose(labels, perm=[0, 3, 1, 2])
        weight=None
        _ = calculate_loss_regressor(n_net.model_output, labels, params, hyper_params, weight=weight)
    if hyper_params['network_type'] == 'fft_inverter':
        n_net.model_output = tf.exp(1.j * tf.cast(n_net.model_output, tf.complex64))
        psi_pos = tf.ones([1,16,512,512], dtype=tf.complex64)
        n_net.model_output = tf.multiply(psi_pos, n_net.model_output)
        n_net.model_output = tf.fft2d(n_net.model_output / np.prod(np.array(n_net.model_output.get_shape().as_list())[2:]))
        n_net.model_output = tf.abs(n_net.model_output)
        n_net.model_output = tf.cast(n_net.model_output, tf.float16)
        n_net.model_output = tf.reduce_mean(n_net.model_output, axis=[1], keepdims=True)
        _ = calculate_loss_regressor(n_net.model_output, labels, params, hyper_params)
    if hyper_params['network_type'] == 'YNet':
        weight=None
        probe_im = n_net.model_output['decoder']['IM']
        probe_re = n_net.model_output['decoder']['RE']
        pot = n_net.model_output['inverter']
        # probe_labels = tf.transpose(tf.reduce_mean(images, axis=1, keepdims=True), perm=[0,2,3,1])
        # probe_labels = tf.image.resize_bilinear(probe_labels, [128,128])
        # probe_labels = tf.transpose(probe_labels, perm=[0,3,1,2])
        # probe_arr = np.load('probe_amp.npy')
        probe_arr = np.load('psi_k.npy')
        probe_arr = np.expand_dims(np.expand_dims(probe_arr, axis=0), axis=0)
        probe_arr = np.tile(probe_arr, [4,1,1,1])
        probe_labels_re = tf.constant(np.abs(probe_arr), dtype=tf.float32)
        probe_labels_im = tf.constant(np.angle(probe_arr), dtype=tf.float32)
        pot_labels = labels
        # weight=10
        inverter_loss = calculate_loss_regressor(pot, pot_labels, params, hyper_params, weight=weight)
        # weight=1
        decoder_loss_im = calculate_loss_regressor(probe_im, probe_labels_im, params, hyper_params, weight=weight)
        decoder_loss_re = calculate_loss_regressor(probe_re, probe_labels_re, params, hyper_params, weight=weight)
        tf.summary.scalar('Inverter loss (raw)', inverter_loss)
        tf.summary.scalar('Decoder loss (IM)', decoder_loss_im)
        tf.summary.scalar('Decoder loss (RE)', decoder_loss_re)
    if hyper_params['network_type'] == 'classifier':
        if labels.dtype is not tf.int64:
            labels = tf.cast(labels, tf.int64)
        output = fully_connected(n_net, layer_params, params['batch_size'],
                                name='linear', wd=hyper_params['weight_decay'])
        _ = calculate_loss_classifier(output, labels, params, hyper_params)
    if hyper_params['langevin']:
        stochastic_labels = tf.random_normal(labels_shape, stddev=0.01, dtype=tf.float32)
        output = fully_connected(n_net, layer_params, params['batch_size'],
                            name='linear_stochastic', wd=hyper_params['weight_decay'])
        _ = calculate_loss_regressor(output, stochastic_labels, params, hyper_params, weight=hyper_params['mixing'])
   
    #Assemble all of the losses.
    losses = tf.get_collection(tf.GraphKeys.LOSSES)
    if hyper_params['network_type'] == 'YNet':
        losses = [inverter_loss , decoder_loss_im, decoder_loss_re ]
    # losses = [inverter_loss]
    regularization = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    # Calculate the total loss 
    total_loss = tf.add_n(losses + regularization, name='total_loss')
    # return tf.add_n(losses), None
    total_loss = tf.cast(total_loss, tf.float32)
    # Generate summaries for the losses and get corresponding op
    loss_averages_op = _add_loss_summaries(total_loss, losses, summaries=summary)
    return total_loss, loss_averages_op


def fully_connected(n_net, layer_params, batch_size, wd=0, name=None, reuse=None):
    input = tf.cast(tf.reshape(n_net.model_output,[batch_size, -1]), tf.float32)
    dim_input = input.shape[1].value
    weights_shape = [dim_input, layer_params['weights']]
    def weight_decay(tensor):
        return tf.multiply(tf.nn.l2_loss(tensor), wd)
    with tf.variable_scope(name, reuse=reuse) as output_scope:
        if layer_params['regularize']:
            weights = tf.get_variable('weights', weights_shape,
            initializer=tf.random_normal_initializer(0,0.01),
            regularizer=weight_decay)
            bias = tf.get_variable('bias', layer_params['bias'], initializer=tf.constant_initializer(1.e-3),
            regularizer=weight_decay)
        else:
            weights = tf.get_variable('weights', weights_shape,
            initializer=tf.random_normal_initializer(0,0.01))
            bias = tf.get_variable('bias', layer_params['bias'], initializer=tf.constant_initializer(1.e-3))
        output = tf.nn.bias_add(tf.matmul(input, weights), bias, name=name)
    # Add output layer to neural net scopes for layerwise optimization
    n_net.scopes.append(output_scope)
    return output


def calculate_loss_classifier(net_output, labels, params, hyper_params, summary=False):
    """
    Calculate the loss objective for classification
    :param params: dictionary, specifies the objective to use
    :return: cost
    """
    labels = tf.argmax(labels, axis=1)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=net_output)
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    precision_1 = tf.scalar_mul(1. / params['batch_size'],
                                tf.reduce_sum(tf.cast(tf.nn.in_top_k(net_output, labels, 1), tf.float32)))
    precision_5 = tf.scalar_mul(1. / params['batch_size'],
                                tf.reduce_sum(tf.cast(tf.nn.in_top_k(net_output, labels, 5), tf.float32)))
    if summary :
        tf.summary.scalar('precision@1_train', precision_1)
        tf.summary.scalar('precision@5_train', precision_5)
    tf.add_to_collection(tf.GraphKeys.LOSSES, cross_entropy_mean)
    return cross_entropy_mean


def calculate_loss_regressor(net_output, labels, params, hyper_params, weight=None, summary=False, global_step=None):
    """
    Calculate the loss objective for regression
    :param params: dictionary, specifies the objective to use
    :return: cost
    """
    # weight = 1./ hyper_params.get('scaling', 1)
    if weight is None:
        weight = 1.0
    if global_step is None:
        global_step = 1
    loss_params = hyper_params['loss_function']
    assert loss_params['type'] == 'Huber' or loss_params['type'] == 'MSE' \
    or loss_params['type'] == 'LOG' or loss_params['type'] == 'MSE_PAIR' or loss_params['type'] == 'ABS_DIFF' or loss_params['type'] == 'ABS_DIFF_SCALED', "Type of regression loss function must be 'Huber' or 'MSE'"
    if loss_params['type'] == 'Huber':
        # decay the residual cutoff exponentially
        decay_steps = int(params['NUM_EXAMPLES_PER_EPOCH'] / params['batch_size'] \
                          * loss_params['residual_num_epochs_decay'])
        initial_residual = loss_params['residual_initial']
        min_residual = loss_params['residual_minimum']
        decay_residual = loss_params['residual_decay_factor']
        residual_tol = tf.train.exponential_decay(initial_residual, global_step, decay_steps,
                                                  decay_residual, staircase=False)
        # cap the residual cutoff to some min value.
        residual_tol = tf.maximum(residual_tol, tf.constant(min_residual))
        if summary:
            tf.summary.scalar('residual_cutoff', residual_tol)
        # calculate the cost
        cost = tf.losses.huber_loss(labels, weights=weight, predictions=net_output, delta=residual_tol,
                                    reduction=tf.losses.Reduction.MEAN)
    if loss_params['type'] == 'MSE':
        cost = tf.losses.mean_squared_error(labels, weights=weight, predictions=net_output,
                                            reduction=tf.losses.Reduction.MEAN)
    if loss_params['type'] == 'ABS_DIFF':
        cost = tf.losses.absolute_difference(labels, weights=weight, predictions=net_output,
                                            reduction=tf.losses.Reduction.MEAN)
    #if loss_params['type'] == 'ABS_DIFF_SCALED':
    #    weight= 1./512.
    #    cost = tf.losses.absolute_difference(labels, weights=weight, predictions=net_output,
                                            #reduction=tf.losses.Reduction.SUM)
    if loss_params['type'] == 'MSE_PAIR':
        cost = tf.losses.mean_pairwise_squared_error(labels, net_output, weights=weight)
    if loss_params['type'] == 'LOG':
        cost = tf.losses.log_loss(labels, weights=weight, predictions=net_output, reduction=tf.losses.Reduction.MEAN)
    return cost
