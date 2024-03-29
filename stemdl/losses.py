import tensorflow as tf
from .optimizers import get_regularization_loss
import numpy as np
from tensorflow.python.ops import manip_ops

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

def calc_loss(n_net, scope, hyper_params, params, labels, step=None, images=None, summary=False):
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
        if labels_shape[1] > 1:
            pot_labels, _, _ = [tf.expand_dims(itm, axis=1) for itm in tf.unstack(labels, axis=1)]
        else:
            pot_labels = labels
        weight=None
        _ = calculate_loss_regressor(n_net.model_output, pot_labels, params, hyper_params, weight=weight)
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
        probe_im = n_net.model_output['decoder_IM']
        probe_re = n_net.model_output['decoder_RE']
        pot = n_net.model_output['inverter']
        pot_labels, probe_labels_re, probe_labels_im = [tf.expand_dims(itm, axis=1) for itm in tf.unstack(labels, axis=1)]
        #weight= np.prod(pot_labels.shape.as_list()[-2:])
        weight=None
        inv_str = hyper_params.get('inv_strength', 1)
        reg_str = hyper_params.get('reg_strength', 0.01)
        dec_str = hyper_params.get('dec_strength', 1) 
        inverter_loss = inv_str * calculate_loss_regressor(pot, pot_labels, params, hyper_params, weight=weight)
        decoder_loss_im = dec_str * calculate_loss_regressor(probe_im, probe_labels_im, params, hyper_params, weight=weight)
        decoder_loss_re = dec_str * calculate_loss_regressor(probe_re, probe_labels_re, params, hyper_params, weight=weight)
        psi_out_mod = thin_object(probe_re, probe_im, pot, summarize=False)
        reg_loss = reg_str * calculate_loss_regressor(psi_out_mod, tf.reduce_mean(images, axis=[1], keepdims=True), 
                    params, hyper_params, weight=weight)
        tf.summary.scalar('reg_loss ', reg_loss)
        tf.summary.scalar('Inverter loss ', inverter_loss)
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
        losses = [inverter_loss , decoder_loss_re, decoder_loss_im, reg_loss]
        # losses, prefac = ynet_adjusted_losses(losses, step)
    regularization = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    # Calculate the total loss 
    total_loss = tf.add_n(losses + regularization, name='total_loss')
    # return tf.add_n(losses), None
    total_loss = tf.cast(total_loss, tf.float32)
    # Generate summaries for the losses and get corresponding op
    loss_averages_op = _add_loss_summaries(total_loss, losses, summaries=summary)
    if hyper_params['network_type'] == 'YNet':
        return total_loss, loss_averages_op, losses 
    return total_loss, loss_averages_op

def get_YNet_constraint(n_net, hyper_params, params, psi_out_true, weight=1):
    probe_im = tf.cast(n_net.model_output['decoder_IM'], tf.float32)
    probe_re = tf.cast(n_net.model_output['decoder_RE'], tf.float32)
    pot = tf.cast(n_net.model_output['inverter'], tf.float32)
    psi_out_mod = thin_object(probe_re, probe_im, pot)
    reg_loss = calculate_loss_regressor(psi_out_mod, tf.reduce_mean(psi_out_true, axis=[1], keepdims=True), 
                    params, hyper_params, weight=weight)
    reg_loss = tf.cast(reg_loss, tf.float32)
    regularization = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg_total_loss = tf.add_n([reg_loss] + regularization, name='total_loss')
    reg_totat_loss = tf.cast(reg_total_loss, tf.float32)
    return reg_total_loss

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
    if weight is None:
        weight = 1.0
    if global_step is None:
        global_step = 1
    loss_params = hyper_params['loss_function']
    assert loss_params['type'] == 'Huber' or loss_params['type'] == 'MSE' \
    or loss_params['type'] == 'LOG' or loss_params['type'] == 'MSE_PAIR' or loss_params['type'] == 'ABS_DIFF' or loss_params['type'] == 'ABS_DIFF_SCALED' or loss_params['type'] == 'rMSE', "Type of regression loss function must be 'Huber' or 'MSE'"
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
    if loss_params['type'] == 'MSE_PAIR':
        cost = tf.losses.mean_pairwise_squared_error(labels, net_output, weights=weight)
    if loss_params['type'] == 'rMSE':
        labels = tf.cast(labels, tf.float32)
        l2_true = tf.sqrt(tf.reduce_sum(labels ** 2, axis=[1,2,3]))
        l2_output = tf.sqrt(tf.reduce_sum(net_output **2, axis = [1,2,3]))
        cost = tf.reduce_mean(tf.abs(l2_true - l2_output)/l2_true)
        #cost = tf.reduce_mean(tf.sqrt(tf.reduce_sum((labels - net_output)**2, axis=[1,2,3]))/tf.sqrt(tf.reduce_sum(labels**2, axis=[1,2,3])))
        cost *= 100
        tf.add_to_collection(tf.GraphKeys.LOSSES, cost)
    if loss_params['type'] == 'LOG':
        cost = tf.losses.log_loss(labels, weights=weight, predictions=net_output, reduction=tf.losses.Reduction.MEAN)
    return cost

def ynet_adjusted_losses(losses, global_step):
    '''
    Schedule the different loss components based on global training step
    '''
    threshold = tf.constant(0.8)
    max_prefac = 1 
    ema = tf.train.ExponentialMovingAverage(0.9)
    loss_averages_op = ema.apply(losses)
    prefac_initial = 1e-5 

    with tf.control_dependencies([loss_averages_op]):
        def ramp():
            prefac = tf.cast(tf.train.exponential_decay(tf.constant(prefac_initial), global_step, 1000, 
                            tf.constant(0.1, dtype=tf.float32), staircase=False), tf.float32)
            prefac = tf.constant(prefac_initial) ** 2 * tf.pow(prefac, tf.constant(-1., dtype=tf.float32))
            prefac = tf.minimum(prefac, tf.cast(max_prefac, tf.float32))
            return prefac
    
        def decay(prefac_current):
            prefac = tf.train.exponential_decay(prefac_current, global_step, 1000, tf.constant(0.1, dtype=tf.float32),
                                            staircase=True)
            return prefac
        inv_loss, dec_re_loss, dec_im_loss = losses 

        prefac  = tf.cond(inv_loss > threshold, false_fn=ramp, true_fn=lambda: decay(prefac_initial))
        tf.summary.scalar("prefac_inverter", prefac)
        losses = [inv_loss , prefac * dec_re_loss, prefac * dec_im_loss]
        return losses, prefac

def fftshift(tensor, tens_format='NCHW'):
    dims = [2,3] if tens_format == 'NCHW' else [1,2]
    shift = [int((tensor.shape[dim]) // 2) for dim in dims]
    shift_tensor = manip_ops.roll(tensor, shift, dims)
    return shift_tensor

def thin_object(psi_k_re, psi_k_im, potential, summarize=True):
    # mask = np.zeros(psi_k_re.shape.as_list(), dtype=np.float32)
    # ratio = 0
    # if ratio == 0:
    #     center = slice(None, None) 
    # else:
    #     center = slice(int(ratio * mask.shape[-1]), int((1-ratio)* mask.shape[-1]))
    # mask[:,:,center,center] = 1.
    # mask = tf.constant(mask, dtype=tf.complex64)
    psi_x = fftshift(tf.ifft2d(tf.cast(psi_k_re, tf.complex64) * tf.exp( 1.j * tf.cast(psi_k_im, tf.complex64))))
    scan_range = psi_x.shape.as_list()[-1]//2
    vx, vy = np.linspace(-scan_range, scan_range, num=4), np.linspace(-scan_range, scan_range, num=4)
    X, Y = np.meshgrid(vx.astype(np.int), vy.astype(np.int))
    psi_x_stack = [tf.roll(psi_x, shift=[x,y], axis=[1,2]) for (x,y) in zip(X.flatten(), Y.flatten())]
    psi_x_stack = tf.concat(psi_x_stack, axis=1)
    pot_frac = tf.exp(1.j * tf.cast(potential, tf.complex64))
    psi_out = tf.fft2d(psi_x_stack * pot_frac / np.prod(psi_x.shape.as_list()))
    psi_out_mod = tf.cast(tf.abs(psi_out), tf.float32) ** 2
    psi_out_mod = tf.reduce_mean(psi_out_mod, axis=1, keep_dims=True)
    if summarize:
        tf.summary.image('Psi_k_out', tf.transpose(tf.abs(psi_out_mod)**0.25, perm=[0,2,3,1]), max_outputs=1)
        tf.summary.image('Psi_x_in', tf.transpose(tf.abs(psi_x)**0.25, perm=[0,2,3,1]), max_outputs=1)
    return psi_out_mod 
