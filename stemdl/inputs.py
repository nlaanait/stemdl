"""
Created on 10/8/17.
@author: Numan Laanait.
email: laanaitn@ornl.gov
"""

import tensorflow as tf
import numpy as np
import os
from multiprocessing import cpu_count
from tensorflow.python.ops import data_flow_ops


class DatasetTFRecords(object):
    """
    Handles training and evaluation data operations.  \n
    Data is read from a TFRecords filename queue.
    """

    def __init__(self, filename_queue, params, is_train=True, num_gpus=1, train_cpu_frac=1,
                 max_cpu_utilization=0.9, using_horovod=False):
        self.is_train = is_train
        self.filename_queue = filename_queue
        self.params = params

        max_cpu_utilization = max(0, min(1, max_cpu_utilization))
        self.max_threads = int(max_cpu_utilization * cpu_count())
        self.train_cpu_frac = max(0, min(1, train_cpu_frac))
        if using_horovod:
            self.max_threads = self.max_threads // num_gpus

        self.max_threads = max(1, self.max_threads)

        if self.train_cpu_frac == 1:
            print('WARNING: All threads devoted to training, cannot use any for evaluation')

        print('***************************************************')
        print('\t\tUsing DatasetTFRecords')
        print('Original parameters:')
        print('Horvod: {}, GPUs: {}, Max CPU utilization: {}, Training fraction allowed: {}'.format(using_horovod, num_gpus, max_cpu_utilization,
                                                                                        self.train_cpu_frac))
        print('CPU Threads: available: {}, allowed: {}'.format(int(cpu_count()), self.max_threads))

        print('***************************************************')

    def _decode_image_label(self):
        """
        Returns: image, label decoded from tfrecords
        """
        reader = tf.TFRecordReader()
        key, serialized_example  = reader.read(self.filename_queue)

        # get raw image bytes and label string
        features = tf.parse_single_example(
            serialized_example,
            features={
                'image_raw': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.string),
            })
        # decode from byte and reshape label and image
        label_dtype = tf.as_dtype(self.params['LABEL_DTYPE'])
        label = tf.decode_raw(features['label'], label_dtype)
        label.set_shape(self.params['NUM_CLASSES'])
        image = tf.decode_raw(features['image_raw'], tf.float16)
        image.set_shape([self.params['IMAGE_HEIGHT'] * self.params['IMAGE_WIDTH'] * self.params['IMAGE_DEPTH']])
        image = tf.reshape(image, [self.params['IMAGE_HEIGHT'], self.params['IMAGE_WIDTH'], self.params['IMAGE_DEPTH']])
        # standardize the image to [-1.,1.]
        image = tf.image.per_image_standardization(image)
        return image, label

    def _distort(self, image, noise_min, noise_max, geometric=False):
        """
        Performs distortions on an image: noise + global affine transformations.
        Args:
            image: 3D Tensor
            noise_min:float, lower limit in (0,1)
            noise_max:float, upper limit in (0,1)
            geometric: bool, apply affine distortion

        Returns:
            distorted_image: 3D Tensor
        """

        # Apply random global affine transformations
        # if geometric:

        # 1. Apply random global affine transformations, sampled from a normal distributions.
        # Setting bounds and generating random values for scaling and rotations
        scale_X = np.random.normal(1.0, 0.05, size=1)
        scale_Y = np.random.normal(1.0, 0.05, size=1)
        theta_angle = np.random.normal(0., 1, size=1)
        nu_angle = np.random.normal(0., 1, size=1)

        # Constructing transfomation matrix
        a_0 = scale_X * np.cos(np.deg2rad(theta_angle))
        a_1 = -scale_Y * np.sin(np.deg2rad(theta_angle + nu_angle))
        a_2 = 0.
        b_0 = scale_X * np.sin(np.deg2rad(theta_angle))
        b_1 = scale_Y * np.cos(np.deg2rad(theta_angle + nu_angle))
        b_2 = 0.
        c_0 = 0.
        c_1 = 0.
        affine_transform = tf.constant(np.array([a_0, a_1, a_2, b_0, b_1, b_2, c_0, c_1]).flatten(),
                                       dtype=tf.float32)
        # Transform
        aff_image = tf.contrib.image.transform(image, affine_transform,
                                           interpolation='BILINEAR')
        # 2. Apply isotropic scaling, sampled from a normal distribution.
        zoom_factor = np.random.normal(1.0, 0.05, size=1)
        crop_y_size, crop_x_size = self.params['IMAGE_HEIGHT'], self.params['IMAGE_WIDTH']
        size = tf.constant(value=[int(np.round(crop_y_size / zoom_factor)),
                                  int(np.round(crop_x_size / zoom_factor))], dtype=tf.int32)
        cen_y = np.ones((1,), dtype=np.float32) * int(self.params['IMAGE_HEIGHT'] / 2)
        cen_x = np.ones((1,), dtype=np.float32) * int(self.params['IMAGE_WIDTH'] / 2)
        offsets = tf.stack([cen_y, cen_x], axis=1)
        scaled_image = tf.expand_dims(aff_image, axis=0)
        scaled_image = tf.image.extract_glimpse(scaled_image, size, offsets,
                                         centered=False,
                                         normalized=False,
                                         uniform_noise=False)
        scaled_image = tf.reshape(scaled_image, (scaled_image.shape[1].value, scaled_image.shape[2].value,
                                                 scaled_image.shape[3].value))
        scaled_image = tf.image.resize_images(scaled_image, (self.params['IMAGE_HEIGHT'], self.params['IMAGE_WIDTH']))

        # Apply noise
        alpha = tf.random_uniform([1], minval=noise_min, maxval=noise_max)
        noise = tf.random_uniform(scaled_image.shape, dtype=tf.float32)
        trans_image = (1 - alpha[0]) * scaled_image + alpha[0] * noise

        return trans_image

    def _get_glimpses(self, batch_images, random=False):
        """
        Get bounded glimpses from images, corresponding to ~ 2x1 supercell
        :param batch_images: batch of training images
        :return: batch of glimpses
        """
        # set size of glimpses
        y_size, x_size = self.params['IMAGE_HEIGHT'], self.params['IMAGE_WIDTH']
        crop_y_size, crop_x_size = self.params['CROP_HEIGHT'], self.params['CROP_WIDTH']
        size = tf.constant(value=[crop_y_size, crop_x_size],
                           dtype=tf.int32)

        if random is 'uniform':
            # generate uniform random window centers for the batch with overlap with input
            y_low, y_high = int(crop_y_size / 2), int(y_size - crop_y_size / 2)
            x_low, x_high = int(crop_x_size / 2), int(x_size - crop_x_size / 2)
            cen_y = tf.random_uniform([self.params['batch_size']], minval=y_low, maxval=y_high)
            cen_x = tf.random_uniform([self.params['batch_size']], minval=x_low, maxval=x_high)
            offsets = tf.stack([cen_y, cen_x], axis=1)

        if random is 'normal':
            # generate normal random window centers for the batch with overlap with input
            cen_y = tf.random_normal([self.params['batch_size']], mean=y_size / 2, stddev=4.)
            cen_x = tf.random_normal([self.params['batch_size']], mean=x_size / 2, stddev=4.)
            offsets = tf.stack([cen_y, cen_x], axis=1)

        if not random:
            # fixed crop
            cen_y = np.ones((self.params['batch_size'],), dtype=np.int32) * 38
            cen_x = np.ones((self.params['batch_size'],), dtype=np.int32) * 70
            offsets = np.vstack([cen_y, cen_x]).T
            offsets = tf.constant(value=offsets, dtype=tf.float32)

        # extract glimpses
        glimpse_batch = tf.image.extract_glimpse(batch_images, size, offsets, centered=False,
                                                 normalized=False,
                                                 uniform_noise=False,
                                                 name='batch_glimpses')
        return glimpse_batch

    def get_batch(self, distort=False, noise_min=0., noise_max=0.3, random_glimpses=True, geometric=False):

        image_raw, label = self._decode_image_label()

        if distort:
            # we first cast to float32 to do some image operations
            image = tf.cast(image_raw, tf.float32)
            # Apply image distortions
            image = self._distort(image, noise_min, noise_max, geometric=geometric)
        else:
            image = image_raw

        # Generate batch

        cpu_frac = self.train_cpu_frac
        min_after_dequeue = 5000
        tensor_name = 'Train_Images'
        if not self.is_train:
            cpu_frac = 1 - cpu_frac
            min_after_dequeue = 1000
            tensor_name = 'Test_Images'

        # TODO: Need to change num_threads so that it's determined from horovod total_rank
        num_threads = int(self.max_threads * cpu_frac)
        capacity = max(int(min_after_dequeue * 1.1),
                       int(self.params['NUM_EXAMPLES_PER_EPOCH'] * 0.1))

        images, labels = tf.train.shuffle_batch([image, label],
                                                batch_size=self.params['batch_size'],
                                                capacity=capacity,
                                                num_threads=num_threads,
                                                min_after_dequeue=min_after_dequeue,
                                                name='shuffle_batch')

        # Extract glimpses from training batch
        if random_glimpses is not None:
            images = self._get_glimpses(images, random=random_glimpses)

        # Display the training images in the Tensorboard visualizer.
        tf.summary.image(tensor_name, images, max_outputs=1)

        # resize images using the new params
        if self.params['RESIZE_HEIGHT'] != self.params['IMAGE_HEIGHT'] or \
                self.params['RESIZE_WIDTH'] != self.params['IMAGE_WIDTH']:
            images = tf.image.resize_images(images, [self.params['RESIZE_HEIGHT'], self.params['RESIZE_WIDTH']])

        # change from NHWC to NCHW format
        images = tf.transpose(images, perm=[0, 3, 1, 2])

        # if running with half-precision we cast back to float16
        if self.params['IMAGE_FP16']:
            images = tf.cast(images, tf.float16)

        return images, labels


def decode_image_label(record, params, hyper_params):
    """
    Returns: image, label decoded from tfrecords
    """
    # get raw image bytes and label string
    features = tf.parse_single_example(
        record,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.string),
        })
    # decode from byte and reshape label and image
    label_dtype = tf.as_dtype(params['LABEL_DTYPE'])
    label = tf.decode_raw(features['label'], label_dtype)
    label.set_shape(params['NUM_CLASSES'])
    image = tf.decode_raw(features['image_raw'], tf.float16)
    image.set_shape([params['IMAGE_HEIGHT'] * params['IMAGE_WIDTH'] * params['IMAGE_DEPTH']])
    image = tf.reshape(image, [params['IMAGE_HEIGHT'], params['IMAGE_WIDTH'], params['IMAGE_DEPTH']])
    # standardize the image to [-1.,1.]
    image = tf.image.per_image_standardization(image)

    # Manipulate labels
    # turn into 1-hot vector for classification. So that we don't modify the data.
    if hyper_params['network_type'] == 'classifier' and label_dtype == tf.float64:
        label = onehot(label)
    elif hyper_params['network_type'] == 'regressor':
        # scale labels for regression
        # TODO: pull max and min values out of here and into input.json
        label = label_minmaxscaling(label, [20., 60., -3., -3.],
                                    [200., 200., 3., 3.], scale_range=[-10., 10.])

    return image, label


def label_minmaxscaling(label, min_vals, max_vals, scale_range=[0,1]):
    """

    :param label: tensor
    :param min_vals: list, minimum value for each label dimension
    :param max_vals: list, maximum value for each label dimension
    :param range: list, range of label, default [0,1]
    :return:
    scaled label tensor
    """
    min_tensor = tf.constant(min_vals, dtype=tf.float64)
    max_tensor = tf.constant(max_vals, dtype=tf.float64)
    scaled_label = (label - min_tensor)/(max_tensor - min_tensor)
    scaled_label = scaled_label * (scale_range[-1] - scale_range[0]) + scale_range[0]
    return scaled_label

def onehot(label):
    index = tf.cast(label[0],tf.int32)
    full_vec = tf.cast(tf.linspace(20., 200., 91),tf.int32)
    bool_vector = tf.equal(index, full_vec)
    onehot_vector = tf.cast(bool_vector, tf.int64)
    return onehot_vector

def minibatch(batchsize, params, hyper_params, mode='train'):
    """
    Returns minibatch of images and labels from TF records file.
    :param batchsize: int,
    :param params: dict, json input parameters
    :param mode: str, "train" or "test", default "train
    :return:
    """
    # if mode != 'train' or mode != 'test':
    #     mode = 'train'

    record_input = data_flow_ops.RecordInput(
        file_pattern=os.path.join(params['data_dir'], '*_%s.tfrecords' % mode),
        parallelism=params['IO_threads'],
        # Note: This causes deadlock during init if larger than dataset
        buffer_size=params['buffer_cap'],
        batch_size=batchsize)
    records = record_input.get_yield_op()
    # Split batch into individual images
    records = tf.split(records, batchsize, 0)
    records = [tf.reshape(record, []) for record in records]
    # Deserialize and preprocess images into batches for each device
    images = []
    labels = []
    with tf.name_scope('input_pipeline'):
        for i, record in enumerate(records):
            image, label = decode_image_label(record, params, hyper_params)
            images.append(image)
            labels.append(label)
        # Stack images back into a single tensor
        images = tf.parallel_stack(images)
        labels = tf.concat(labels, 0)
        labels = tf.reshape(labels, [-1, params['NUM_CLASSES']])
        images = tf.reshape(images, [-1, params['IMAGE_HEIGHT'], params['IMAGE_WIDTH'], params['IMAGE_DEPTH']])

        # Display the training images in the Tensorboard visualizer.
        #tf.summary.image("images", images, max_outputs=4)

        # change from NHWC to NCHW format
        images = tf.transpose(images, perm=[0, 3, 1, 2])

        # Cast to FP_16
        if params['IMAGE_FP16']:
            images = tf.cast(images, tf.float16)
        else:
            images = tf.cast(images, tf.float32)

    return images, labels


def stage(tensors):
    """
    Stages the given tensors in a StagingArea for asynchronous put/get
    :param tensors: tf.Tensor
    :return: get and put tf.Op operations.
    """
    stage_area = data_flow_ops.StagingArea(
        dtypes=[tensor.dtype       for tensor in tensors],
        shapes=[tensor.get_shape() for tensor in tensors])
    put_op = stage_area.put(tensors)
    get_tensors = stage_area.get()

    get_tensors = [tf.reshape(gt, t.get_shape())
                   for (gt,t) in zip(get_tensors, tensors)]
    return put_op, get_tensors