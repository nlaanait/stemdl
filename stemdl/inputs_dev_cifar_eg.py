"""
Created on 10/8/17.
@author: Suhas Somnath
email: syz@ornl.gov
"""

import tensorflow as tf
import numpy as np
from multiprocessing import cpu_count


class CifarEgReader(object):
    """
    Based on:
    https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10_estimator/cifar10.py
    called by:
    https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10_estimator/cifar10_main.py

    For whatever reason the labels shape is not correctly set in the the graph.
    The images show up correctly as [batch_size, 120, 80, 1] while labels show up as [?, 27] or something.
    Images go through some sort of a
    """

    def __init__(self, file_path, flags, mode='train', num_gpus=1, max_cpu_utilization=0.9, train_cpu_frac=1,
                 using_horovod=False):
        self.file_path = file_path
        self.flags = flags
        self.mode = mode

        max_cpu_utilization = max(0, min(1, max_cpu_utilization))
        self.max_threads = int(max_cpu_utilization * cpu_count())
        self.train_cpu_frac = max(0, min(1, train_cpu_frac))
        if self.train_cpu_frac == 1:
            print('WARNING: All threads devoted to training, cannot use any for evaluation')

        print('***************************************************')
        print('\t\tUsing CifarEgReader')
        print('Original parameters:')
        print('GPUs: {}, Max CPU utilization: {}, Training fraction allowed: {}'.format(num_gpus, max_cpu_utilization,
                                                                                        self.train_cpu_frac))

        if using_horovod:
            self.max_threads = self.max_threads // num_gpus
            num_gpus = 1

        print('CPU Threads: available: {}, allowed: {}'.format(int(cpu_count()), self.max_threads))

        self.num_gpus = num_gpus
        self.batch_size = num_gpus * self.flags.batch_size

        print('Horvod: {}, GPUs (re)set to: {}, batch size scaled from {} to {}'.format(
            using_horovod, self.num_gpus, self.flags.batch_size, self.batch_size))
        print('***************************************************')

    def _parser(self, serialized_example):

        features = tf.parse_single_example(
            serialized_example,
            features={
                'image_raw': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.string),
            })
        # decode from byte and reshape label and image
        label_dtype = tf.as_dtype(self.flags.LABEL_DTYPE)
        label = tf.decode_raw(features['label'], label_dtype)
        label.set_shape(self.flags.NUM_CLASSES)

        image = tf.decode_raw(features['image_raw'], tf.float16)
        image.set_shape([self.flags.IMAGE_HEIGHT * self.flags.IMAGE_WIDTH * self.flags.IMAGE_DEPTH])
        image = tf.reshape(image, [self.flags.IMAGE_HEIGHT, self.flags.IMAGE_WIDTH, self.flags.IMAGE_DEPTH])
        # standardize the image to [-1.,1.]
        image = tf.image.per_image_standardization(image)

        # Now continue what was there in train_images_labels prior to batch creation
        image = self._preprocess(image)

        return image, label

    def _preprocess(self, image):
        if self.distort:
            # we first cast to float32 to do some image operations
            image = tf.cast(image, tf.float32)
            # Apply image distortions
            image = self._distort(image, self.noise_min, self.noise_max, geometric=self.geometric)

        # if running with half-precision we cast to float16
        if self.flags.IMAGE_FP16:
            if self.mode == 'train':
                image = tf.cast(image, tf.float16)
            else:
                image = tf.image.convert_image_dtype(image, tf.float16)
        else:
            image = tf.cast(image, tf.float32)

        return image

    def _make_batch(self, distort=False, noise_min=0., noise_max=0.3,
                    random_glimpses='normal', geometric=False):
        """Read the images and labels from 'filenames'."""
        self.distort = distort
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.geometric = geometric

        tf_version = float(tf.__version__[:3])

        if tf_version < 1:
            raise NotImplementedError('Will not work on TF versions < 1')

        filenames = [self.file_path]

        # Repeat infinitely.
        if tf_version > 1.3:
            dataset = tf.data.TFRecordDataset(filenames).repeat()
        else:
            """
            TFRecordDataset.__init__ (from tensorflow.contrib.data.python.ops.readers) is deprecated and will be removed in
            a future version.
            Instructions for updating: Use `tf.data.TFRecordDataset`.
            """
            dataset = tf.contrib.data.TFRecordDataset(filenames).repeat()

        if self.mode == 'train':
            num_threads = int(self.max_threads * self.train_cpu_frac)
        else:
            num_threads = int(self.max_threads * (1-self.train_cpu_frac))

        num_threads = max(1, num_threads)

        # Parse records.
        if tf_version > 1.3:
            dataset.prefetch(2 * self.batch_size)
            dataset = dataset.map(self._parser, num_parallel_calls=num_threads)
        else:
            """
            calling Dataset.map (from tensorflow.contrib.data.python.ops.dataset_ops) with output_buffer_size is deprecated
            and will be removed in a future version.

            num_threads is deprecated and will be removed in a future version.

            Instructions for updating:
            Replace `num_threads=T` with `num_parallel_calls=T`. Replace `output_buffer_size=N` with `ds.prefetch(N)` on
            the returned dataset.
            """
            dataset = dataset.map(self._parser, num_threads=num_threads, output_buffer_size=2 * self.batch_size)

        # shuffle records.
        min_queue_examples = int(self.flags.NUM_EXAMPLES_PER_EPOCH * 0.1)
        # Ensure that the capacity is sufficiently large to provide good random shuffling.
        dataset = dataset.shuffle(buffer_size=min_queue_examples + 3 * self.batch_size)

        # Batch it up.
        dataset = dataset.batch(self.batch_size)
        iterator = dataset.make_one_shot_iterator()
        images, labels = iterator.get_next()

        # Extract glimpses from batch
        # print(images.get_shape().as_list(), labels.get_shape().as_list())
        # at this point the first axis of glimpses becomes batch_size while labels remains None!
        if random_glimpses is not None:
            images = self._get_glimpses(images, random=random_glimpses)
        else:
            # the tf.image.extract_glimpse() is necessary to get the correct batch size.
            size = tf.constant(value=[self.flags.IMAGE_HEIGHT, self.flags.IMAGE_WIDTH],
                               dtype=tf.int32)
            offs = np.zeros((self.batch_size,), dtype=np.int32)
            offsets = np.vstack([offs, offs]).T
            offsets = tf.constant(value=offsets, dtype=tf.float32)
            images = tf.image.extract_glimpse(images, size, offsets, centered=False,
                                                     normalized=False, uniform_noise=False, name='batch_glimpses')
        # forcing the first dimension of the labels to be batch size via a reshape:
        labels = tf.reshape(labels, shape=(self.batch_size, self.flags.NUM_CLASSES))
        # print(images.get_shape().as_list(), labels.get_shape().as_list())

        # Display the images in the Tensorboard visualizer.
        tf.summary.image(self.mode + '_Images', images, max_outputs=1)

        # resize images using the new flags
        images = tf.image.resize_images(images, [self.flags.RESIZE_HEIGHT, self.flags.RESIZE_WIDTH])

        # change to NCHW format
        images = tf.transpose(images, perm=[0, 3, 1, 2])

        return images, labels

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
        crop_y_size, crop_x_size = self.flags.IMAGE_HEIGHT, self.flags.IMAGE_WIDTH
        size = tf.constant(value=[int(np.round(crop_y_size / zoom_factor)),
                                  int(np.round(crop_x_size / zoom_factor))], dtype=tf.int32)

        cen_y = np.ones((1,), dtype=np.float32) * int(self.flags.IMAGE_HEIGHT / 2)
        cen_x = np.ones((1,), dtype=np.float32) * int(self.flags.IMAGE_WIDTH / 2)
        offsets = tf.stack([cen_y, cen_x], axis=1)
        scaled_image = tf.expand_dims(aff_image, axis=0)
        scaled_image = tf.image.extract_glimpse(scaled_image, size, offsets, centered=False, normalized=False,
                                                uniform_noise=False)
        scaled_image = tf.reshape(scaled_image, (scaled_image.shape[1].value, scaled_image.shape[2].value,
                                                 scaled_image.shape[3].value))
        scaled_image = tf.image.resize_images(scaled_image, (self.flags.IMAGE_HEIGHT, self.flags.IMAGE_WIDTH))

        # Apply noise
        alpha = tf.random_uniform([1], minval=noise_min, maxval=noise_max)
        noise = tf.random_uniform(scaled_image.shape, dtype=tf.float32)
        trans_image = (1 - alpha[0]) * scaled_image + alpha[0] * noise

        return trans_image

    def _get_glimpses(self, batch_images, random='normal'):
        """
        Get bounded glimpses from images, corresponding to ~ 2x1 supercell
        :param batch_images: batch of training images
        :return: batch of glimpses
        """
        # set size of glimpses
        y_size, x_size = self.flags.IMAGE_HEIGHT, self.flags.IMAGE_WIDTH
        crop_y_size, crop_x_size = self.flags.CROP_HEIGHT, self.flags.CROP_WIDTH
        size = tf.constant(value=[crop_y_size, crop_x_size],
                           dtype=tf.int32)

        if random is 'uniform':
            # generate uniform random window centers for the batch with overlap with input
            y_low, y_high = int(crop_y_size / 2), int(y_size - crop_y_size / 2)
            x_low, x_high = int(crop_x_size / 2), int(x_size - crop_x_size / 2)
            cen_y = tf.random_uniform([self.batch_size], minval=y_low, maxval=y_high)
            cen_x = tf.random_uniform([self.batch_size], minval=x_low, maxval=x_high)
            offsets = tf.stack([cen_y, cen_x], axis=1)

        elif random is 'normal':
            # generate normal random window centers for the batch with overlap with input
            cen_y = tf.random_normal([self.batch_size], mean=y_size / 2, stddev=4.)
            cen_x = tf.random_normal([self.batch_size], mean=x_size / 2, stddev=4.)
            offsets = tf.stack([cen_y, cen_x], axis=1)

        else:
            # fixed crop
            cen_y = np.ones((self.batch_size,), dtype=np.int32) * 38
            cen_x = np.ones((self.batch_size,), dtype=np.int32) * 70
            offsets = np.vstack([cen_y, cen_x]).T
            offsets = tf.constant(value=offsets, dtype=tf.float32)

        # extract glimpses
        glimpse_batch = tf.image.extract_glimpse(batch_images, size, offsets, centered=False,
                                                 normalized=False, uniform_noise=False, name='batch_glimpses')
        return glimpse_batch

    def _split_among_gpus(self, image_batch, label_batch):
        """
        Distributes the provided images and labels tensors among the GPUs.

        Adapted from input_fn() in cifar10_main.py

        Args:
            image_batch : 4D tensor with images
            label_batch : 2D tensor with labels
        Returns:
            two lists of tensors for features and labels, each of num_shards length.
        """

        if self.num_gpus <= 1:
            # No GPU available or only 1 GPU.
            return [image_batch], [label_batch]

        # Note that passing num=batch_size is safe here, even though
        # dataset.batch(batch_size) can, in some cases, return fewer than batch_size
        # examples. This is because it does so only when repeating for a limited
        # number of epochs, but our dataset repeats forever.

        image_batch = tf.unstack(image_batch, num=self.batch_size, axis=0)
        label_batch = tf.unstack(label_batch, num=self.batch_size, axis=0)
        feature_shards = [[] for _ in range(self.num_gpus)]
        label_shards = [[] for _ in range(self.num_gpus)]
        for i in range(self.batch_size):
            idx = i % self.num_gpus
            # print(i, idx)
            feature_shards[idx].append(image_batch[i])
            label_shards[idx].append(label_batch[i])
        feature_shards = [tf.parallel_stack(x) for x in feature_shards]
        label_shards = [tf.parallel_stack(x) for x in label_shards]
        return feature_shards, label_shards

    def get_batch(self, distort=False, noise_min=0., noise_max=0.3, random_glimpses='normal', geometric=False):

        image_batch, label_batch = self._make_batch(distort=distort, noise_min=noise_min, noise_max=noise_max,
                                                    random_glimpses=random_glimpses, geometric=geometric)
        feature_shards, label_shards = self._split_among_gpus(image_batch, label_batch)
        return feature_shards, label_shards
