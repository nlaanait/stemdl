"""
Created on 10/8/17.
@author: Numan Laanait.
email: laanaitn@ornl.gov
"""

import tensorflow as tf
import numpy as np


class DatasetTFRecords(object):
    """
    Handles training and evaluation data operations.  \n
    Data is read from a TFRecords filename queue.
    """

    def __init__(self, filename_queue, flags):
        self.filename_queue = filename_queue
        self.flags = flags

    def decode_image_label(self):
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
        label = tf.decode_raw(features['label'], tf.float64)
        label.set_shape(self.flags.NUM_CLASSES)
        image = tf.decode_raw(features['image_raw'], tf.float16)
        image.set_shape([self.flags.IMAGE_HEIGHT * self.flags.IMAGE_WIDTH * self.flags.IMAGE_DEPTH])
        image = tf.reshape(image, [self.flags.IMAGE_HEIGHT, self.flags.IMAGE_WIDTH, self.flags.IMAGE_DEPTH])
        return image, label

    def train_images_labels_batch(self, image_raw, label, distort=False, noise_min=0., noise_max=0.3,
                                  random_glimpses=True, geometric=False):
        """
        Returns: batch of images and labels to train on.
        """

        if distort:
            # we first cast to float32 to do some image operations
            image = tf.cast(image_raw, tf.float32)
            # Apply image distortions
            image = self._distort(image, noise_min, noise_max, geometric=geometric)
        else:
            image = image_raw

        # if running with half-precision we cast to float16
        if self.flags.IMAGE_FP16:
            image = tf.cast(image, tf.float16)
        else:
            image = tf.cast(image, tf.float32)

        # Generate batch
        images, labels = tf.train.shuffle_batch([image, label],
                                                batch_size=self.flags.batch_size,
                                                capacity=100000,
                                                num_threads=16,
                                                min_after_dequeue=10000,
                                                name='shuffle_batch')

        # Extract glimpses from training batch
        images = self._getGlimpses(images, random=random_glimpses)

        # Display the training images in the Tensorboard visualizer.
        tf.summary.image('Train_Images', images, max_outputs=1)

        # change to NCHW format
        images = tf.transpose(images, perm=[0, 3, 1, 2])

        return images, labels

    def eval_images_labels_batch(self, image_raw, label, distort= False, noise_min=0., noise_max=0.3,
                                 random_glimpses=False, geometric=False):
        """
        Returns: batch of images and labels to test on.
        """

        if distort:
            # we first cast to float32 to do some image operations
            image = tf.cast(image_raw, tf.float32)
            # Apply image distortions
            image = self._distort(image, noise_min, noise_max, geometric=geometric)
        else:
            image = image_raw

        # if running with half-precision we cast to float16
        if self.flags.IMAGE_FP16:
            image = tf.cast(image, tf.float16)
        else:
            image = tf.cast(image, tf.float32)

        # Generate batch
        images, labels = tf.train.shuffle_batch([image, label],
                                                batch_size=self.flags.batch_size,
                                                capacity=5100,
                                                num_threads=4,
                                                min_after_dequeue=100,
                                                name='shuffle_batch')

        #extract glimpses from evaluation batch
        images = self._getGlimpses(images, random=random_glimpses)

        # Display the training images in the visualizer.
        tf.summary.image('Test_Images', images, max_outputs=1)
        return images, labels

    @staticmethod
    def _distort(image, noise_min, noise_max, geometric=False):
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
        if geometric:
            # Setting bounds and generating random values for scaling and rotations
            scale_X = np.random.normal(1.0, 0.04, size=1)
            scale_Y = np.random.normal(1.0, 0.04, size=1)
            theta_angle = np.random.normal(0., 0.5, size=1)
            nu_angle = np.random.normal(0., 0.5, size=1)

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
            image = tf.contrib.image.transform(image, affine_transform,
                                               interpolation='BILINEAR')

        # Apply noise
        alpha = tf.random_uniform([1], minval=noise_min, maxval=noise_max)
        noise = tf.random_uniform(image.shape, dtype=tf.float32)
        image = (1 - alpha[0]) * image + alpha[0] * noise

        # normalize
        image = tf.image.per_image_standardization(image)

        return image

    def _getGlimpses(self, batch_images, **kwargs):
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
        random = kwargs.get('random', False)

        if random is 'uniform':
            # generate uniform random window centers for the batch with overlap with input
            y_low, y_high = int(crop_y_size / 2), int(y_size - crop_y_size / 2)
            x_low, x_high = int(crop_x_size / 2), int(x_size - crop_x_size / 2)
            cen_y = tf.random_uniform([self.flags.batch_size], minval=y_low, maxval=y_high)
            cen_x = tf.random_uniform([self.flags.batch_size], minval=x_low, maxval=x_high)
            offsets = tf.stack([cen_y, cen_x], axis=1)

        if random is 'normal':
            # generate normal random window centers for the batch with overlap with input
            cen_y = tf.random_normal([self.flags.batch_size], mean=y_size / 2, stddev=4.)
            cen_x = tf.random_normal([self.flags.batch_size], mean=x_size / 2, stddev=4.)
            offsets = tf.stack([cen_y, cen_x], axis=1)

        if not random:
            # fixed crop
            cen_y = np.ones((self.flags.batch_size,), dtype=np.int32) * 38
            cen_x = np.ones((self.flags.batch_size,), dtype=np.int32) * 70
            offsets = np.vstack([cen_y, cen_x]).T
            offsets = tf.constant(value=offsets, dtype=tf.float32)

        # extract glimpses
        glimpse_batch = tf.image.extract_glimpse(batch_images, size, offsets, centered=False,
                                                 normalized=False,
                                                 uniform_noise=False,
                                                 name='batch_glimpses')
        return glimpse_batch