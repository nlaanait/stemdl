"""
Created on 10/8/17.
@author: Numan Laanait.
email: laanaitn@ornl.gov
"""

import tensorflow as tf
import numpy as np
import sys
import os
from itertools import chain, cycle
from tensorflow.python.ops import data_flow_ops
import horovod.tensorflow as hvd
import lmdb
import time

tf.logging.set_verbosity(tf.logging.ERROR)

def print_rank(*args, **kwargs):
    if hvd.rank() == 0:
        print(*args, **kwargs)

class DatasetTFRecords(object):
    """
    Handles training and evaluation data operations.  \n
    Data is read from a TFRecords filename queue.
    """

    def __init__(self, params, dataset=None, mode='train',
                debug=False):
        self.params = params
        self.mode = mode
        self.debug = debug
        self.dataset = dataset
        if self.dataset == 'spacegroup_classification':
            self.features_specs={'image_keys':['image_raw'], 'label_keys':['space_group'],
                                'specs': spacegroup}
        elif self.dataset == 'chemicalcomp_regression':
            self.features_specs={'specs': spacegroup,
                            'image_keys':['image_raw'],
                            'label_keys':['chemical_comp']}
        elif self.dataset == 'spacegroup_chemicalcomp':
            self.features_specs = {'specs':spacegroup,
                            'image_keys':['image_raw'],
                            'label_keys':['chemical_comp', 'space_group']}
        elif self.dataset == '2d_reconstruction':
            self.features_specs = {'image_keys': ['cbed'],
                            'label_keys': ['2d_potential'], 'specs': reconstruction_2d}
        elif self.dataset == 'abf_oxides_regression':
            self.features_specs = {'image_keys': ['image_raw'],
                            'label_keys': ['label'], 'specs': abf_oxides_regression}
        elif self.dataset == 'abf_oxides_classification':
            self.features_specs = {'image_keys': ['image_raw'],
                            'label_keys': ['label'], 'specs': abf_oxides_classification}
        elif self.dataset is None:
            self.features_specs = None

    def set_mode(self,mode='train'):
        self.mode = mode


    def decode_image_label(self, record):
        """
        Returns: image, label decoded from ds
        """

        if self.features_specs is None:
            features = tf.parse_single_example( record,
                                                features={
                 'image_raw': tf.FixedLenFeature([], tf.string),
                 'label': tf.FixedLenFeature([], tf.string),
            })
            # decode from byte and reshape label and image
            label_dtype = tf.as_dtype(self.params['LABEL_DTYPE'])
            label = tf.decode_raw(features['label'], label_dtype)
            label.set_shape(self.params['NUM_CLASSES'])
            image = tf.decode_raw(features['image_raw'], self.params['IMAGE_DTYPE'] )
            image.set_shape([self.params['IMAGE_HEIGHT'] * self.params['IMAGE_WIDTH'] * self.params['IMAGE_DEPTH']])
            image = tf.reshape(image, [self.params['IMAGE_HEIGHT'], self.params['IMAGE_WIDTH'], self.params['IMAGE_DEPTH']])
        else:
            specs = self.features_specs['specs']
            features_images = [(image_key, tf.FixedLenFeature([], tf.string))
                                    for image_key in self.features_specs['image_keys']]
            features_labels = [(label_key, tf.FixedLenFeature([], tf.string))
                                    for label_key in self.features_specs['label_keys']]
            # parse a single record
            features_all = dict(features_labels + features_images)
            features = tf.parse_single_example(record, features=features_all)

            # process labels
            labels = []
            for label_key in self.features_specs['label_keys']:
                label_dtype = tf.as_dtype(specs[label_key]['dtype'])
                label_shape = specs[label_key]['shape']
                label = tf.decode_raw(features[label_key], label_dtype)
                label.set_shape(np.prod(np.array(label_shape)))
                label = tf.reshape(label, label_shape)
                if specs[label_key]['dtype'] == 'int64':
                    label = tf.cast(label, tf.float64)
                labels.append(label)
            if len(labels) == 1:
                label = labels[0]
            else:
                label = tf.concat([tf.expand_dims(label, 0) for label in labels], 1)

            # process images
            images = []
            for image_key in self.features_specs['image_keys']:
                image_dtype = tf.as_dtype(specs[image_key]['dtype'])
                image_shape = specs[image_key]['shape']
                image = tf.decode_raw(features[image_key], image_dtype)
                image.set_shape(np.prod(np.array(image_shape)))
                image = tf.reshape(image, image_shape)
                images.append(image)
            if len(images) > 1:
                image = images[0]
            else:
                image = tf.concat(images, 1)
                # TODO stack images and return as one
                pass
        if self.features_specs is None or specs['preprocess']:
            # TODO: all of this should be cached
            # standardize the image to [-1.,1.]
            #image = tf.sqrt(image)
            #image = tf.image.per_image_standardization(image)
            pass
            # Checking for nan, bug in simulation codes...
            #image = tf.where(tf.is_nan(image), -tf.ones_like(image), image)
            # Manipulate labels
            #label = tf.expand_dims(label,axis=0)
            #label = tf.sqrt(tf.sqrt(label))
            #label = tf.image.per_image_standardization(label)
        return image, label

    def glimpse_at_image(self, image):
        """
        Apply isotropic scaling, sampled from a normal distribution.

        :param image: 2D tensor
        :param params: dict, image dimension parameters must be included
        :return: 2D tensor
        """
        #TODO: change calls to image specs from self.params to self.features
        image_params = self.features_specs['specs'][self.features_specs['image_keys'][0]]
        zoom_factor = np.random.normal(1.0, 0.05, size=1)
        crop_y_size, crop_x_size = image_params['IMAGE_HEIGHT'], image_params['IMAGE_WIDTH']
        size = tf.constant(value=[int(np.round(crop_y_size / zoom_factor)),
                                  int(np.round(crop_x_size / zoom_factor))], dtype=tf.int32)
        cen_y = np.ones((1,), dtype=np.float32) * int(image_params['IMAGE_HEIGHT'] / 2)
        cen_x = np.ones((1,), dtype=np.float32) * int(image_params['IMAGE_WIDTH'] / 2)
        offsets = tf.stack([cen_y, cen_x], axis=1)
        scaled_image = tf.expand_dims(image, axis=0)
        scaled_image = tf.image.extract_glimpse(scaled_image, size, offsets,
                                                centered=False,
                                                normalized=False,
                                                uniform_noise=False)
        scaled_image = tf.reshape(scaled_image, (scaled_image.shape[1].value, scaled_image.shape[2].value,
                                                 scaled_image.shape[3].value))
        scaled_image = tf.image.resize_images(scaled_image, (image_params['IMAGE_HEIGHT'], image_params['IMAGE_WIDTH']))
        return scaled_image


    def add_noise_image(self, image):
        """
        Adds random noise to the provided image

        :param image: 2D image specified as a tensor
        :param params: dict, parameters required - noise_min and noise_max
        :return: 2d tensor
        """

        alpha = tf.random_uniform([1], minval=self.params['noise_min'], maxval=self.params['noise_max'], dtype=image.dtype)
        noise = tf.random_uniform(image.shape, dtype=image.dtype)
        trans_image = (1 - alpha[0]) * image / tf.reduce_max(image, keepdims=True) + alpha[0] * noise
        return trans_image


    def distort(self, image):
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

        image = self.rotate_image(image)
        image = self.add_noise_image(image)
        return image


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
        image_params = self.features_specs['specs'][self.features_specs['image_keys'][0]]
        y_size, x_size = image_params['IMAGE_HEIGHT'], image_params['IMAGE_WIDTH']
        crop_y_size, crop_x_size = image_params['CROP_HEIGHT'], image_params['CROP_WIDTH']
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


    def minibatch(self):
        """
        Returns minibatch of images and labels from TF records file.
        """
        mode = self.mode
        batch_size = self.params['batch_size']
        if mode not in ['train', 'validation', 'test']:
            mode = 'train'

        if self.debug: self.inspect_tfrecords(mode)

        record_input = data_flow_ops.RecordInput(
            file_pattern=os.path.join(self.params['data_dir'], '*.tfrecords'),
            parallelism=self.params['IO_threads'],
            buffer_size=self.params['buffer_cap'],
            batch_size=batch_size)
        records = record_input.get_yield_op()

        # Split batch into individual images
        records = tf.split(records, batch_size, 0)
        records = [tf.reshape(record, []) for record in records]
        #print('record contents %s' %(format(records)))
        #print('record length %s and contents %s' %(len(records),format(records)))
        # Deserialize and preprocess images into batches for each device
        images = []
        labels = []
        with tf.name_scope('input_pipeline'):
            if self.params[mode + '_distort']:
                print_rank('images will be distorted')

            for i, record in enumerate(records):
                image, label = self.decode_image_label(record)
                if self.params[mode + '_distort']:
                    # image = self.add_noise_image(image)
                    image = self.distort(image)
                images.append(image)
                labels.append(label)
                image_shape = image.get_shape().as_list()
                label_shape = label.get_shape().as_list()
            # Stack images and labels back into a single tensor
            labels = tf.parallel_stack(labels)
            images = tf.parallel_stack(images)

            # reshape them to the expected shape:
            labels_newshape = [batch_size] + label_shape
            images_newshape = [batch_size] + image_shape
            labels = tf.reshape(labels, labels_newshape)
            images = tf.reshape(images, images_newshape)

            # glimpse images: moved to GPU
            #images = self.get_glimpses(images)

            # Display the training images in the Tensorboard visualizer.
            if self.debug: tf.summary.image("images", images, max_outputs=4)

            # resize
            if self.params['resize']:
                images = tf.image.resize_bilinear(images, [self.params['RESIZE_WIDTH'],
                                    self.params['RESIZE_HEIGHT']])
            if self.params['tile']:
                images = tf.ones([self.params['IMAGE_DEPTH'], self.params['IMAGE_HEIGHT'],
                        self.params['IMAGE_WIDTH']], dtype=self.params['IMAGE_DTYPE'])
                labels = tf.ones([256, 512,512], dtype=self.params['LABEL_DTYPE'])

        return images, labels


    @staticmethod
    def stage(tensors):
        """
        Stages the given tensors in a StagingArea for asynchronous put/get
        :param tensors: tf.Tensor
        :return: get and put tf.Op operations.
        """
        staging_area = data_flow_ops.StagingArea(
            dtypes=[tensor.dtype       for tensor in tensors],
            shapes=[tensor.get_shape() for tensor in tensors])
        load_op = staging_area.put(tensors)
        get_tensors = staging_area.get()

        get_tensors = [tf.reshape(get_t, t.get_shape())
                       for (get_t,t) in zip(get_tensors, tensors)]
        return load_op, get_tensors


    @staticmethod
    def onehot(label):
        index = tf.cast(label[0],tf.int32)
        full_vec = tf.cast(tf.linspace(20., 200., 91),tf.int32)
        bool_vector = tf.equal(index, full_vec)
        onehot_vector = tf.cast(bool_vector, tf.int64)
        return onehot_vector


    @staticmethod
    def label_minmaxscaling(label, min_vals, max_vals, scale_range=[0,1]):
        """

        :param label: tensor
        :param min_vals: list, minimum value for each label dimension
        :param max_vals: list, maximum value for each label dimension
        :param range: list, range of label, default [0,1]
        :return:
        scaled label tensor
        """
        min_tensor = tf.constant(min_vals, dtype=tf.float32)
        max_tensor = tf.constant(max_vals, dtype=tf.float32)
        if label.dtype != tf.float32:
            orig_dtype = label.dtype
            label = tf.cast(label, tf.float32)
        scaled_label = (label - min_tensor)/(max_tensor - min_tensor)
        scaled_label = scaled_label * (scale_range[-1] - scale_range[0]) + scale_range[0]
        if scaled_label.dtype != orig_dtype:
            scaled_label = tf.cast(scaled_label, orig_dtype)
        return scaled_label


    @staticmethod
    def rotate_image(image):
        """
        Apply random global affine transformations, sampled from a normal distributions.

        :param image: 2D tensor
        :return: 2D tensor
        """

        # Setting bounds and generating random values for scaling and rotations
        scale_X = np.random.normal(1.0, 0.025, size=1)
        scale_Y = np.random.normal(1.0, 0.025, size=1)
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
        return aff_image

    def inspect_tfrecords(self, mode):
        dir = self.params['data_dir']
        tf_filenames = os.listdir(dir)
        for fname in tf_filenames:
            if fname.split('.')[-1] == "tfrecords":
                tf_filename = os.path.join(dir, fname)
                break
        print_rank('inspecting file: %s' %tf_filename)
        record_iterator = tf.python_io.tf_record_iterator(path=tf_filename)
        if self.features_specs is None:
            image_key, label_key = ['image_raw', 'label']
            label_dtype = tf.as_dtype(self.params['LABEL_DTYPE'])
            image_shape = [self.params['IMAGE_HEIGHT'], self.params['IMAGE_WIDTH'], self.params['IMAGE_DEPTH']]
            image_dtype = tf.as_dtype(self.params['IMAGE_DTYPE'])
        else:
            specs = self.features_specs['specs']
            image_key, label_key = [self.features_specs['image_keys'][0], self.features_specs['label_keys'][0]]
            image_size =  np.prod(np.array(specs[image_key]['shape']))
            label_size = np.prod(np.array(specs[label_key]['shape']))
            image_dtype = specs[image_key]['dtype']
            label_dtype = specs[label_key]['dtype']

        for (i,string_record) in enumerate(record_iterator):
            example = tf.train.Example()
            example.ParseFromString(string_record)
            label = np.fromstring(example.features.feature[label_key].bytes_list.value[0],dtype=label_dtype)
            image = np.fromstring(example.features.feature[image_key].bytes_list.value[0], dtype=image_dtype)
            if i > 2:
                break
        if label.size != label_size or image.size != image_size:
            print_rank('image size and label size are not as expected.')
            print_rank('found: %s, %s' %(format(image.size) ,format(label.size)))
            print_rank('expected: %s, %s' %(format(image_size) ,format(label_size)))
            sys.exit()
        else:
            print_rank('found image and label with sizes: %s, %s' %(format(image.size) ,format(label.size)))


class DatasetLMDB(DatasetTFRecords):
    def __init__(self, *args, **kwargs):
        super(DatasetLMDB, self).__init__(*args, **kwargs)
        self.mode = self.params['mode']
        lmdb_dir = self.params['data_dir']
        lmdb_files = os.listdir(lmdb_dir)
        lmdb_path = os.path.join(lmdb_dir, 'batch_%s_%d.db' %  (self.mode, int(hvd.rank())))
        self.env = lmdb.open(lmdb_path, create=False, readahead=False, readonly=True, writemap=False, lock=False)
        self.num_samples = (self.env.stat()['entries'] - 6)//2 ## TODO: remove hard-coded # of headers by storing #samples key, val
        self.first_record = 0
        self.records = np.arange(self.first_record, self.num_samples)
        np.random.shuffle(self.records)
        with self.env.begin(write=False) as txn:
            input_shape = np.frombuffer(txn.get(b"input_shape"), dtype='int64')
            output_shape = np.frombuffer(txn.get(b"output_shape"), dtype='int64')
            input_dtype = np.dtype(txn.get(b"input_dtype").decode("ascii"))
            output_dtype = np.dtype(txn.get(b"output_dtype").decode("ascii"))
            output_name = txn.get(b"output_name").decode("ascii")
            input_name = txn.get(b"input_name").decode("ascii")
            
        self.data_specs={'label_shape': list(output_shape), 'image_shape': list(input_shape), 
            'label_dtype':output_dtype, 'image_dtype': input_dtype, 'label_key':output_name, 'image_key': input_name}
        self.image_keys = [bytes(self.data_specs['image_key']+str(idx), "ascii") for idx in self.records]
        self.label_keys = [bytes(self.data_specs['label_key']+str(idx), "ascii") for idx in self.records]
        if self.debug:
            print('rank=%d, lmdb=%s, num_samples=%d' %(hvd.rank(),lmdb_path, self.num_samples))

    def decode_image_label(self, idx):
        """
        idx: index of sample
        Returns: image, label tensors read from lmdb environment
        """
        idx = idx[0]        
        t = time.time()
        image_key = self.image_keys[idx]
        label_key = self.label_keys[idx]
        with self.env.begin(write=False, buffers=True) as txn:
            image_bytes = txn.get(image_key)
            label_bytes = txn.get(label_key)
        label = np.frombuffer(label_bytes, dtype=self.data_specs['label_dtype'])
        image = np.frombuffer(image_bytes, dtype=self.data_specs['image_dtype'])

        #if self.debug: 
        #    print_rank('rank=%d, read image %s %s and label %s %s from lmdb' %(hvd.rank(),format(image.shape), 
        #    format(image.dtype), format(label.shape), format(label.dtype)))
        #if self.debug:
        #    print_rank('time to read and convert to tensor: %2.2f' % (time.time()-t))
        return image, label
            
    def generator(self):
        for record in cycle(self.records):
            yield (record)

    def wrapped_decode(self, idx):
        return tf.py_func(self.decode_image_label, [idx], 
                         [tf.as_dtype(self.data_specs['image_dtype']),tf.as_dtype(self.data_specs['label_dtype'])])

    def minibatch(self):
        """
        Returns minibatch of images and labels from TF records file.
        """
        with tf.name_scope('pipeline'):
            ds = tf.data.Dataset.from_generator(
                self.generator, 
                (tf.int64),
                (tf.TensorShape([]))
                )
            if self.mode == 'train':
                max_num_records = self.params['num_epochs'] * self.params['NUM_EXAMPLES_PER_EPOCH']
                ds = ds.take(max_num_records)
                ds = ds.prefetch(min(1, self.num_samples))
                ds = ds.batch(self.params['batch_size'], drop_remainder=True)
                #ds = ds.map(self.wrapped_decode, num_parallel_calls=tf.data.experimental.AUTOTUNE)
                ds = ds.map(self.wrapped_decode)
                iterator = ds.make_one_shot_iterator()
                images, labels = [],[]
                for _ in range(self.params['batch_size']):
                    image, label = iterator.get_next()
                    image = tf.reshape(image, self.data_specs['image_shape'])
                    if self.params[self.mode + '_distort']:
                        image = self.add_noise_image(image)
                    images.append(tf.reshape(image, self.data_specs['image_shape']))
                    labels.append(tf.reshape(label, self.data_specs['label_shape']))
            elif self.mode == 'eval':
                # self.params['batch_size'] = 1
                ds = ds.batch(self.params['batch_size'], drop_remainder=True)
                ds = ds.map(self.wrapped_decode)
                iterator = ds.make_one_shot_iterator()
                images, labels = [],[]
                if self.params[self.mode + '_distort']:
                    print('images will be distorted')
                for _ in range(self.params['batch_size']):
                    image, label = iterator.get_next()
                    image = tf.reshape(image, self.data_specs['image_shape'])
                    if self.params[self.mode + '_distort']:
                        image = self.add_noise_image(image)
                    images.append(image)
                    labels.append(tf.reshape(label, self.data_specs['label_shape']))
            if tf.executing_eagerly():
                images = tf.stack(images)
                labels = tf.stack(labels)
            else:
                images = tf.parallel_stack(images)
                labels = tf.parallel_stack(labels)
            # reshape them to the expected shape:
            labels_newshape = [self.params['batch_size']] + self.data_specs['label_shape']
            images_newshape = [self.params['batch_size']] + self.data_specs['image_shape']
            labels = tf.reshape(labels, labels_newshape)
            images = tf.reshape(images, images_newshape)
            
            labels = self.image_scaling(labels)
            # labels -= tf.reduce_min(labels, keepdims=True) 
            # abels= self.label_minmaxscaling(labels, 0.0, 1.0, scale_range=[0., 10.0])
        # images = self.image_scaling(images)
        # Display the training images in the Tensorboard visualizer.
        #if self.debug: 
        #    tf.summary.image("potential", tf.transpose(labels, perm=[0,2,3,1]), max_outputs=4)
        #    tf.summary.image("images", tf.transpose(tf.reduce_mean(images, axis=1, keepdims=True), perm=[0,2,3,1]), max_outputs=4)
        return images, labels

    @staticmethod
    def image_scaling(image_batch):
        image_batch -= tf.reduce_min(image_batch, axis=[2,3], keepdims=True)
        image_batch = tf.sqrt(image_batch)
        return image_batch


# class DatasetLMDB(DatasetTFRecords):
#     def __init__(self, *args, **kwargs):
#         super(DatasetLMDB, self).__init__(*args, **kwargs)
#         self.mode = self.params['mode']
#         lmdb_dir = self.params['data_dir']
#         lmdb_dir = os.path.join(lmdb_dir, 'batch_%d' %  int(hvd.rank()))
#         lmdb_path = [os.path.join(lmdb_dir, p) for p in os.listdir(lmdb_dir)]
#         self.env = [lmdb.open(path, create=False, readahead=False, readonly=True, writemap=False, lock=False) for path in lmdb_path]
#         self.num_samples = [(env.stat()['entries'] - 6)//2 for env in self.env] ## TODO: remove hard-coded # of headers by storing #samples key, val
#         self.first_record = 0
#         self.records = [np.arange(self.first_record, num_sample) for num_sample in self.num_samples]
#         _ = [np.random.shuffle(rec) for rec in self.records]
#         input_shape, output_shape, input_dtype, output_dtype, output_name, input_name = [], [], [], [], [], []
#         for env in self.env:
#             with env.begin(write=False) as txn:
#                 input_shape.append(np.frombuffer(txn.get(b"input_shape"), dtype='int64'))
#                 output_shape.append(np.frombuffer(txn.get(b"output_shape"), dtype='int64'))
#                 input_dtype.append(np.dtype(txn.get(b"input_dtype").decode("ascii")))
#                 output_dtype.append(np.dtype(txn.get(b"output_dtype").decode("ascii")))
#                 output_name.append(txn.get(b"output_name").decode("ascii"))
#                 input_name.append(txn.get(b"input_name").decode("ascii"))
            
#         self.data_specs={'label_shape': list(output_shape), 'image_shape': list(input_shape), 
#             'label_dtype':output_dtype, 'image_dtype': input_dtype, 'label_key':output_name, 'image_key': input_name}
#         self.image_keys, self.label_keys = [], []
#         for (env_idx, _) in enumerate(self.env):
#             self.image_keys.append([bytes(self.data_specs['image_key'][env_idx]+str(idx), "ascii") for idx in self.records[env_idx]]) 
#             self.label_keys.append([bytes(self.data_specs['label_key'][env_idx]+str(idx), "ascii") for idx in self.records[env_idx]])
      
#         if self.debug:
#             # print(self.data_specs)
#             # print(self.image_keys)
#             print('rank=%d' % (hvd.rank()))
#             for env_idx, _ in enumerate(self.env):
#                 print('lmdb=%s, num_samples=%d' %(lmdb_path[env_idx], self.num_samples[env_idx]))

#     def decode_image_label(self, idx):
#         """
#         idx: index of sample
#         Returns: image, label tensors read from lmdb environment
#         """
#         idx = idx[0]        
#         t = time.time()
#         # randomly pick database 
#         db_idx = np.random.randint(len(self.env))
#         if idx >= len(self.image_keys[db_idx]):
#             db_idx = (db_idx + 1) % 2
#         # pick database based on rank num
#         db_idx = 0
#         if hvd.rank() in [0,1]:
#             db_idx = 1
#         image_key = self.image_keys[db_idx][idx]
#         label_key = self.label_keys[db_idx][idx]
#             # print("db_idx, len, idx", db_idx, len(self.image_keys[db_idx]), idx)
#         with self.env[db_idx].begin(write=False, buffers=True) as txn:
#             image_bytes = txn.get(image_key)
#             label_bytes = txn.get(label_key)
#         label = np.frombuffer(label_bytes, dtype=self.data_specs['label_dtype'][db_idx])
#         image = np.frombuffer(image_bytes, dtype=self.data_specs['image_dtype'][db_idx])
#         #TODO: modify to calculate largest size from self.data_specs['image_shape']
#         # if image.size != int(256**3):
#         #     image = np.pad(image, (0, 256**3 - image.size), mode='constant')

#         # if self.debug: 
#         #    print_rank('rank=%d, read image %s %s and label %s %s from lmdb' %(hvd.rank(),format(image.shape), 
#         #    format(image.dtype), format(label.shape), format(label.dtype)))
#         # if self.debug:
#         #    print_rank('time to read and convert to tensor: %2.2f' % (time.time()-t))
#         return image, label
            
#     def generator(self):
#         #TODO: modify generator since self.records is a list. for now hack...
#         db_idx = 0
#         if hvd.rank() in [0,1]:
#             db_idx = 1
#         for record in cycle(self.records[db_idx]):
#             yield (record)

#     def wrapped_decode(self, idx):
#         #TODO: note that if 'image_dtype' and/or 'label_dtype' are different for different dbs in self.env the tf.py_func call will break.
#         return tf.py_func(self.decode_image_label, [idx], 
#                          [tf.as_dtype(self.data_specs['image_dtype'][0]),tf.as_dtype(self.data_specs['label_dtype'][0])])


#     def minibatch(self):
#         """
#         Returns minibatch of images and labels from TF records file.
#         """
#         with tf.name_scope('pipeline'):
#             ds = tf.data.Dataset.from_generator(
#                 self.generator, 
#                 (tf.int64),
#                 (tf.TensorShape([]))
#                 )
#             if self.mode == 'train':
#                 max_num_records = self.params['num_epochs'] * self.params['NUM_EXAMPLES_PER_EPOCH']
#                 ds = ds.take(max_num_records)
#                 ds = ds.prefetch(min(1, np.min(self.num_samples)))
#                 ds = ds.batch(self.params['batch_size'], drop_remainder=True)
#                 #ds = ds.map(self.wrapped_decode, num_parallel_calls=tf.data.experimental.AUTOTUNE)
#                 ds = ds.map(self.wrapped_decode)
#                 iterator = ds.make_one_shot_iterator()
#                 images, labels = [],[]
#                 db_idx = 0
#                 if hvd.rank() in [0,1]:
#                     db_idx = 1
#                 image_shape = self.data_specs['image_shape'][db_idx]
#                 label_shape = self.data_specs['label_shape'][db_idx]
#                 for _ in range(self.params['batch_size']):
#                     image, label = iterator.get_next()
#                     #TODO: modify to calculate largest size from self.data_specs['image_shape'] 
#                     image = tf.reshape(image, image_shape)
#                     if self.params[self.mode + '_distort']:
#                         image = self.add_noise_image(image)
#                     #TODO: modify to calculate largest size from self.data_specs['image_shape'] 
#                     images.append(image)
#                     labels.append(tf.reshape(label, label_shape))
#             elif self.mode == 'eval':
#                 # self.params['batch_size'] = 1
#                 ds = ds.batch(self.params['batch_size'], drop_remainder=True)
#                 ds = ds.map(self.wrapped_decode)
#                 iterator = ds.make_one_shot_iterator()
#                 images, labels = [],[]
#                 if self.params[self.mode + '_distort']:
#                     print('images will be distorted')
#                 for _ in range(self.params['batch_size']):
#                     image, label = iterator.get_next()
#                     image = tf.reshape(image, self.data_specs['image_shape'])
#                     if self.params[self.mode + '_distort']:
#                         image = self.add_noise_image(image)
#                     images.append(image)
#                     labels.append(tf.reshape(label, self.data_specs['label_shape']))
#             if tf.executing_eagerly():
#                 images = tf.stack(images)
#                 labels = tf.stack(labels)
#             else:
#                 images = tf.parallel_stack(images)
#                 labels = tf.parallel_stack(labels)
#             # reshape them to the expected shape:
#             # labels_newshape = [self.params['batch_size']] + self.data_specs['label_shape'][-1]
#             # images_newshape = [self.params['batch_size']] + self.data_specs['image_shape'][-1]
#             # labels = tf.reshape(labels, labels_newshape)
#             # images = tf.reshape(images, images_newshape)

#             labels -= tf.reduce_min(labels, keepdims=True) 
#             # abels= self.label_minmaxscaling(labels, 0.0, 1.0, scale_range=[0., 10.0])
#         # images = self.image_scaling(images)
#         # Display the training images in the Tensorboard visualizer.
#         #if self.debug: 
#         #    tf.summary.image("potential", tf.transpose(labels, perm=[0,2,3,1]), max_outputs=4)
#         #    tf.summary.image("images", tf.transpose(tf.reduce_mean(images, axis=1, keepdims=True), perm=[0,2,3,1]), max_outputs=4)
#         return images, labels

#     @staticmethod
#     def image_scaling(image_batch):
#         image_batch -= tf.reduce_min(image_batch, axis=[2,3], keepdims=True)
#         image_batch = tf.sqrt(image_batch)
#         return image_batch

### common datasets ### 
spacegroup = {'energy': {'dtype':'float64', 'shape':[1]},
                      'thickness': {'dtype':'float64', 'shape':[1]},
                      'space_group': {'dtype': 'int64', 'shape':[230]},
                      'a': {'dtype':'float64', 'shape':[1]},
                      'b': {'dtype':'float64', 'shape':[1]},
                      'c': {'dtype':'float64', 'shape':[1]},
                      'alpha': {'dtype':'float64', 'shape':[1]},
                      'beta': {'dtype':'float64', 'shape':[1]},
                      'gamma': {'dtype':'float64', 'shape':[1]},
                      'chemical_comp': {'dtype': 'float64', 'shape':[94]},
                       # image
                      'image_raw': {'dtype': 'float16', 'shape':[512,512,1]},
                      'preprocess': True}

reconstruction_2d = {'material': {'dtype':'str', 'shape':[1]},
                      'space_group': {'dtype': 'int64', 'shape':[1]},
                      'abc': {'dtype':'float64', 'shape':[3]},
                      'angles': {'dtype':'float64', 'shape':[3]},
                      'formula': {'dtype': 'str', 'shape':[1]},
                       # images
                      'cbed': {'dtype': 'float16', 'shape':[1024,512,512]},
                      '2d_potential': {'dtype': 'float16', 'shape':[1,512,512]},
                      'preprocess': False}

abf_oxides_regression = {'label':{'dtype': 'float64', 'shape':[3]},
            # 'rotation_pattern':{'dtype': 'int64', 'shape':[27]},
            # images
            'image_raw':{'dtype':'float16', 'depth':1, 'IMAGE_HEIGHT': 85,
            'IMAGE_DEPTH': 1, 'IMAGE_WIDTH': 120, 'shape': [85, 120, 1],
            'preprocess': True, 'CROP_HEIGHT': 85, 'CROP_WIDTH': 120,
            'RESIZE_WIDTH': 120, 'RESIZE_HEIGHT': 85},
            'preprocess': True}

abf_oxides_classification = {'label':{'dtype': 'int64', 'shape':[27]},
            # 'rotation_pattern':{'dtype': 'int64', 'shape':[27]},
            # images
            'image_raw':{'dtype':'float16', 'depth':1, 'IMAGE_HEIGHT': 85,
            'IMAGE_DEPTH': 1, 'IMAGE_WIDTH': 120, 'shape': [85, 120, 1],
            'preprocess': True, 'CROP_HEIGHT': 85, 'CROP_WIDTH': 120,
            'RESIZE_WIDTH': 120, 'RESIZE_HEIGHT': 85},
            'preprocess': True}
