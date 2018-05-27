"""
Created on 1/22/18
@author: Suhas Somnath
"""

import numpy as np
import sys
import os

import tensorflow as tf
import mxnet as mx

import json


def load_json_hyper_params(file):
    """
    Loads hyper_parameters dictionary from .json file
    :param file: string, path to .json file
    :return: dict, hyper-paramters.
    """
    with open(file, mode='r') as f:
        hyper_params = json.load(f)
    print('Read %d hyperparameters from %s' % (len(hyper_params.keys()), file))
    return hyper_params


def load_flags_from_json(file_path, flags, verbose=False):
    image_parms = load_json_hyper_params(file_path)
    for parm_name, parm_values in list(image_parms.items()):
        if parm_values['type'] == 'bool':
            func = flags.DEFINE_boolean
        elif parm_values['type'] == 'int':
            func = flags.DEFINE_integer
        elif parm_values['type'] == 'float':
            func = flags.DEFINE_float
        elif parm_values['type'] == 'str':
            func = flags.DEFINE_string
        else:
            raise NotImplemented('Cannot handle type: {} for parameter: {}'.format(parm_values['type'], parm_name))
        if verbose:
            print('{} : {} saved as {} with description: {}'.format(parm_name, parm_values['value'],
                                                                    parm_values['type'], parm_values['desc']))
        func(parm_name, parm_values['value'], parm_values['desc'])


class Tf2MxRecords(object):

    def __init__(self, input_tf_record, data_params):
        load_flags_from_json(data_params, tf.app.flags)
        self.tf_filepath = input_tf_record
        self.file_queue = tf.train.string_input_producer([self.tf_filepath], num_epochs=1)
        self.flags = tf.app.flags.FLAGS

    def __decode_image_label(self, reader):
        """
        Returns: image, label decoded from tfrecords
        """
        key, serialized_example = reader.read(self.file_queue)

        # get raw image bytes and label string
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
        return image, label

    def convert(self):
        """
        Creates a new tfRecords data file by applying the provided processing routines.

        Inspired from:
        https://stackoverflow.com/questions/37151895/tensorflow-read-all-examples-from-a-tfrecords-at-once
        """
        in_tfrecord = self.tf_filepath
        out_mx_path = in_tfrecord.replace('.tfrecords', '.rec')

        if os.path.exists(out_mx_path):
            print('Removing old output file')
            os.remove(out_mx_path)

        with tf.Session() as sess:

            reader = tf.TFRecordReader()

            image, label = self.__decode_image_label(reader=reader)

            # Initialize all global and local variables
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init_op)

            # Create a coordinator and run all QueueRunner objects
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            # open the TFRecords output file
            writer = mx.recordio.MXRecordIO(out_mx_path, 'w')

            log_per_examples = 5000

            try:
                image_index = 0
                # go image by image in the entire file!
                # for _ in range(32):
                while True:
                    image_in, label_in = sess.run([image, label])

                    # risky but won't work with 32 bit
                    image_in = np.float16(image_in)
                    # print(image_in.shape)

                    # Create a header
                    header = mx.recordio.IRHeader(0, label_in, image_index, 0)

                    packed_str = mx.recordio.pack(header, image_in.tostring())

                    # write to the file
                    writer.write(packed_str)

                    # count number of images. The number in the flags is wrong
                    image_index += 1

                    if image_index % log_per_examples == 0:
                        print('Finished {} examples'.format(image_index))

            except tf.errors.OutOfRangeError, e:
                print('Encountered an OutofRangeError!')
                coord.request_stop(e)
            finally:
                # close out the writer
                writer.close()
                sys.stdout.flush()

                # Stop the threads
                coord.request_stop()

                # Wait for threads to stop
                coord.join(threads)

            print('Finished preprocessing entire file having {} images'.format(image_index))


if __name__ == '__main__':
    if  len(sys.argv) != 3:
        print('1st argument is the path to the data file and the second is the path to the .json file with '
              'information about this data file')
    converter = Tf2MxRecords(sys.argv[1], sys.argv[2])
    converter.convert()
