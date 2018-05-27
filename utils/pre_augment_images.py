import tensorflow as tf
import h5py
import numpy as np


def pre_process_hdf5_only(h5_raw, h5_aug, params):
    """
    Adds noise + glimpse for a set of images in a HDF5 dataset and writes results back to the h5_aug dataset.
    One image is generated per input image.

    :param h5_raw: 3D h5py.Dataset object structured as [example, rows, cols] containing the clean images
    :param h5_aug: 3D h5py.Dataset object structured as [example, rows, cols] that will hold the augmented image
    :param params: dict
    :return:
    """
    for det in [h5_raw, h5_aug]:
        assert isinstance(det, h5py.Dataset)
    assert isinstance(params, dict)

    assert h5_raw.shape == h5_aug.shape
    assert h5_raw.dtype == h5_aug.dtype
    assert h5_raw.dtype == np.float16

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    raw_image = tf.placeholder(tf.float16, shape=(h5_raw.shape[1], h5_raw.shape[2]), name='raw_image')

    # upcast to float32 since noise addition only works on float32
    raw_image = tf.cast(raw_image, tf.float32)

    # add noise to 2D image
    noisy_image = add_noise_image(raw_image, params)

    # make it a 4D tensor:
    # fake batch size of 1
    noisy_image_3d = tf.expand_dims(noisy_image, axis=0)
    # fake depth of 1:
    noisy_image_4d = tf.expand_dims(noisy_image_3d, axis=-1)

    # glimpse this 3D image:
    params.update({'batch_size': 1})
    glimpsed_image = get_glimpses(noisy_image_4d, params)

    with tf.Session() as session:
        session.run(init_op)
        for eg_ind in range(h5_raw.shape[0]):

            aug_image = session.run(glimpsed_image, feed_dict={raw_image: h5_raw[eg_ind]})
            if eg_ind % 100 == 0 and eg_ind > 0:
                print(eg_ind)
            h5_aug[eg_ind] = aug_image[0, :, :, 0]
