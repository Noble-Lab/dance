from __future__ import absolute_import, division, print_function, unicode_literals

import os, sys, random
sys.path.append('..')

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import numpy as np


def create_sprite(images, normalize=True):
    """
    Creates a sprite of provided images.

    :param images: Images to construct the sprite.
    :type images: `np.array`
    :param normalize: whether to divided by the maximum value per image.
    :type normalize: `bool`
    :return: An image array containing the sprite.
    :rtype: `np.ndarray`
    """

    shape = np.shape(images);

    if len(shape) < 3 or len(shape) > 4:
        raise ValueError('Images provided for sprite have wrong dimensions ' + str(len(shape)));

    if len(shape) == 3:
        # Check to see if it's mnist type of images and add axis to show image is gray-scale
        images = np.expand_dims(images, axis=3);
        shape = np.shape(images);

    if normalize: images /= np.max(images);

    # Change black and white images to RGB
    if shape[3] == 1:
        images = convert_to_rgb(images)

    n = int(np.ceil(np.sqrt(images.shape[0])))
    padding = ((0, n ** 2 - images.shape[0]), (0, 0), (0, 0)) + ((0, 0),) * (images.ndim - 3)
    images = np.pad(images, padding, mode='constant', constant_values=0)

    # Tile the individual thumbnails into an image
    images = images.reshape((n, n) + images.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, images.ndim + 1)))
    images = images.reshape((n * images.shape[1], n * images.shape[3]) + images.shape[4:])
    sprite = (images * 255).astype(np.uint8)

    return sprite


def convert_to_rgb(images):
    """
    Converts grayscale images to RGB. It changes NxHxWx1 to a NxHxWx3 array, where N is the number of figures,
    H is the high and W the width.

    :param images: Grayscale images of shape (NxHxWx1).
    :type images: `np.ndarray`
    :return: Images in RGB format of shape (NxHxWx3).
    :rtype: `np.ndarray`
    """
    s = np.shape(images)
    if not ((len(s) == 4 and s[-1] == 1) or len(s) == 3):
        raise ValueError('Unexpected shape for grayscale images:' + str(s))

    if s[-1] == 1:
        # Squeeze channel axis if it exists
        rgb_images = np.squeeze(images, axis=-1)
    else:
        rgb_images = images
    rgb_images = np.stack((rgb_images,) * 3, axis=-1)

    return rgb_images


def save_image(image, filename, path=None, cmap = 'gray'):
    """
    Saves image into a file inside `DATA_PATH` with the name `filename`.

    :param image: Image to be saved
    :type image: `np.ndarray`
    :param filename: File name containing extension e.g., my_img.jpg, my_img.png, my_images/my_img.png
    :type filename: `str`
    :return: `None`
    """
    if path is None:
        path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')), 'data')

    if not os.path.exists(path): os.makedirs(path)
    full_path = os.path.join(path, filename)

    # from PIL import Image
    # im = Image.fromarray(image);
    # im.save(full_path)
    # logger.info('Image saved to %s.', full_path)

    from matplotlib import pyplot as plt

    logger.info('image shape: {}'.format(image.shape));
    assert len(image.shape) <= 3;

    if cmap == 'gray':
        if len(image.shape) == 3: image = np.max(image, axis=2);
        plt.imsave(full_path, image.reshape(image.shape[0], image.shape[1]), cmap='viridis');
        # plt.imsave(full_path, image.reshape(image.shape[0], image.shape[1]), cmap='plasma');
    else:
        plt.imsave(full_path, image);
