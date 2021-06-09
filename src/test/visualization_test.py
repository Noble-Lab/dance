from __future__ import absolute_import, division, print_function, unicode_literals

import os, sys, re, unittest
sys.path.append('..')

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import numpy as np

import models.model_zoo as model_zoo;
import viz.visualization as visualization;

NB_TRAIN = 20000
NB_TEST = 1000
BATCH_SIZE = 128

class TestVisualization(unittest.TestCase):
    """
    This class tests the functionalities of visualization
    """

    def setUp(self):
        logger.info('setUp');
        model_zoo.master_seed(1234);


    def tearDown(self):
        logger.info('tearDown');

    def test_save_image(self):
        (x, _), (_, _), _, _ = model_zoo.load_mnist(raw=True);

        f_name = 'image1.png';
        visualization.save_image(x[0], f_name);
        path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')), 'data', f_name);
        self.assertTrue(os.path.isfile(path))
        os.remove(path)

    def test_convert_gray_to_rgb(self):
        (x, _), (_, _), _, _ = model_zoo.load_mnist(raw=True);
        n = 100;
        x = x[:n];

        # Test RGB
        x_rgb = visualization.convert_to_rgb(x);
        s_original = np.shape(x);
        s_new = np.shape(x_rgb);

        self.assertEqual(s_new[0], s_original[0]);
        self.assertEqual(s_new[1], s_original[1]);
        self.assertEqual(s_new[2], s_original[2]);
        self.assertEqual(s_new[3], 3);  # Should have added 3 channels

    def test_sprites_gray(self):
        (x, _), (_, _), _, _ = model_zoo.load_mnist(raw=True);
        n = 100;
        x = x[:n];

        sprite = visualization.create_sprite(x);
        f_name = 'test_sprite_mnist.png';
        visualization.save_image(sprite, f_name);
        path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')), 'data', f_name);
        self.assertTrue(os.path.isfile(path));
        os.remove(path);

    def test_sprites_color(self):
        (x, _), (_, _), _, _ = model_zoo.load_cifar10(raw=True);
        n = 100;
        x = x[:n];
        logger.info('cifar x size: {}'.format(x.shape));

        sprite = visualization.create_sprite(x);
        f_name = 'test_cifar.png';
        visualization.save_image(sprite, f_name);
        # path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')), 'data', f_name);
        # self.assertTrue(os.path.isfile(path));
        # os.remove(path);

if __name__ == '__main__':
    unittest.main()