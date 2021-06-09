from __future__ import absolute_import, division, print_function, unicode_literals

import os, sys, re, unittest
sys.path.append('..')

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import numpy as np
import tensorflow as tf

from classifiers.tensorflow import TFClassifier;
import models.model_zoo as model_zoo;

NB_TRAIN = 20000
NB_TEST = 1000


class TestTFClassifier(unittest.TestCase):
    """
    This class tests the functionalities of the Tensorflow-based classifier.
    """
    @classmethod
    def setUpClass(cls):
        # Get MNIST
        (x_train, y_train), (x_test, y_test), _, _ = model_zoo.load_mnist();
        x_train, y_train = x_train[:NB_TRAIN], y_train[:NB_TRAIN]
        x_test, y_test = x_test[:NB_TEST], y_test[:NB_TEST]
        cls.mnist = (x_train, y_train), (x_test, y_test)

    def setUp(self):
        logger.info('setUp');
        model_zoo.master_seed(1234);

        # Define input and output placeholders
        input_ph = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
        output_ph = tf.placeholder(tf.int32, shape=[None, 10])

        # Define the tensorflow graph
        conv = tf.layers.conv2d(input_ph, 16, 5, activation=tf.nn.relu)
        conv = tf.layers.max_pooling2d(conv, 2, 2)
        fc = tf.contrib.layers.flatten(conv)

        # Logits layer
        logits = tf.layers.dense(fc, 10)

        # Train operator
        loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=output_ph))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        train = optimizer.minimize(loss)

        # Tensorflow session and initialization
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        # Create classifier
        self.classifier = TFClassifier((0, 1), input_ph, logits, output_ph, train, loss, self.sess)

    def tearDown(self):
        logger.info('tearDown');
        self.sess.close();

    def test_fit_predict(self):
        logger.info('test_fit_predict');
        (x_train, y_train), (x_test, y_test) = self.mnist

        # Test fit and predict
        self.classifier.fit(x_train, y_train, batch_size=100, nb_epochs=1)
        preds = self.classifier.predict(x_test)
        preds_class = np.argmax(preds, axis=1)
        trues_class = np.argmax(y_test, axis=1)
        acc = np.sum(preds_class == trues_class) / len(trues_class)

        logger.info('Accuracy after fitting: %.2f%%', (acc * 100))
        self.assertGreater(acc, 0.1)
        tf.reset_default_graph()

    def test_nb_classes(self):
        logger.info('test_nb_classes');
        self.assertTrue(self.classifier.nb_classes == 10)
        tf.reset_default_graph()

    def test_input_shape(self):
        logger.info('test_input_shape');
        self.assertTrue(np.array(self.classifier.input_shape == (28, 28, 1)).all())
        tf.reset_default_graph()

    def test_class_gradient(self):
        logger.info('test_class_gradient');
        (_, _), (x_test, _) = self.mnist

        # Test all gradients label = None
        grads = self.classifier.class_gradient(x_test)

        self.assertTrue(np.array(grads.shape == (NB_TEST, 10, 28, 28, 1)).all())
        self.assertTrue(np.sum(grads) != 0)

        # Test 1 gradient label = 5
        grads = self.classifier.class_gradient(x_test, label=5)

        self.assertTrue(np.array(grads.shape == (NB_TEST, 1, 28, 28, 1)).all())
        self.assertTrue(np.sum(grads) != 0)

        # Test a set of gradients label = array
        label = np.random.randint(5, size=NB_TEST)
        grads = self.classifier.class_gradient(x_test, label=label)
        self.assertTrue(np.array(grads.shape == (NB_TEST, 1, 28, 28, 1)).all())
        self.assertTrue(np.sum(grads) != 0)

        grads = self.classifier.class_gradient(x_test, logits=True)
        self.assertTrue(np.array(grads.shape == (NB_TEST, 10, 28, 28, 1)).all())
        self.assertTrue(np.sum(grads) != 0)

    def test_loss_gradient(self):
        logger.info('test_loss_gradient');
        (_, _), (x_test, y_test) = self.mnist

        # Test gradient
        grads = self.classifier.loss_gradient(x_test, y_test)

        self.assertTrue(np.array(grads.shape == (NB_TEST, 28, 28, 1)).all())
        self.assertTrue(np.sum(grads) != 0)
        tf.reset_default_graph();

    def test_activations(self):
        logger.info('test_activations');
        (_, _), (x_test, _) = self.mnist

        # Test and get layers
        layer_names = self.classifier.layer_names
        logger.info(layer_names)
        # self.assertTrue(layer_names == ['conv2d/Relu:0', 'max_pooling2d/MaxPool:0', 'Flatten/Reshape:0', 'dense/BiasAdd:0'])

        for i, name in enumerate(layer_names):
            act_i = self.classifier.get_activations(x_test, i)
            act_name = self.classifier.get_activations(x_test, name)
            self.assertAlmostEqual(np.sum(act_name - act_i), 0)
            logger.info('layer_name={}\t{}'.format(name, act_name.shape));

        self.assertTrue(self.classifier.get_activations(x_test, 0).shape == (NB_TEST, 24, 24, 16))
        self.assertTrue(self.classifier.get_activations(x_test, 1).shape == (NB_TEST, 12, 12, 16))
        self.assertTrue(self.classifier.get_activations(x_test, 2).shape == (NB_TEST, 2304))
        self.assertTrue(self.classifier.get_activations(x_test, 3).shape == (NB_TEST, 10))
        tf.reset_default_graph()

    def test_layer_gradient(self):
        logger.info('test_layer_gradient');
        (x_train, _), (x_test, _) = self.mnist;

        layer_names = self.classifier.layer_names
        logger.info(layer_names)
        # self.assertTrue(layer_names == ['conv2d/Relu:0', 'max_pooling2d/MaxPool:0', 'Flatten/Reshape:0', 'dense/BiasAdd:0'])

        for i, name in enumerate(layer_names):
            grads = self.classifier.layer_gradient(x_test[0:100,:,:,:], i, x_test[100:200,:,:,:]);
            logger.info('grads={}\t{}'.format(grads.shape, np.sum(grads)));

    def test_save(self):
        logger.info('test_save');

        path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')), 'data');
        filename = 'model.ckpt'
        self.classifier.save(filename, path=path)
        self.assertTrue(os.path.isfile(os.path.join(path, filename + '.meta')))
        self.assertTrue(os.path.isfile(os.path.join(path, filename + '.index')))

        # Remove saved files
        for f in os.listdir(path):
            if re.search(filename, f):
                os.remove(os.path.join(path, f))


if __name__ == '__main__':
    unittest.main()