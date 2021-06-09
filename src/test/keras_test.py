from __future__ import absolute_import, division, print_function, unicode_literals

import os, sys, re, unittest, shutil
sys.path.append('..')

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import keras
import keras.backend as k
import numpy as np
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Input, Flatten
from keras.models import Sequential, Model

from classifiers.keras import KerasClassifier;
import models.model_zoo as model_zoo;

BATCH_SIZE = 10
NB_TRAIN = 20000
NB_TEST = 1000


class TestKerasClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        k.clear_session()
        k.set_learning_phase(1)

        # Get MNIST
        (x_train, y_train), (x_test, y_test), _, _ = model_zoo.load_mnist();
        x_train, y_train, x_test, y_test = x_train[:NB_TRAIN], y_train[:NB_TRAIN], x_test[:NB_TEST], y_test[:NB_TEST]
        cls.mnist = ((x_train, y_train), (x_test, y_test))
        im_shape = x_train[0].shape

        # Create basic CNN on MNIST; architecture from Keras examples
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=im_shape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(10, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

        model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=1)
        cls.model_mnist = model


    @classmethod
    def tearDownClass(cls):
        k.clear_session()

    def setUp(self):
        # Set master seed
        model_zoo.master_seed(1234)


    def test_fit(self):
        logger.info('test_fit');
        labels = np.argmax(self.mnist[1][1], axis=1)
        classifier = KerasClassifier((0, 1), self.model_mnist, use_logits=False)
        acc = np.sum(np.argmax(classifier.predict(self.mnist[1][0]), axis=1) == labels) / NB_TEST
        logger.info('Accuracy: %.2f%%', (acc * 100))

        classifier.fit(self.mnist[0][0], self.mnist[0][1], batch_size=BATCH_SIZE, nb_epochs=1)
        acc2 = np.sum(np.argmax(classifier.predict(self.mnist[1][0]), axis=1) == labels) / NB_TEST
        logger.info('Accuracy: %.2f%%', (acc2 * 100))

        self.assertTrue(acc2 >= .9 * acc)

    def test_class_gradient(self):
        logger.info('test_class_gradient');
        (_, _), (x_test, _) = self.mnist
        classifier = KerasClassifier((0, 1), self.model_mnist)

        # Test all gradients label
        grads = classifier.class_gradient(x_test)
        self.assertTrue(np.array(grads.shape == (NB_TEST, 10, 28, 28, 1)).all())
        self.assertTrue(np.sum(grads) != 0)

        # Test 1 gradient label = 5
        grads = classifier.class_gradient(x_test, label=5)
        self.assertTrue(np.array(grads.shape == (NB_TEST, 1, 28, 28, 1)).all())
        self.assertTrue(np.sum(grads) != 0)

        # Test a set of gradients label = array
        label = np.random.randint(5, size=NB_TEST)
        grads = classifier.class_gradient(x_test, label=label)
        self.assertTrue(np.array(grads.shape == (NB_TEST, 1, 28, 28, 1)).all())
        self.assertTrue(np.sum(grads) != 0)

        grads = classifier.class_gradient(x_test, logits=True)
        self.assertTrue(np.array(grads.shape == (NB_TEST, 10, 28, 28, 1)).all())
        self.assertTrue(np.sum(grads) != 0)

    def test_loss_gradient(self):
        logger.info('test_loss_gradient');
        (_, _), (x_test, y_test) = self.mnist
        classifier = KerasClassifier((0, 1), self.model_mnist)

        # Test gradient
        grads = classifier.loss_gradient(x_test, y_test)

        self.assertTrue(np.array(grads.shape == (NB_TEST, 28, 28, 1)).all())
        self.assertTrue(np.sum(grads) != 0)

    def test_activations(self):
        logger.info('test_activations');
        (_, _), (x_test, _) = self.mnist;
        x_test = x_test[:NB_TEST]

        classifier = KerasClassifier((0, 1), model=self.model_mnist)
        self.assertEqual(len(classifier.layer_names), 5)

        layer_names = classifier.layer_names;
        for i, name in enumerate(layer_names):
            if 'dropout' not in name:
                act_i = classifier.get_activations(x_test, i)
                act_name = classifier.get_activations(x_test, name)
                self.assertAlmostEqual(np.sum(act_name - act_i), 0)
                logger.info('layer_names={}\t{}'.format(name, act_i.shape ));

        self.assertTrue(classifier.get_activations(x_test, 0).shape == (NB_TEST, 26, 26, 32))
        self.assertTrue(classifier.get_activations(x_test, 4).shape == (NB_TEST, 128))

    def test_layer_gradient(self):
        logger.info('test_layer_gradient');
        (x_train, _), (x_test, _) = self.mnist;

        classifier = KerasClassifier((0, 1), model=self.model_mnist)
        layer_names = classifier.layer_names

        for i, name in enumerate(layer_names):
            if 'dropout' not in name:
                grads = classifier.layer_gradient(x_test[0:10, :, :, :], i, x_test[10:20, :, :, :]);

    def test_save(self):
        logger.info('test_save');
        path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')), 'data');
        filename = 'model.h5'
        classifier = KerasClassifier((0, 1), model=self.model_mnist)
        classifier.save(filename, path=path)
        self.assertTrue(os.path.isfile(os.path.join(path, filename)))

        # Remove saved file
        os.remove(os.path.join(path, filename))


if __name__ == '__main__':
    unittest.main()