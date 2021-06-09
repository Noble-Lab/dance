from __future__ import absolute_import, division, print_function, unicode_literals
import os, sys, random, six
sys.path.append('..')

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import numpy as np

import keras
import keras.backend as k
from keras.models import load_model
import tensorflow as tf
from keras.engine.topology import InputLayer

from classifiers.classifier import Classifier


class KerasClassifier(Classifier):
    """
    Wrapper class for importing Keras models. The supported backends for Keras are TensorFlow and Theano.
    """
    def __init__(self, clip_values, model, use_logits=False, preprocessing=(0, 1), input_layer=0, output_layer=0):
        """
        Create a `Classifier` instance from a Keras model. Assumes the `model` passed as argument is compiled.

        :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
               for features.
        :type clip_values: `tuple`
        :param model: Keras model
        :type model: `keras.models.Model`
        :param use_logits: True if the output of the model are the logits.
        :type use_logits: `bool`
        :param preprocessing: Tuple of the form `(substractor, divider)` of floats or `np.ndarray` of values to be
               used for data preprocessing. The first value will be substracted from the input. The input will then
               be divided by the second one.
        :type preprocessing: `tuple`
        :param input_layer: Which layer to consider as the Input when the model has multple input layers.
        :type input_layer: `int`
        :param output_layer: Which layer to consider as the Output when the model has multiple output layers.
        :type output_layer: `int`

        """

        super(KerasClassifier, self).__init__(clip_values=clip_values, preprocessing=preprocessing)

        self._model = model
        if hasattr(model, 'inputs'):
            self._input = model.inputs[input_layer]
        else:
            self._input = model.input

        if hasattr(model, 'outputs'):
            self._output = model.outputs[output_layer]
        else:
            self._output = model.output

        _, self._nb_classes = k.int_shape(self._output)
        self._input_shape = k.int_shape(self._input)[1:]

        # Get predictions and loss function
        label_ph = k.placeholder(shape=(None,))
        if not use_logits:
            if k.backend() == 'tensorflow':
                preds, = self._output.op.inputs
                loss = k.sparse_categorical_crossentropy(label_ph, preds, from_logits=True)
            else:
                loss = k.sparse_categorical_crossentropy(label_ph, self._output, from_logits=use_logits)

                # Convert predictions to logits for consistency with the other cases
                eps = 10e-8
                preds = k.log(k.clip(self._output, eps, 1. - eps))
        else:
            preds = self._output
            loss = k.sparse_categorical_crossentropy(label_ph, self._output, from_logits=use_logits)
        if preds == self._input:  # recent Tensorflow version does not allow a model with an output same as the input.
            preds = k.identity(preds)
        loss_grads = k.gradients(loss, self._input)

        if k.backend() == 'tensorflow':
            loss_grads = loss_grads[0]
        elif k.backend() == 'cntk':
            raise NotImplementedError('Only TensorFlow and Theano support is provided for Keras.')

        # Set loss, grads and prediction functions
        self._preds_op = preds
        self._loss = k.function([self._input], [loss])
        self._loss_grads = k.function([self._input, label_ph], [loss_grads])
        self._preds = k.function([self._input], [preds])

        # Get the internal layer
        self._layer_names = self._get_layers()

    def loss_gradient(self, x, y):
        """
        Compute the gradient of the loss function w.r.t. `x`.

        :param x: Sample input with shape as expected by the model.
        :type x: `np.ndarray`
        :param y: Correct labels, one-vs-rest encoding.
        :type y: `np.ndarray`
        :return: Array of gradients of the same shape as `x`.
        :rtype: `np.ndarray`
        """
        x_ = self._apply_processing(x)
        grads = self._loss_grads([x_, np.argmax(y, axis=1)])[0]
        grads = self._apply_processing_gradient(grads)
        assert grads.shape == x_.shape

        return grads

    def class_gradient(self, x, label=None, logits=False):
        """
        Compute per-class derivatives w.r.t. `x`.

        :param x: Sample input with shape as expected by the model.
        :type x: `np.ndarray`
        :param label: Index of a specific per-class derivative. If an integer is provided, the gradient of that class
                      output is computed for all samples. If multiple values as provided, the first dimension should
                      match the batch size of `x`, and each value will be used as target for its corresponding sample in
                      `x`. If `None`, then gradients for all classes will be computed for each sample.
        :type label: `int` or `list`
        :param logits: `True` if the prediction should be done at the logits layer.
        :type logits: `bool`
        :return: Array of gradients of input features w.r.t. each class in the form
                 `(batch_size, nb_classes, input_shape)` when computing for all classes, otherwise shape becomes
                 `(batch_size, 1, input_shape)` when `label` parameter is specified.
        :rtype: `np.ndarray`
        """
        # Check value of label for computing gradients
        if not (label is None or (isinstance(label, (int, np.integer)) and label in range(self.nb_classes))
                or (isinstance(label, np.ndarray) and len(label.shape) == 1 and (label < self.nb_classes).all()
                    and label.shape[0] == x.shape[0])):
            raise ValueError('Label %s is out of range.' % str(label))

        self._init_class_grads(label=label, logits=logits)

        x_ = self._apply_processing(x)

        if label is None:
            # Compute the gradients w.r.t. all classes
            if logits:
                grads = np.swapaxes(np.array(self._class_grads_logits([x_])), 0, 1)
            else:
                grads = np.swapaxes(np.array(self._class_grads([x_])), 0, 1)

            grads = self._apply_processing_gradient(grads)

        elif isinstance(label, (int, np.integer)):
            # Compute the gradients only w.r.t. the provided label
            if logits:
                grads = np.swapaxes(np.array(self._class_grads_logits_idx[label]([x_])), 0, 1)
            else:
                grads = np.swapaxes(np.array(self._class_grads_idx[label]([x_])), 0, 1)

            grads = self._apply_processing_gradient(grads)
            assert grads.shape == (x_.shape[0], 1) + self.input_shape

        else:
            # For each sample, compute the gradients w.r.t. the indicated target class (possibly distinct)
            unique_label = list(np.unique(label))
            if logits:
                grads = np.array([self._class_grads_logits_idx[l]([x_]) for l in unique_label])
            else:
                grads = np.array([self._class_grads_idx[l]([x_]) for l in unique_label])
            grads = np.swapaxes(np.squeeze(grads, axis=1), 0, 1)
            lst = [unique_label.index(i) for i in label]
            grads = np.expand_dims(grads[np.arange(len(grads)), lst], axis=1)

            grads = self._apply_processing_gradient(grads)

        return grads

    def layer_gradient(self, x, layer, layer_ref):
        """
        Compute the gradient of the intermediate layer for input `x`. `layer` is specified by layer index (between 0 and
        `nb_layers - 1`) or by name. The number of layers can be determined by counting the results returned by
        calling `layer_names`.

        :param x: Input for computing the layer gradient.
        :type x: `np.ndarray`
        :param layer: Layer for computing the layer gradient
        :type layer: `int` or `str`
        :param layer_ref: The intermediate representation of the reference input.
        :type layer_ref: `np.ndarray`
        :return: Array of gradients of input features w.r.t. the intermediate layer and the reference input in the form
                 `(batch_size, input_shape)`, where the first dimension is the batch size corresponding to `x`.
        :rtype: `np.ndarray`
        """
        k.set_learning_phase(0)

        if isinstance(layer, six.string_types):
            if layer not in self._layer_names:
                raise ValueError('Layer name %s is not part of the graph.' % layer)
            layer_name = layer
        elif isinstance(layer, int):
            if layer < 0 or layer >= len(self._layer_names):
                raise ValueError('Layer index %d is outside of range (0 to %d included).'
                                 % (layer, len(self._layer_names) - 1))
            layer_name = self._layer_names[layer]
        else:
            raise TypeError('Layer must be of type `str` or `int`.')

        layer_output = self._model.get_layer(layer_name).output

        if x.shape == self.input_shape:
            x = np.expand_dims(x, 0)

        loss_grads = k.gradients(k.sum(k.square(layer_output - layer_ref)), self._input)[0]
        _loss_grads = k.function([self._input], [loss_grads])
        grads = _loss_grads([x])[0]

        assert grads.shape == x.shape
        return grads

    def layer_loss(self, x, layer, layer_ref):
        """
        Compute the difference between the intermediate layer for input `x` and 'x_ref'. `layer` is specified by layer index (between 0 and
        `nb_layers - 1`) or by name. The number of layers can be determined by counting the results returned by
        calling `layer_names`.

        :param x: Input for computing the layer gradient.
        :type x: `np.ndarray`
        :param layer: Layer for computing the layer gradient
        :type layer: `int` or `str`
        :param layer_ref: The intermediate representation of the reference input.
        :type layer_ref: `np.ndarray`
        :return: Array of MSE between two intermediate layer representations in the form (batch_size,).
        :rtype: `np.ndarray`
        """

        layer_x = self.get_activations(x, layer)
        loss = np.sum(np.square(layer_x - layer_ref), axis=tuple(range(1, len(layer_x.shape))))
        assert len(loss) == x.shape[0]
        return loss

    def predict(self, x, logits=False, batch_size=128):
        """
        Perform prediction for a batch of inputs.

        :param x: Test set.
        :type x: `np.ndarray`
        :param logits: `True` if the prediction should be done at the logits layer.
        :type logits: `bool`
        :param batch_size: Size of batches.
        :type batch_size: `int`
        :return: Array of predictions of shape `(nb_inputs, self.nb_classes)`.
        :rtype: `np.ndarray`
        """
        k.set_learning_phase(0)

        x_ = self._apply_processing(x)

        # Run predictions with batching
        preds = np.zeros((x_.shape[0], self.nb_classes), dtype=np.float32)
        for b in range(int(np.ceil(x_.shape[0] / float(batch_size)))):
            begin, end = b * batch_size, min((b + 1) * batch_size, x_.shape[0])
            preds[begin:end] = self._preds([x_[begin:end]])[0]

            if not logits:
                exp = np.exp(preds[begin:end] - np.max(preds[begin:end], axis=1, keepdims=True))
                preds[begin:end] = exp / np.sum(exp, axis=1, keepdims=True)

        return preds

    def fit(self, x, y, batch_size=128, nb_epochs=20):
        """
        Fit the classifier on the training set `(x, y)`.

        :param x: Training data.
        :type x: `np.ndarray`
        :param y: Labels, one-vs-rest encoding.
        :type y: `np.ndarray`
        :param batch_size: Size of batches.
        :type batch_size: `int`
        :param nb_epochs: Number of epochs to use for trainings.
        :type nb_epochs: `int`
        :return: `None`
        """
        k.set_learning_phase(1)

        # Apply preprocessing and defences
        x_ = self._apply_processing(x)
        y_ = y

        gen = generator_fit(x_, y_, batch_size)
        self._model.fit_generator(gen, steps_per_epoch=x_.shape[0] / batch_size, epochs=nb_epochs)

    def attack_saliency_gradient(self, method, x, label=None, k_top=100, target_map=None,
                                 logits=False, positive_only=False):
        """
        Compute the perturbation of attack on gradient saliency method,
        :param method: attack method: 'top_k', 'targeted', or 'mass-center'
        :type method: `string'
        :param x: Sample input with shape as expected by the model.
        :type x: `numpy.array`
        :param label: Index of a specific per-class derivative.
        :type label: `numpy.array`
        :param k_top: Top K features for the topK attack.
        :type k_top: `int`
        :param target_map: Target features for the targeted attack.
        :type target_map: `numpy.array`
        :param logits: Using the logits as outputs.
        :type logits: `bool`
        :param positive_only: Attacking only the positive features.
        :type positive_only: `bool`
        :return: Array of perturbation
        :rtype: `numpy.array`
        """
        k.set_learning_phase(0)

        if k.backend() != 'tensorflow':
            raise ValueError('Unexpected backend of Keras for attack.')

        if label is None:
            raise ValueError('Missing label information.')
        if isinstance(label, int) == None:
            raise ValueError('Unexpected label type, only support int.')


        ## Compute the normalized gradient saliency for the target label
        b = x.shape[0]
        w = self._input.shape[1].value
        h = self._input.shape[2].value
        c = self._input.shape[3].value

        if logits == True:
            class_grads_logits = [k.gradients(self._preds_op[:, label[i]], self._input)[0][i] for i in xrange(label.shape[0])]
            class_grads_logits = k.stack(class_grads_logits)
        else:
            class_grads_logits = [k.gradients(k.softmax(self._preds_op)[:, label[i]], self._input)[0][i] for i in xrange(label.shape[0])]
            class_grads_logits = k.stack(class_grads_logits)
        if positive_only == True:
            saliency_unnormalized = tf.reduce_sum(tf.nn.relu(class_grads_logits), -1)
            self._saliency = saliency_unnormalized/(tf.reduce_sum(saliency_unnormalized, axis=(1,2))+tf.keras.backend.epsilon())[:,None,None]
        else:
            saliency_unnormalized = tf.reduce_sum(tf.abs(class_grads_logits), -1)
            self._saliency = saliency_unnormalized/(tf.reduce_sum(saliency_unnormalized, axis=(1,2))+tf.keras.backend.epsilon())[:,None,None]

        self._saliency_flatten = tf.reshape(self._saliency, [b, w*h])
        top_val, self._top_idx = tf.nn.top_k(self._saliency_flatten, k_top)

        ## Compute the center mass of x: C(x)
        y_mesh, x_mesh = np.meshgrid(np.arange(h), np.arange(w))
        self._mass_center = tf.stack([tf.reduce_sum(self._saliency * x_mesh, axis=(1,2)) / (w * h),
                                      tf.reduce_sum(self._saliency * y_mesh, axis=(1,2)) / (w * h)])
        self._mass_center = tf.transpose(self._mass_center)

        saliency_func = k.function([self._input], [self._saliency, self._saliency_flatten, self._top_idx, self._mass_center])
        saliency, saliency_flatten, topK, mass_center = saliency_func([x])

        ## get the gradient of each attack loss w.r.t. input
        elem1 = np.argsort(np.reshape(saliency, [b, w * h]), axis=-1)[:,-k_top:]
        elements1 = np.zeros((b, w * h))
        for ii in xrange(b):
            elements1[ii,elem1[ii]]=1
        topK_loss = -tf.reduce_sum(self._saliency_flatten *elements1, axis=1)
        self._topK_direction = tf.stack([tf.gradients(topK_loss[ll], self._input)[0][ll] for ll in xrange(b)])

        mass_center_loss = tf.reduce_sum((self._mass_center - mass_center) ** 2, axis=1)
        self._mass_center_direction = tf.stack([tf.gradients(mass_center_loss[ll], self._input)[0][ll] for ll in xrange(b)])

        if target_map is not None:
            target_loss = tf.reduce_sum((self._saliency[:,:,:,None] * target_map), axis=(1,2,3))
            self._target_direction = tf.stack([tf.gradients(target_loss[ll], self._input)[0][ll] for ll in xrange(b)])

        if method == "topK":
            perturbation_func = k.function([self._input], [self._topK_direction])
        elif method == "mass_center":
            perturbation_func = k.function([self._input], [self._mass_center_direction])
        elif method == "target":
            if target_map is None:
                raise ValueError("No target region determined!")
            else:
                perturbation_func = k.function([self._input], [self._target_direction])

        return saliency, saliency_flatten, topK, mass_center, saliency_func, perturbation_func

    def attack_saliency_integratedgrad(self, method, x, label, num_steps = 25, k_top = 100,
                                       target_map = None, reference_image = None,
                                       logits = False, positive_only = False):
        """
        Compute the perturbation of attack on the Intergrated gradient saliency method,
        :param method: attack method: 'top_k', 'targeted', or 'mass-center'
        :type method: `string'
        :param x: Sample input with shape as expected by the model.
        :type x: `numpy.array`
        :param label: Index of a specific per-class derivative.
        :type label: `numpy.array`
        :param k_top: Top K features for the topK attack.
        :type k_top: `int`
        :param target_map: Target features for the targeted attack.
        :type target_map: `numpy.array`
        :param reference_image: Starting point of the intergrated gradient attack.
        :type reference_image: `numpy.array`
        :param logits: Using the logits as outputs.
        :type logits: `bool`
        :param positive_only: Attacking only the positive features.
        :type positive_only: `bool`
        :return: Array of perturbation
        :rtype: `numpy.array`
        """
        k.set_learning_phase(0)

        if k.backend() != 'tensorflow':
            raise ValueError('Unexpected backend of Keras for attack.')

        if len(self._output.shape) == 2:
            nb_outputs = self._output.shape[1]
        else:
            raise ValueError('Unexpected output shape for classification in Keras model.')

        if label is None:
            raise ValueError('Missing label information.')
        if isinstance(label, int) == None:
            raise ValueError('Unexpected label type, only support int.')

        # Compute the normalized integrated gradient saliency for the target label
        b = x.shape[0]
        w = self._input.shape[1].value
        h = self._input.shape[2].value
        c = self._input.shape[3].value
        self._reference_image = tf.placeholder(tf.float32, shape=(w,h,c))

        if logits == True:
            class_grads_logits = [tf.reduce_sum(tf.gradients(self._preds_op[:, label[i]], self._input)[0][i*num_steps:(i+1)*num_steps], axis=0)
                                  for i in xrange(label.shape[0])]
        else:
            class_grads_logits = [tf.reduce_sum(tf.gradients(k.softmax(self._preds_op)[:, label[i]], self._input)[0][i*num_steps:(i+1)*num_steps], axis=0)
                                  for i in xrange(label.shape[0])]
        inte_map_logits = [class_grads_logits[i] * (self._input[(i+1)*num_steps-1] -self._reference_image)
                           for i in xrange(label.shape[0])]
        inte_map_logits = tf.stack(inte_map_logits)

        if positive_only == True:
            saliency_unnormalized = tf.reduce_sum(tf.nn.relu(inte_map_logits), -1)
            self._saliency = saliency_unnormalized/(tf.reduce_sum(saliency_unnormalized, axis=(1,2))+tf.keras.backend.epsilon())[:,None,None]
        else:
            saliency_unnormalized = tf.reduce_sum(tf.abs(inte_map_logits), -1)
            self._saliency = saliency_unnormalized/(tf.reduce_sum(saliency_unnormalized, axis=(1,2))+tf.keras.backend.epsilon())[:,None,None]

        self._saliency_flatten = tf.reshape(self._saliency, [b, w*h])

        # Compute the top K features and the center mass of x: C(x)
        top_val, self._top_idx = tf.nn.top_k(self._saliency_flatten, k_top)
        y_mesh, x_mesh = np.meshgrid(np.arange(h), np.arange(w))
        self._mass_center = tf.stack([tf.reduce_sum(self._saliency * x_mesh, axis=(1,2)) / (w * h),
                                      tf.reduce_sum(self._saliency * y_mesh, axis=(1,2)) / (w * h)])
        self._mass_center = tf.transpose(self._mass_center)

        # Create a set of images
        counterfactuals = []
        for ii in xrange(b):
            ref_subtracted = x[ii,] - reference_image
            counterfactuals_tmp = np.array([(float(i + 1) / num_steps) * ref_subtracted + reference_image
                                            for i in range(num_steps)])
            counterfactuals.extend(counterfactuals_tmp)

        counterfactuals = np.array(counterfactuals)

        ## get the saliency map and the top k faetures for the counterfactuals
        saliency_func = k.function([self._input, self._reference_image], [self._saliency, self._saliency_flatten,
                                                                        self._top_idx, self._mass_center])
        saliency, saliency_flatten, topK, mass_center = saliency_func([counterfactuals, reference_image])

        ## get the gradient of each attack loss w.r.t. input

        elem1 = np.argsort(np.reshape(saliency, [b, w * h]), axis=-1)[:,-k_top:]
        elements1 = np.zeros((b, w * h))
        for ii in xrange(b):
            elements1[ii,elem1[ii]]=1
        topK_loss = -tf.reduce_sum(self._saliency_flatten *elements1, axis=1)
        self._topK_direction = tf.stack([tf.gradients(topK_loss[ll], self._input)[0][ll*num_steps:(ll+1)*num_steps] for ll in xrange(b)])

        mass_center_loss = tf.reduce_sum((self._mass_center - mass_center) ** 2, axis=1)
        self._mass_center_direction = tf.stack([tf.gradients(mass_center_loss[ll], self._input)[0][ll*num_steps:(ll+1)*num_steps] for ll in xrange(b)])

        if target_map is not None:
            target_loss = tf.reduce_sum((self._saliency[:,:,:,None] * target_map), axis=(1,2,3))
            self._target_direction = tf.stack([tf.gradients(target_loss[ll], self._input)[0][ll*num_steps:(ll+1)*num_steps] for ll in xrange(b)])

        if method == "topK":
            perturbation_func = k.function([self._input, self._reference_image], [self._topK_direction])
        elif method == "mass_center":
            perturbation_func = k.function([self._input, self._reference_image], [self._mass_center_direction])
        elif method == "target":
            if target_map is None:
                raise ValueError("No target region determined!")
            else:
                perturbation_func = k.function([self._input, self._reference_image], [self._target_direction])
        return saliency, saliency_flatten, topK, mass_center, saliency_func, perturbation_func

    def attack_saliency_smoothgrad(self, method, x, label, k_top, target_map, stdev_spread=0.15, nsamples=25,
                                   logits = False, positive_only = False):
        """
        Compute the perturbation of attack on the Integrated gradient saliency method,
        :param method: attack method: 'top_k', 'targeted', or 'mass-center'
        :type method: `string'
        :param x: Sample input with shape as expected by the model.
        :type x: `numpy.array`
        :param label: Index of a specific per-class derivative.
        :type label: `numpy.array`
        :param k_top: Top K features for the topK attack.
        :type k_top: `int`
        :param target_map: Target features for the targeted attack.
        :type target_map: `numpy.array`
        :param reference_image: Starting point of the integrated gradient attack.
        :type reference_image: `numpy.array`
        :return: Array of perturbation
        :rtype: `numpy.array`
        :param stdev_spread: Amount of noise to add to the input, as fraction of the
                    total spread (x_max - x_min).
        :type stdev_spread: `float`
        :param nsamples: Number of samples to average across to get the smooth gradient.
        :type nsamples: `int`
        :param logits: Using the logits as outputs.
        :type logits: `bool`
        :param positive_only: Attacking only the positive features.
        :type positive_only: `bool`
        :return: Array of perturbation
        :rtype: `numpy.array`
        """
        k.set_learning_phase(0)

        if k.backend() != 'tensorflow':
            raise ValueError('Unexpected backend of Keras for attack.')

        if label is None:
            raise ValueError('Missing label information.')
        if isinstance(label, int) == None:
            raise ValueError('Unexpected label type, only support int.')

        # Compute the normalized integrated gradient saliency for the target label
        b = x.shape[0]
        w = self._input.shape[1].value
        h = self._input.shape[2].value
        c = self._input.shape[3].value

        if logits == True:
            class_grads_logits = [tf.reduce_mean(tf.gradients(self._preds_op[:, label[i]], self._input)[0][i*nsamples:(i+1)*nsamples], axis=0)
                                  for i in xrange(label.shape[0])]
        else:
            class_grads_logits = [tf.reduce_mean(tf.gradients(k.softmax(self._preds_op)[:, label[i]], self._input)[0][i*nsamples:(i+1)*nsamples], axis=0)
                                  for i in xrange(label.shape[0])]
        class_grads_logits = tf.stack(class_grads_logits)

        if positive_only == True:
            saliency_unnormalized = tf.reduce_sum(tf.nn.relu(class_grads_logits), -1)
            self._saliency = saliency_unnormalized/(tf.reduce_sum(saliency_unnormalized, axis=(1,2))+tf.keras.backend.epsilon())[:,None,None]
        else:
            saliency_unnormalized = tf.reduce_sum(tf.abs(class_grads_logits), -1)
            self._saliency = saliency_unnormalized/(tf.reduce_sum(saliency_unnormalized, axis=(1,2))+tf.keras.backend.epsilon())[:,None,None]

        self._saliency_flatten = tf.reshape(self._saliency, [b, w*h])
        top_val, self._top_idx = tf.nn.top_k(self._saliency_flatten, k_top)

        ## Compute the center mass of x: C(x)
        y_mesh, x_mesh = np.meshgrid(np.arange(h), np.arange(w))
        self._mass_center = tf.stack([tf.reduce_sum(self._saliency * x_mesh, axis=(1,2)) / (w * h),
                                      tf.reduce_sum(self._saliency * y_mesh, axis=(1,2)) / (w * h)])
        self._mass_center = tf.transpose(self._mass_center)

        x_plus_noise = []
        for ii in xrange(b):
            stdev = stdev_spread * (np.max(x[ii,]) - np.min(x[ii,]))
            noise = np.random.normal(0, stdev, (nsamples, w, h, c))
            x_plus_noise_tmp = np.repeat(np.expand_dims(x[ii,], axis = 0), nsamples, axis=0) + noise
            x_plus_noise.extend(x_plus_noise_tmp)
        x_plus_noise = np.array(x_plus_noise)

        ## get the saliency map and the top k faetures for the counterfactuals
        saliency_func = k.function([self._input], [self._saliency, self._saliency_flatten, self._top_idx, self._mass_center])
        saliency, saliency_flatten, topK, mass_center = saliency_func([x_plus_noise])

        ## get the gradient of each attack loss w.r.t. input
        elem1 = np.argsort(np.reshape(saliency, [b, w * h]), axis=-1)[:,-k_top:]
        elements1 = np.zeros((b, w * h))
        for ii in xrange(b):
            elements1[ii,elem1[ii]]=1
        topK_loss = -tf.reduce_sum(self._saliency_flatten *elements1, axis=1)
        self._topK_direction = tf.stack([tf.gradients(topK_loss[ll], self._input)[0][ll*nsamples:(ll+1)*nsamples] for ll in xrange(b)])

        mass_center_loss = tf.reduce_sum((self._mass_center - mass_center) ** 2, axis=1)
        self._mass_center_direction = tf.stack([tf.gradients(mass_center_loss[ll], self._input)[0][ll*nsamples:(ll+1)*nsamples] for ll in xrange(b)])

        if target_map is not None:
            target_loss = tf.reduce_sum((self._saliency[:,:,:,None] * target_map), axis=(1,2,3))
            self._target_direction = tf.stack([tf.gradients(target_loss[ll], self._input)[0][ll*nsamples:(ll+1)*nsamples] for ll in xrange(b)])

        if method == "topK":
            perturbation_func = k.function([self._input], [self._topK_direction])
        elif method == "mass_center":
            perturbation_func = k.function([self._input], [self._mass_center_direction])
        elif method == "target":
            if target_map is None:
                raise ValueError("No target region determined!")
            else:
                perturbation_func = k.function([self._input], [self._target_direction])
        else:
            print("Using random perturbation")
            perturbation_all = np.random.normal(size=(b*nsamples, w, h, c))
        return saliency, saliency_flatten, topK, mass_center, saliency_func, perturbation_func

    def attack_saliency_gradcam(self, method, x, label, k_top, target_map, layer, logits = False, positive_only = False):
        """
        Compute the perturbation of attack on the Integrated gradient saliency method,
        :param method: attack method: 'top_k', 'targeted', or 'mass-center'
        :type method: `string'
        :param x: Sample input with shape as expected by the model.
        :type x: `np.ndarray`
        :param label: Index of a specific per-class derivative.
        :type label: `np.ndarray`
        :param k_top: Top K features for the topK attack.
        :type k_top: `int`
        :param target_map: Target features for the targeted attack.
        :type target_map: `numpy.array`
        :param layer: Specifying the last convolutional layer.
        :type layer: `int` or 'string'
        :param logits: Using the logits as outputs.
        :type logits: `bool`
        :param positive_only: Attacking only the positive features.
        :type positive_only: `bool`
        :return: Array of perturbation
        :rtype: 'numpy.array'
        """
        if label is None:
            raise ValueError('Missing label information.')

        if isinstance(layer, six.string_types):
            if layer not in self._layer_names:
                raise ValueError('Layer name %s is not part of the graph.' % layer)
            layer_name = layer
            layer_output = self._model.get_layer(layer_name).output
        elif isinstance(layer, int):
            if layer < 0 or layer >= len(self._layer_names):
                raise ValueError('Layer index %d is outside of range (0 to %d included).'
                                 % (layer, len(self._layer_names) - 1))
            layer_name = self._layer_names[layer]
            layer_output = self._model.get_layer(layer_name).output
        elif isinstance(layer, tf.Tensor):
            layer_output = layer
        else:
            raise TypeError("Layer must be of type `str` or `int`. Received '%s'", layer)

        # Check value of label for computing gradients
        if not (isinstance(label, (int, np.ndarray)) and np.max(label) in range(self.nb_classes)):
            raise ValueError('Only support int lable and current label %s is out of range.' % label)

        # Compute the normalized integrated gradient saliency for the target label
        b = x.shape[0]
        w = layer_output.shape[1].value
        h = layer_output.shape[2].value
        c = layer_output.shape[3].value

        # Compute the gradient and activations
        if logits == True:
            self._logit_gradcam_grads = [tf.gradients(self._preds_op[:, label[l]], layer_output)[0][l]
                                         for l in xrange(label.shape[0])]
        else:
            self._logit_gradcam_grads = [tf.gradients(k.softmax(self._preds_op)[:, label[l]], layer_output)[0][l]
                                         for l in xrange(label.shape[0])]
        self._logit_gradcam_grads = tf.stack(self._logit_gradcam_grads)
        weights = tf.reduce_mean(self._logit_gradcam_grads, axis=(1,2))

        grad_cams = tf.reduce_sum(weights[:, None, None, :] * layer_output, axis = -1)

        if positive_only == True:
            saliency_unnormalized = tf.nn.relu(grad_cams)
            self._saliency = saliency_unnormalized/(tf.reduce_sum(saliency_unnormalized, axis=(1,2))+tf.keras.backend.epsilon())[:,None,None]
        else:
            saliency_unnormalized = tf.abs(grad_cams)
            self._saliency = saliency_unnormalized/(tf.reduce_sum(saliency_unnormalized, axis=(1,2))+tf.keras.backend.epsilon())[:,None,None]

        self._saliency_flatten = tf.reshape(self._saliency, [b, w*h])

        # Compute the top K features and the center mass of x: C(x)
        top_val, self._top_idx = tf.nn.top_k(self._saliency_flatten, k_top)
        y_mesh, x_mesh = np.meshgrid(np.arange(h), np.arange(w))
        self._mass_center = tf.stack([tf.reduce_sum(self._saliency * x_mesh, axis=(1,2)) / (w * h),
                                      tf.reduce_sum(self._saliency * y_mesh, axis=(1,2)) / (w * h)])
        self._mass_center = tf.transpose(self._mass_center)

        ## get the saliency map and the top k faetures for the counterfactuals
        saliency_func = k.function(inputs=[self._input], outputs=[self._saliency, self._saliency_flatten,
                                                                      self._top_idx, self._mass_center])
        saliency, saliency_flatten, topK, mass_center = saliency_func([x])

        ## get the gradient of each attack loss w.r.t. input
        elem1 = np.argsort(np.reshape(saliency, [b, w * h]), axis=-1)[:,-k_top:]
        elements1 = np.zeros((b, w * h))
        for ii in xrange(b):
            elements1[ii,elem1[ii]]=1
        topK_loss = -tf.reduce_sum(self._saliency_flatten *elements1, axis=1)
        self._topK_direction = tf.stack([tf.gradients(topK_loss[ll], self._input)[0][ll] for ll in xrange(b)])

        mass_center_loss = tf.reduce_sum((self._mass_center - mass_center) ** 2, axis=1)
        self._mass_center_direction = tf.stack([tf.gradients(mass_center_loss[ll], self._input)[0][ll] for ll in xrange(b)])

        if target_map is not None:
            target_loss = tf.reduce_sum((self._saliency[:,:,:,None] * target_map), axis=(1,2,3))
            self._target_direction = tf.stack([tf.gradients(target_loss[ll], self._input)[0][ll] for ll in xrange(b)])

        if method == "topK":
            perturbation_func = k.function(inputs=[self._input], outputs=[self._topK_direction])
        elif method == "mass_center":
            perturbation_func = k.function(inputs=[self._input], outputs=[self._mass_center_direction])
        elif method == "target":
            if target_map is None:
                raise ValueError("No target region determined!")
            else:
                perturbation_func = k.function(inputs=[self._input], outputs=[self._target_direction])
        return saliency, saliency_flatten, topK, mass_center, saliency_func, perturbation_func

    def bruteforce_knockoffs(self, x, y, saliency_gen, layer=0, tol=1e-3, step_size=2e-3):
        """
        Generate the pixel-swappable knockoff for input `x`, by using brute-force searching pixel by pixel.
        :param x: Input for computing the layer gradient.
        :type x: `np.ndarray`
        :param y: Labels for generating saliency maps, not for generating knockoffs.
        :type y: `np.ndarray`
        :param saliency_gen: The saliency generator for knockoffs
        :type saliency_gen: `Saliency`
        :param layer: Layer for computing the latent representation
        :type layer: `int` or `str`
        :param tol: Tolarence for loss difference
        :type tol: `float`
        :param step_size: Unit step_size for searching
        :type step_size: `int`
        :return: Two arrays holding the knockoff upper and lower bounds, and two arrays holding the saliency upper and lower bounds
        :rtype: `np.ndarray quadruple`
        """
        raise NotImplementedError

    def pixel_knockoffs(self, x, y, saliency_gen, layer=0, eps=2e3, C=5e-7, tol=1e-3, max_iter=20, multi_size=10, repeat=5):
        """
        Generate the pixel-swappable knockoff for input `x`.

        :param x: Input for computing the layer gradient.
        :type x: `np.ndarray`
        :param y: Labels for generating saliency maps, not for generating knockoffs.
        :type y: `np.ndarray`
        :param saliency_gen: The saliency generator for knockoffs
        :type saliency_gen: `Saliency`
        :param layer: Layer for computing the latent representation
        :type layer: `int` or `str`
        :param eps: Step size for gradient descent
        :type eps: `float`
        :param C: Coefficient of the max(x-x_ref, 0) term
        :type C: `float`
        :param tol: Tolarence for loss difference
        :type tol: `float`
        :param max_iter: Maximum iteration for gradient descent
        :type max_iter: `int`
        :param multi_size: The number of pixels optimized at once
        :type multi_size: `int`
        :param repeat: The number of repeating epoches
        :type repeat: `int`
        :return: Two arrays holding the knockoff upper and lower bounds, and two arrays holding the saliency upper and lower bounds
        :rtype: `np.ndarray quadruple`
        """
        raise NotImplementedError

    def patch_knockoffs(self, x, y, saliency_gen, layer=0, eps=2e3, C=5e-7, tol=1e-3, max_iter=20, patch_size=2, stride=1, multi_size=5, repeat=5):
        """
        Generate the patch-swappable knockoff for input `x`.

        :param x: Input for computing the layer gradient.
        :type x: `np.ndarray`
        :param y: Labels for generating saliency maps, not for generating knockoffs.
        :type y: `np.ndarray`
        :param saliency_gen: The saliency generator for knockoffs
        :type saliency_gen: `Saliency`
        :param layer: Layer for computing the latent representation
        :type layer: `int` or `str`
        :param eps: Step size for gradient descent
        :type eps: `float`
        :param C: Coefficient of the max(x-x_ref, 0) term
        :type C: `float`
        :param tol: Tolarence for loss difference
        :type tol: `float`
        :param max_iter: Maximum iteration for gradient descent
        :type max_iter: `int`
        :param patch_size: The width/height of the patch
        :type patch_size: `int`
        :param stride: The stride length of the patch, which is no larger than the patch_size
        :type stride: `int`
        :param multi_size: The number of pixels optimized at once
        :type multi_size: `int`
        :param repeat: The number of repeating epoches
        :type repeat: `int`
        :return: Two arrays holding the knockoff upper and lower bounds, and two arrays holding the saliency upper and lower bounds
        :rtype: `np.ndarray quadruple`
        """
        raise NotImplementedError

    def grad_cam_gradient(self, x, label, layer):
        """
        Compute pre-softmax derivatives w.r.t. activation of the last convolutional layer.
        :param x: Sample input with shape as expected by the model.
        :type x: `np.ndarray`
        :param label: Index of a specific class
        :type label: `int` or `list`
        :param layer: Specifying the last convolutional layer.
        :type layer: `int` or 'string'
        :return: Array of gradients of input features w.r.t. each class in the form
                 `(batch_size, nb_classes, input_shape)` when computing for all classes, otherwise shape becomes
                 `(batch_size, 1, input_shape)` when `label` parameter is specified.
        :rtype: `np.ndarray`
        """
        """
        Compute pre-softmax derivatives w.r.t. activation of the last convolutional layer.
        :param x: Sample input with shape as expected by the model.
        :type x: `np.ndarray`
        :param label: Index of a specific class
        :type label: `int` or `np.ndarray`
        :param layer: Specifying the last convolutional layer.
        :type layer: `int` or 'string' or 'tf.Tensor'
        :return: Array of gradients of input features w.r.t. each class in the form
                 `(batch_size, nb_classes, input_shape)` when computing for all classes, otherwise shape becomes
                 `(batch_size, 1, input_shape)` when `label` parameter is specified.
        :rtype: `np.ndarray`
        """
        # get the activations of the specific layer
        x_ = self._apply_processing(x)

        if isinstance(layer, six.string_types):
            if layer not in self._layer_names:
                raise ValueError('Layer name %s is not part of the graph.' % layer)
            layer_name = layer
            layer_output = self._model.get_layer(layer_name).output
        elif isinstance(layer, int):
            if layer < 0 or layer >= len(self._layer_names):
                raise ValueError('Layer index %d is outside of range (0 to %d included).'
                                 % (layer, len(self._layer_names) - 1))
            layer_name = self._layer_names[layer]
            layer_output = self._model.get_layer(layer_name).output
        elif isinstance(layer, tf.Tensor):
            layer_output = layer
        else:
            raise TypeError("Layer must be of type `str` or `int`. Received '%s'", layer)

        # Check value of label for computing gradients
        if not (isinstance(label, (int, np.ndarray)) and np.max(label) in range(self.nb_classes)):
            raise ValueError('Only support int lable and current label %s is out of range.' % label)

        # Compute the gradient and activations
        self._logit_gradcam_grads = [k.gradients(self._preds_op[:, label[i]], layer_output)[0][i] for i in range(label.shape[0])]
        self._logit_gradcam_grads = k.stack(self._logit_gradcam_grads)
        grads, activations = k.function(inputs=self._input, outputs=[self._logit_gradcam_grads, layer_output])([x_])
        return grads, activations

    @property
    def layer_names(self):
        """
        Return the hidden layers in the model, if applicable.

        :return: The hidden layers in the model, input and output layers excluded.
        :rtype: `list`

        .. warning:: `layer_names` tries to infer the internal structure of the model.
                     This feature comes with no guarantees on the correctness of the result.
                     The intended order of the layers tries to match their order in the model, but this is not
                     guaranteed either.
        """
        return self._layer_names

    def get_activations(self, x, layer):
        """
        Return the output of the specified layer for input `x`. `layer` is specified by layer index (between 0 and
        `nb_layers - 1`) or by name. The number of layers can be determined by counting the results returned by
        calling `layer_names`.

        :param x: Input for computing the activations.
        :type x: `np.ndarray`
        :param layer: Layer for computing the activations
        :type layer: `int` or `str`
        :return: The output of `layer`, where the first dimension is the batch size corresponding to `x`.
        :rtype: `np.ndarray`
        """
        k.set_learning_phase(0)

        if isinstance(layer, six.string_types):
            if layer not in self._layer_names:
                raise ValueError('Layer name %s is not part of the graph.' % layer)
            layer_name = layer
        elif isinstance(layer, int):
            if layer < 0 or layer >= len(self._layer_names):
                raise ValueError('Layer index %d is outside of range (0 to %d included).'
                                 % (layer, len(self._layer_names) - 1))
            layer_name = self._layer_names[layer]
        else:
            raise TypeError('Layer must be of type `str` or `int`.')

        layer_output = self._model.get_layer(layer_name).output
        output_func = k.function([self._input], [layer_output])

        # Apply preprocessing and defences
        if x.shape == self.input_shape:
            x = np.expand_dims(x, 0)

        return output_func([x])[0]

    def _init_class_grads(self, label=None, logits=False):
        k.set_learning_phase(0)

        if len(self._output.shape) == 2:
            nb_outputs = self._output.shape[1]
        else:
            raise ValueError('Unexpected output shape for classification in Keras model.')

        if label is None:
            logger.debug('Computing class gradients for all %i classes.', self.nb_classes)
            if logits:
                if not hasattr(self, '_class_grads_logits'):
                    class_grads_logits = [k.gradients(self._preds_op[:, i], self._input)[0]
                                          for i in range(nb_outputs)]
                    self._class_grads_logits = k.function([self._input], class_grads_logits)
            else:
                if not hasattr(self, '_class_grads'):
                    class_grads = [k.gradients(k.softmax(self._preds_op)[:, i], self._input)[0]
                                   for i in range(nb_outputs)]
                    self._class_grads = k.function([self._input], class_grads)

        else:
            if isinstance(label, int):
                unique_labels = [label]
                logger.debug('Computing class gradients for class %i.', label)
            else:
                unique_labels = np.unique(label)
                logger.debug('Computing class gradients for classes %s.', str(unique_labels))

            if logits:
                if not hasattr(self, '_class_grads_logits_idx'):
                    self._class_grads_logits_idx = [None for _ in range(nb_outputs)]

                for l in unique_labels:
                    if self._class_grads_logits_idx[l] is None:
                        class_grads_logits = [k.gradients(self._preds_op[:, l], self._input)[0]]
                        self._class_grads_logits_idx[l] = k.function([self._input], class_grads_logits)
            else:
                if not hasattr(self, '_class_grads_idx'):
                    self._class_grads_idx = [None for _ in range(nb_outputs)]

                for l in unique_labels:
                    if self._class_grads_idx[l] is None:
                        class_grads = [k.gradients(k.softmax(self._preds_op)[:, l], self._input)[0]]
                        self._class_grads_idx[l] = k.function([self._input], class_grads)

    def _get_layers(self):
        """
        Return the hidden layers in the model, if applicable.

        :return: The hidden layers in the model, input and output layers excluded.
        :rtype: `list`
        """

        layer_names = [layer.name for layer in self._model.layers[:-1] if not isinstance(layer, InputLayer)]
        logger.info('Inferred %i hidden layers on Keras classifier.', len(layer_names))

        return layer_names

    def save(self, filename, path=None):
        """
        Save a model to file in the format specific to the backend framework. For Keras, .h5 format is used.

        :param filename: Name of the file where to store the model.
        :type filename: `str`
        :param path: Path of the folder where to store the model. If no path is specified, the model will be stored in
                     the default data location of the library `DATA_PATH`.
        :type path: `str`
        :return: None
        """

        if path is None:
            path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')), 'data')

        if not os.path.exists(path): os.makedirs(path)
        full_path = os.path.join(path, filename)

        self._model.save(str(full_path))
        logger.info('Model saved in path: %s.', full_path)

    def load(self, filename, path=None):
        """
        Save a model to file in the format specific to the backend framework. For Keras, .h5 format is used.

        :param filename: Name of the file where to store the model.
        :type filename: `str`
        :param path: Path of the folder where to store the model. If no path is specified, the model will be stored in
                     the default data location of the library `DATA_PATH`.
        :type path: `str`
        :return: None
        """

        if path is None:
            path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')), 'data')

        if not os.path.exists(path): os.makedirs(path)
        full_path = str(os.path.join(path, filename))

        self._model.set_weights(load_model(full_path).get_weights())
        logger.info('Load model from path: %s.', full_path)


def generator_fit(x, y, batch_size=128):
    """
    Minimal data generator for randomly batching large datasets.

    :param x: The data sample to batch.
    :type x: `np.ndarray`
    :param y: The labels for `x`. The first dimension has to match the first dimension of `x`.
    :type y: `np.ndarray`
    :param batch_size: The size of the batches to produce.
    :type batch_size: `int`
    :return: A batch of size `batch_size` of random samples from `(x, y)`
    :rtype: `tuple(np.ndarray, np.ndarray)`
    """
    while True:
        indices = np.random.randint(x.shape[0], size=batch_size)
        yield x[indices], y[indices]
