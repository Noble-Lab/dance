from __future__ import absolute_import, division, print_function, unicode_literals

import os, sys, abc
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Ensure compatibility with Python 2 and 3 when using ABCMeta
if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta(str('ABC'), (), {})


class Classifier(ABC):
    """
    Base class for all classifiers.
    """
    def __init__(self, clip_values, preprocessing=(0, 1)):
        """
        Initialize a `Classifier` object.
        :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
               for features.
        :type clip_values: `tuple`
        :param preprocessing: Tuple of the form `(substractor, divider)` of floats or `np.ndarray` of values to be
               used for data preprocessing. The first value will be substracted from the input. The input will then
               be divided by the second one.
        :type preprocessing: `tuple`
        """
        if len(clip_values) != 2:
            raise ValueError('`clip_values` should be a tuple of 2 floats containing the allowed data range.')
        self._clip_values = clip_values

        if len(preprocessing) != 2:
            raise ValueError('`preprocessing` should be a tuple of 2 floats with the substract and divide values for'
                             'the model inputs.')
        self._preprocessing = preprocessing

    @abc.abstractmethod
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
        raise NotImplementedError

    @abc.abstractmethod
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
        raise NotImplementedError

    @property
    def nb_classes(self):
        """
        Return the number of output classes.

        :return: Number of classes in the data.
        :rtype: `int`
        """
        return self._nb_classes

    @property
    def input_shape(self):
        """
        Return the shape of one input.

        :return: Shape of one input for the classifier.
        :rtype: `tuple`
        """
        return self._input_shape

    @property
    def clip_values(self):
        """
        :return: Tuple of the form `(min, max)` representing the minimum and maximum values allowed for features.
        :rtype: `tuple`
        """
        return self._clip_values

    @abc.abstractmethod
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
        raise NotImplementedError

    @abc.abstractmethod
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
        raise NotImplementedError

    @abc.abstractmethod
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
        raise NotImplementedError

    def layer_gradient_tau(self, x, layer, layer_ref, tau):
        """
        Compute the gradient of the intermediate layer for input `x`, with respect to the LInf norm.
        :param x: Input for computing the layer gradient.
        :type x: `np.ndarray`
        :param layer: Layer for computing the layer gradient
        :type layer: `int` or `str`
        :param layer_ref: The intermediate representation of the reference input.
        :type layer_ref: `np.ndarray`
        :param tau: Paper link: https://arxiv.org/pdf/1608.04644.pdf
        :type tau: `float`
        :return: Array of gradients of input features w.r.t. the intermediate layer and the reference input in the form
                 `(batch_size, input_shape)`, where the first dimension is the batch size corresponding to `x`.
        :rtype: `np.ndarray`
        """
        raise NotImplementedError

    @abc.abstractmethod
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
        raise NotImplementedError


    @abc.abstractmethod
    def grad_cam_gradient(self, x, label, layer):
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
        raise NotImplementedError

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
        raise NotImplementedError

    @abc.abstractmethod
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
        raise NotImplementedError

    @abc.abstractmethod
    def save(self, filename, path=None):
        """
        Save a model to file in the format specific to the backend framework.

        :param filename: Name of the file where to store the model.
        :type filename: `str`
        :param path: Path of the folder where to store the model. If no path is specified, the model will be stored in
                     the default data location of the library `DATA_PATH`.
        :type path: `str`
        :return: None
        """
        raise NotImplementedError

    @abc.abstractmethod
    def load(self, filename, path=None):
        """
        Load a model to file in the format specific to the backend framework.

        :param filename: Name of the file where to store the model.
        :type filename: `str`
        :param path: Path of the folder where to store the model. If no path is specified, the model will be stored in
                     the default data location of the library `DATA_PATH`.
        :type path: `str`
        :return: None
        """
        raise NotImplementedError

    @abc.abstractmethod
    def attack_saliency_gradient(self, method, x, label, k_top, target_map, logits = False, positive_only = False):
        """
        Compute the perturbation of attack on the gradient saliency method,
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
        :return: Array of perturbation
        :rtype: `numpy.array`
        """
        raise NotImplementedError

    @abc.abstractmethod
    def attack_saliency_integratedgrad(self, method, x, label, num_steps, k_top, target_map, reference_image,
                                       logits = False, positive_only = False):
        """
        Compute the perturbation of attack on the Intergrated gradient saliency method,
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
        :param reference_image: Starting point of the intergrated gradient attack.
        :type reference_image: `numpy.array`
        :return: Array of perturbation
        :rtype: `numpy.array`
        :param logits: Using the logits as outputs.
        :type logits: `bool`
        :param positive_only: Attacking only the positive features.
        :type positive_only: `bool`
        """
        raise NotImplementedError

    @abc.abstractmethod
    def attack_saliency_smoothgrad(self, method, x, label, k_top, target_map, stdev_spread, nsamples,
                                   logits = False, positive_only = False):
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
        """
        raise NotImplementedError

    @abc.abstractmethod
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
        :type layer: `int` or 'string' or 'tf.Tensor'
        :param logits: Using the logits as outputs.
        :type logits: `bool`
        :param positive_only: Attacking only the positive features.
        :type positive_only: `bool`
        """
        raise NotImplementedError

    def _apply_processing(self, x):
        import numpy as np

        sub, div = self._preprocessing
        sub = np.asarray(sub, dtype=x.dtype)
        div = np.asarray(div, dtype=x.dtype)

        res = x - sub
        res = res / div

        return res


    def _apply_processing_gradient(self, grad):
        import numpy as np

        _, div = self._preprocessing
        div = np.asarray(div, dtype=grad.dtype)
        res = grad / div
        return res
