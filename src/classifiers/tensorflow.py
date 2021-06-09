from __future__ import absolute_import, division, print_function, unicode_literals
import os, sys, random, six, time
sys.path.append('..')

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import numpy as np
import tensorflow as tf

import psutil
process = psutil.Process(os.getpid())


from classifiers.classifier import Classifier


class TFClassifier(Classifier):
    """
    This class implements a classifier with the Tensorflow framework.
    """
    def __init__(self, clip_values, input_ph, logits, output_ph=None, lr = None, train=None, loss=None, sess=None, preprocessing=(0, 1)):
        """
        Initialization specifically for the Tensorflow-based implementation.

        :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
               for features.
        :type clip_values: `tuple`
        :param input_ph: The input placeholder.
        :type input_ph: `tf.Placeholder`
        :param logits: The logits layer of the model.
        :type logits: `tf.Tensor`
        :param output_ph: The labels placeholder of the model. This parameter is necessary when training the model and
               when computing gradients w.r.t. the loss function.
        :type output_ph: `tf.Tensor`
        :param: lr: learning rate
        :type: lr: `tf.Tensor`
        :param train: The train tensor for fitting, including an optimizer. Use this parameter only when training the
               model.
        :type train: `tf.Tensor`
        :param loss: The loss function for which to compute gradients. This parameter is necessary when training the
               model and when computing gradients w.r.t. the loss function.
        :type loss: `tf.Tensor`
        :param sess: Computation session.
        :type sess: `tf.Session`
        :param preprocessing: Tuple of the form `(substractor, divider)` of floats or `np.ndarray` of values to be
               used for data preprocessing. The first value will be substracted from the input. The input will then
               be divided by the second one.
        :type preprocessing: `tuple`
        """
        super(TFClassifier, self).__init__(clip_values=clip_values, preprocessing=preprocessing)
        self._nb_classes = int(logits.get_shape()[-1])
        self._input_shape = tuple(input_ph.get_shape()[1:])
        self._input_ph = input_ph
        self._logits = logits
        self._output_ph = output_ph
        self._train = train
        self._loss = loss
        self._lr = lr

        # Assign session
        if sess is None: raise ValueError("A session cannot be None.")
        else: self._sess = sess

        # Get the internal layers
        self._layer_names = self._get_layers()

        # Must be set here for the softmax output
        self._probs = tf.nn.softmax(logits)

        # Get the loss gradients graph
        if self._loss is not None:
            self._loss_grads = tf.gradients(self._loss, self._input_ph)[0]

        self._layer_tensor_map = {};
        self._layer_grads_map = {};
        self._layer_grads_tau_map = {};


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
        x_ = self._apply_processing(x)

        # Run prediction with batch processing
        results = np.zeros((x_.shape[0], self.nb_classes), dtype=np.float32)
        num_batch = int(np.ceil(len(x_) / float(batch_size)))
        for m in range(num_batch):
            # Batch indexes
            begin, end = m * batch_size, min((m + 1) * batch_size, x_.shape[0])

            # Create feed_dict
            fd = {self._input_ph: x_[begin:end]}

            # Run prediction
            if logits:
                results[begin:end] = self._sess.run(self._logits, feed_dict=fd)
            else:
                results[begin:end] = self._sess.run(self._probs, feed_dict=fd)

        return results

    def fit(self, x, y, lr, batch_size=128, nb_epochs=10, display_interval = 10):
        """
        Fit the classifier on the training set `(x, y)`.

        :param x: Training data.
        :type x: `np.ndarray`
        :param y: Labels, one-vs-rest encoding.
        :type y: `np.ndarray`
        :param lr: Learning rate.
        :type lr: `float32`
        :param batch_size: Size of batches.
        :type batch_size: `int`
        :param nb_epochs: Number of epochs to use for trainings.
        :type nb_epochs: `int`
        :param display_interval: Display interval.
        :type display_interval: `int`
        :return: `None`
        """
        # Check if train and output_ph available
        if self._train is None or self._output_ph is None:
            raise ValueError("Need the training objective and the output placeholder to train the model.")

        # Apply defences
        x_ = self._apply_processing(x)
        y_ = y

        num_batch = int(np.ceil(len(x_) / float(batch_size)))
        ind = np.arange(len(x_))

        # Start training
        for ii in range(nb_epochs):
            if (ii+1) % display_interval == 0:
                print('epoch No. %d of %d' % (ii+1, nb_epochs))
                print(lr)
            # Shuffle the examples
            random.shuffle(ind)
            # Train for one epoch
            for m in range(num_batch):
                i_batch = x_[ind[m * batch_size:(m + 1) * batch_size]]
                o_batch = y_[ind[m * batch_size:(m + 1) * batch_size]]

                # Run train step
                self._sess.run(self._train, feed_dict={self._input_ph: i_batch, self._output_ph: o_batch})

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
                or (isinstance(label, np.ndarray) and len(label.shape) == 1 and (label < self._nb_classes).all()
                    and label.shape[0] == x.shape[0])):
            raise ValueError('Label %s is out of range.' % label)

        self._init_class_grads(label=label, logits=logits)

        x_ = self._apply_processing(x)

        # Compute the gradient and return
        if label is None:
            # Compute the gradients w.r.t. all classes
            if logits:
                grads = self._sess.run(self._logit_class_grads, feed_dict={self._input_ph: x_})
            else:
                grads = self._sess.run(self._class_grads, feed_dict={self._input_ph: x_})

            grads = np.swapaxes(np.array(grads), 0, 1)
            grads = self._apply_processing_gradient(grads)

        elif isinstance(label, (int, np.integer)):
            # Compute the gradients only w.r.t. the provided label
            if logits:
                grads = self._sess.run(self._logit_class_grads[label], feed_dict={self._input_ph: x_})
            else:
                grads = self._sess.run(self._class_grads[label], feed_dict={self._input_ph: x_})

            grads = grads[None, ...]
            grads = np.swapaxes(np.array(grads), 0, 1)
            grads = self._apply_processing_gradient(grads)

        else:
            # For each sample, compute the gradients w.r.t. the indicated target class (possibly distinct)
            unique_label = list(np.unique(label))
            if logits:
                grads = self._sess.run([self._logit_class_grads[l] for l in unique_label],
                                       feed_dict={self._input_ph: x_})
            else:
                grads = self._sess.run([self._class_grads[l] for l in unique_label],
                                       feed_dict={self._input_ph: x_})

            grads = np.swapaxes(np.array(grads), 0, 1)
            lst = [unique_label.index(i) for i in label]
            grads = np.expand_dims(grads[np.arange(len(grads)), lst], axis=1)

            grads = self._apply_processing_gradient(grads)

        return grads

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

        # Check if loss available
        if not hasattr(self, '_loss_grads') or self._loss_grads is None or self._output_ph is None:
            raise ValueError("Need the loss function and the labels placeholder to compute the loss gradient.")

        # Compute the gradient and return
        grds = self._sess.run(self._loss_grads, feed_dict={self._input_ph: x_, self._output_ph: y})
        grds = self._apply_processing_gradient(grds)
        assert grds.shape == x_.shape

        return grds

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
        # Get the computational graph
        if layer not in self._layer_tensor_map:
            with self._sess.graph.as_default():
                graph = tf.get_default_graph()
            if isinstance(layer, six.string_types):  # basestring for Python 2 (str, unicode) support
                if layer not in self._layer_names: raise ValueError("Layer name %s is not part of the graph." % layer)
                layer_tensor = graph.get_tensor_by_name(layer)
            elif isinstance(layer, (int, np.integer)):
                layer_tensor = graph.get_tensor_by_name(self._layer_names[layer])
            else:
                raise TypeError("Layer must be of type `str` or `int`. Received '%s'", layer)
            self._layer_tensor_map[layer] = layer_tensor;

        if layer not in self._layer_grads_map:
            loss = tf.reduce_mean(tf.squared_difference(self._layer_tensor_map[layer], layer_ref))
            loss_grads = tf.gradients(loss, self._input_ph)[0]
            self._layer_grads_map[layer] = loss_grads;

        # Run prediction
        grds = self._sess.run(self._layer_grads_map[layer], feed_dict={self._input_ph: x})
        assert grds.shape == x.shape
        return grds


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
        if layer not in self._layer_tensor_map:
            with self._sess.graph.as_default():
                graph = tf.get_default_graph()
            if isinstance(layer, six.string_types):  # basestring for Python 2 (str, unicode) support
                if layer not in self._layer_names: raise ValueError("Layer name %s is not part of the graph." % layer)
                layer_tensor = graph.get_tensor_by_name(layer)
            elif isinstance(layer, (int, np.integer)):
                layer_tensor = graph.get_tensor_by_name(self._layer_names[layer])
            else:
                raise TypeError("Layer must be of type `str` or `int`. Received '%s'", layer)
            self._layer_tensor_map[layer] = layer_tensor;

        if layer not in self._layer_grads_tau_map:
            loss = tf.reduce_mean( tf.maximum(tf.subtract(tf.squared_difference(self._layer_tensor_map[layer], layer_ref), tau), 0 ) );
            loss_grads = tf.gradients(loss, self._input_ph)[0]
            self._layer_grads_tau_map[layer] = loss_grads;

        grds = self._sess.run(self._layer_grads_tau_map[layer], feed_dict={self._input_ph: x})
        assert grds.shape == x.shape
        return grds


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
        loss = np.sum(np.square(layer_x-layer_ref),axis=tuple(range(1,len(layer_x.shape))));
        assert len(loss) == x.shape[0];
        return loss

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
        # get the activations of the specific layer
        x_ = self._apply_processing(x)
        with self._sess.graph.as_default():
            graph = tf.get_default_graph()

        if isinstance(layer, six.string_types):  # basestring for Python 2 (str, unicode) support
            if layer not in self._layer_names:
                raise ValueError("Layer name %s is not part of the graph." % layer)
            self._layer_tensor = graph.get_tensor_by_name(layer)
        elif isinstance(layer, (int, np.integer)):
            self._layer_tensor = graph.get_tensor_by_name(self._layer_names[layer])
        elif isinstance(layer, tf.Tensor):
            self._layer_tensor = layer
        else:
            raise TypeError("Layer must be of type `str` or `int`. Received '%s'", layer)

        # Check value of label for computing gradients
        if not (isinstance(label, (int, np.ndarray)) and np.max(label) in range(self.nb_classes)):
            raise ValueError('Only support int lable and current label %s is out of range.' % label)

        # Compute the gradient and activations
        self._logit_gradcam_grads = [tf.gradients(self._logits[:, label[i]], self._layer_tensor)[0][i] for i in range(label.shape[0])]
        self._logit_gradcam_grads = tf.stack(self._logit_gradcam_grads)
        grads, activations = self._sess.run([self._logit_gradcam_grads, self._layer_tensor], feed_dict={self._input_ph: x_})
        return grads, activations


    def _init_class_grads(self, label=None, logits=False):

        if logits:
            if not hasattr(self, '_logit_class_grads'):
                self._logit_class_grads = [None for _ in range(self.nb_classes)]
        else:
            if not hasattr(self, '_class_grads'):
                self._class_grads = [None for _ in range(self.nb_classes)]

        # Construct the class gradients graph
        if label is None:
            if logits:
                if None in self._logit_class_grads:
                    self._logit_class_grads = [tf.gradients(self._logits[:, i], self._input_ph)[0]
                                               if self._logit_class_grads[i] is None else self._logit_class_grads[i]
                                               for i in range(self._nb_classes)]
            else:
                if None in self._class_grads:
                    self._class_grads = [tf.gradients(self._probs[:, i], self._input_ph)[0]
                                         if self._class_grads[i] is None else self._class_grads[i]
                                         for i in range(self._nb_classes)]

        elif isinstance(label, int):
            if logits:
                if self._logit_class_grads[label] is None:
                    self._logit_class_grads[label] = tf.gradients(self._logits[:, label], self._input_ph)[0]
            else:
                if self._class_grads[label] is None:
                    self._class_grads[label] = tf.gradients(self._probs[:, label], self._input_ph)[0]

        else:
            if logits:
                for l in np.unique(label):
                    if self._logit_class_grads[l] is None:
                        self._logit_class_grads[l] = tf.gradients(self._logits[:, l], self._input_ph)[0]
            else:
                for l in np.unique(label):
                    if self._class_grads[l] is None:
                        self._class_grads[l] = tf.gradients(self._probs[:, l], self._input_ph)[0]

    def _get_layers(self):
        """
        Return the hidden layers in the model, if applicable.

        :return: The hidden layers in the model, input and output layers excluded.
        :rtype: `list`
        """

        # Get the computational graph
        with self._sess.graph.as_default():
            graph = tf.get_default_graph()

        # Get the list of operators and heuristically filter them
        tmp_list = []
        ops = graph.get_operations()

        for op in ops:
            filter_cond = ((op.values()) and (not op.values()[0].get_shape() == None) and (
                len(op.values()[0].get_shape().as_list()) > 1) and (
                        op.values()[0].get_shape().as_list()[0] is None) and (
                        op.values()[0].get_shape().as_list()[1] is not None) and (
                        not op.values()[0].name.startswith("gradients")) and (
                        not op.values()[0].name.startswith("softmax_cross_entropy_loss")) and (
                        not op.type == "Placeholder")) and (not 'random_uniform' in op.values()[0].name)

            if filter_cond:
                tmp_list.append(op.values()[0].name)

        # Shorten the list
        if not tmp_list:
            return tmp_list

        result = [tmp_list[-1]]
        for name in reversed(tmp_list[:-1]):
            if len(name.split("/")) > 2:
                if 'BatchNorm' in name:
                    continue
                else:
                    if result[0].split("/")[-2] != name.split("/")[-2]:
                        result = [name] + result
            else:
                if result[0].split("/")[0] != name.split("/")[0]:
                    result = [name] + result
        logger.info('Inferred %i hidden layers on TensorFlow classifier.', len(result))

        return result

    def attack_saliency_gradient(self, method, x, label = None, k_top = 100, target_map = None,
                                 logits = False, positive_only = False):
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

        if len(self._output_ph.shape) == 2:
            nb_outputs = self._output_ph.shape[1]
        else:
            raise ValueError('Unexpected output shape for classification in Keras model.')

        if label is None:
            raise ValueError('Missing label information.')

        ## Compute the normalized gradient saliency for the target label
        b = x.shape[0]
        w = self._input_ph.shape[1].value
        h = self._input_ph.shape[2].value
        c = self._input_ph.shape[3].value

        if logits == True:
            class_grads_logits = [tf.gradients(self._logits[:, label[i]], self._input_ph)[0][i] for i in range(label.shape[0])]
            class_grads_logits = tf.stack(class_grads_logits)
        else:
            class_grads_logits = [tf.gradients(self._probs[:, label[i]], self._input_ph)[0][i] for i in range(label.shape[0])]
            class_grads_logits = tf.stack(class_grads_logits)
        if positive_only == True:
            saliency_unnormalized = tf.reduce_sum(tf.nn.relu(class_grads_logits), -1)
            self._saliency = saliency_unnormalized/(tf.reduce_sum(saliency_unnormalized, axis=(1,2))+1e-32)[:,None,None]
        else:
            saliency_unnormalized = tf.reduce_sum(tf.abs(class_grads_logits), -1)
            self._saliency = saliency_unnormalized/(tf.reduce_sum(saliency_unnormalized, axis=(1,2))+1e-32)[:,None,None]

        self._saliency_flatten = tf.reshape(self._saliency, [b, w*h])
        top_val, self._top_idx = tf.nn.top_k(self._saliency_flatten, k_top)

        ## Compute the center mass of x: C(x)
        y_mesh, x_mesh = np.meshgrid(np.arange(h), np.arange(w))
        self._mass_center = tf.stack([tf.reduce_sum(self._saliency * x_mesh, axis=(1,2)) / (w * h),
                                      tf.reduce_sum(self._saliency * y_mesh, axis=(1,2)) / (w * h)])
        self._mass_center = tf.transpose(self._mass_center)

        # Create feed_dict
        fd = {self._input_ph: x}

        ## get the saliency map and the top k faetures for x
        saliency, saliency_flatten, topK, mass_center = \
            self._sess.run([self._saliency, self._saliency_flatten, self._top_idx,self._mass_center], feed_dict=fd)

        ## get the gradient of each attack loss w.r.t. input
        elem1 = np.argsort(np.reshape(saliency, [b, w * h]), axis=-1)[:,-k_top:]
        elements1 = np.zeros((b, w * h))
        for ii in range(b):
            elements1[ii,elem1[ii]]=1
        topK_loss = -tf.reduce_sum(self._saliency_flatten * elements1, axis=1)
        self._topK_direction = tf.stack([tf.gradients(topK_loss[ll], self._input_ph)[0][ll] for ll in range(b)])

        mass_center_loss = tf.reduce_sum((self._mass_center - mass_center) ** 2, axis=1)
        self._mass_center_direction = tf.stack([tf.gradients(mass_center_loss[ll], self._input_ph)[0][ll] for ll in range(b)])

        if target_map is not None:
            target_loss = tf.reduce_sum((self._saliency * target_map[None,:,:]), axis=(1,2))
            self._target_direction = tf.stack([tf.gradients(target_loss[ll], self._input_ph)[0][ll] for ll in range(b)])
        return saliency, saliency_flatten, topK, mass_center

    def attack_saliency_integratedgrad(self, method, x, label, num_steps = 25, k_top = 100,
                                       target_map = None, reference_image = None, logits = False, positive_only = False):
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
        if len(self._output_ph.shape) == 2:
            nb_outputs = self._output_ph.shape[1]
        else:
            raise ValueError('Unexpected output shape for classification in Keras model.')

        if label is None:
            raise ValueError('Missing label information.')

        # Compute the normalized integrated gradient saliency for the target label
        b = x.shape[0]
        w = self._input_ph.shape[1].value
        h = self._input_ph.shape[2].value
        c = self._input_ph.shape[3].value
        if logits == True:
            class_grads_logits = [tf.reduce_sum(tf.gradients(self._logits[:, label[i]], self._input_ph)[0][i*num_steps:(i+1)*num_steps], axis=0)
                                  for i in range(label.shape[0])]
        else:
            class_grads_logits = [tf.reduce_sum(tf.gradients(self._probs[:, label[i]], self._input_ph)[0][i*num_steps:(i+1)*num_steps], axis=0)
                                  for i in range(label.shape[0])]
        inte_map_logits = [class_grads_logits[i] * (self._input_ph[(i+1)*num_steps-1] -reference_image)
                           for i in range(label.shape[0])]
        inte_map_logits = tf.stack(inte_map_logits)

        if positive_only == True:
            saliency_unnormalized = tf.reduce_sum(tf.nn.relu(inte_map_logits), -1)
            self._saliency = saliency_unnormalized/(tf.reduce_sum(saliency_unnormalized, axis=(1,2))+1e-32)[:,None,None]
        else:
            saliency_unnormalized = tf.reduce_sum(tf.abs(inte_map_logits), -1)
            self._saliency = saliency_unnormalized/(tf.reduce_sum(saliency_unnormalized, axis=(1,2))+1e-32)[:,None,None]

        self._saliency_flatten = tf.reshape(self._saliency, [b, w*h])

        # Compute the top K features and the center mass of x: C(x)
        top_val, self._top_idx = tf.nn.top_k(self._saliency_flatten, k_top)
        y_mesh, x_mesh = np.meshgrid(np.arange(h), np.arange(w))
        self._mass_center = tf.stack([tf.reduce_sum(self._saliency * x_mesh, axis=(1,2)) / (w * h),
                                      tf.reduce_sum(self._saliency * y_mesh, axis=(1,2)) / (w * h)])
        self._mass_center = tf.transpose(self._mass_center)

        # Create a set of images
        counterfactuals = []
        for ii in range(b):
            ref_subtracted = x[ii,] - reference_image
            counterfactuals_tmp = np.array([(float(i + 1) / num_steps) * ref_subtracted + reference_image
                                            for i in range(num_steps)])
            counterfactuals.extend(counterfactuals_tmp)

        counterfactuals = np.array(counterfactuals)
        # Create feed_dict
        fd = {self._input_ph: counterfactuals}

        ## get the saliency map and the top k faetures for the counterfactuals
        saliency, saliency_flatten, topK, mass_center = \
            self._sess.run([self._saliency, self._saliency_flatten,self._top_idx, self._mass_center], feed_dict=fd)

        ## get the gradient of each attack loss w.r.t. input
        elem1 = np.argsort(np.reshape(saliency, [b, w * h]), axis=-1)[:,-k_top:]
        elements1 = np.zeros((b, w * h))
        for ii in range(b):
            elements1[ii,elem1[ii]]=1
        topK_loss = -tf.reduce_sum(self._saliency_flatten *elements1, axis=1)
        self._topK_direction = tf.stack([tf.gradients(topK_loss[ll], self._input_ph)[0][ll*num_steps:(ll+1)*num_steps] for ll in range(b)])

        mass_center_loss = tf.reduce_sum((self._mass_center - mass_center) ** 2, axis=1)
        self._mass_center_direction = tf.stack([tf.gradients(mass_center_loss[ll], self._input_ph)[0][ll*num_steps:(ll+1)*num_steps] for ll in range(b)])

        if target_map is not None:
            target_loss = tf.reduce_sum((self._saliency * target_map[None,:,:]), axis=(1,2))
            self._target_direction = tf.stack([tf.gradients(target_loss[ll], self._input_ph)[0][ll*num_steps:(ll+1)*num_steps] for ll in range(b)])
        return saliency, saliency_flatten, topK, mass_center

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
        if len(self._output_ph.shape) == 2:
            nb_outputs = self._output_ph.shape[1]
        else:
            raise ValueError('Unexpected output shape for classification in Keras model.')

        if label is None:
            raise ValueError('Missing label information.')

        # Compute the normalized integrated gradient saliency for the target label
        b = x.shape[0]
        w = self._input_ph.shape[1].value
        h = self._input_ph.shape[2].value
        c = self._input_ph.shape[3].value

        if logits == True:
            class_grads_logits = [tf.reduce_mean(tf.gradients(self._logits[:, label[i]], self._input_ph)[0][i*nsamples:(i+1)*nsamples], axis=0)
                                  for i in range(label.shape[0])]
        else:
            class_grads_logits = [tf.reduce_mean(tf.gradients(self._probs[:, label[i]], self._input_ph)[0][i*nsamples:(i+1)*nsamples], axis=0)
                                  for i in range(label.shape[0])]
        class_grads_logits = tf.stack(class_grads_logits)

        if positive_only == True:
            saliency_unnormalized = tf.reduce_sum(tf.nn.relu(class_grads_logits), -1)
            self._saliency = saliency_unnormalized/(tf.reduce_sum(saliency_unnormalized, axis=(1,2))+1e-32)[:,None,None]
        else:
            saliency_unnormalized = tf.reduce_sum(tf.abs(class_grads_logits), -1)
            self._saliency = saliency_unnormalized/(tf.reduce_sum(saliency_unnormalized, axis=(1,2))+1e-32)[:,None,None]

        self._saliency_flatten = tf.reshape(self._saliency, [b, w*h])
        top_val, self._top_idx = tf.nn.top_k(self._saliency_flatten, k_top)

        ## Compute the center mass of x: C(x)
        y_mesh, x_mesh = np.meshgrid(np.arange(h), np.arange(w))
        self._mass_center = tf.stack([tf.reduce_sum(self._saliency * x_mesh, axis=(1,2)) / (w * h),
                                      tf.reduce_sum(self._saliency * y_mesh, axis=(1,2)) / (w * h)])
        self._mass_center = tf.transpose(self._mass_center)

        # Create feed_dict
        x_plus_noise = []
        for ii in range(b):
            stdev = stdev_spread * (np.max(x[ii,]) - np.min(x[ii,]))
            noise = np.random.normal(0, stdev, (nsamples, w, h, c))
            x_plus_noise_tmp = np.repeat(np.expand_dims(x[ii,], axis = 0), nsamples, axis=0) + noise
            x_plus_noise.extend(x_plus_noise_tmp)
        x_plus_noise = np.array(x_plus_noise)

        fd = {self._input_ph: x_plus_noise}

        ## get the saliency map and the top k faetures for the counterfactuals
        saliency, saliency_flatten, topK, mass_center = \
            self._sess.run([self._saliency, self._saliency_flatten,self._top_idx, self._mass_center], feed_dict=fd)

        ## get the gradient of each attack loss w.r.t. input
        elem1 = np.argsort(np.reshape(saliency, [b, w * h]), axis=-1)[:,-k_top:]
        elements1 = np.zeros((b, w * h))
        for ii in range(b):
            elements1[ii,elem1[ii]]=1
        topK_loss = -tf.reduce_sum(self._saliency_flatten *elements1, axis=1)
        self._topK_direction = tf.stack([tf.gradients(topK_loss[ll], self._input_ph)[0][ll*nsamples:(ll+1)*nsamples] for ll in range(b)])

        mass_center_loss = tf.reduce_sum((self._mass_center - mass_center) ** 2, axis=1)
        self._mass_center_direction = tf.stack([tf.gradients(mass_center_loss[ll], self._input_ph)[0][ll*nsamples:(ll+1)*nsamples] for ll in range(b)])

        if target_map is not None:
            target_loss = tf.reduce_sum((self._saliency * target_map[None,:,:]), axis=(1,2))
            self._target_direction = tf.stack([tf.gradients(target_loss[ll], self._input_ph)[0][ll*nsamples:(ll+1)*nsamples] for ll in range(b)])
        return saliency, saliency_flatten, topK, mass_center

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
        :return: Array of perturbation
        :rtype: 'numpy.array'
        """
        if label is None:
            raise ValueError('Missing label information.')

        with self._sess.graph.as_default():
            graph = tf.get_default_graph()

        if isinstance(layer, six.string_types):  # basestring for Python 2 (str, unicode) support
            if layer not in self._layer_names:
                raise ValueError("Layer name %s is not part of the graph." % layer)
            self._layer_tensor = graph.get_tensor_by_name(layer)
        elif isinstance(layer, int):
            self._layer_tensor = graph.get_tensor_by_name(self._layer_names[layer])
        elif isinstance(layer, tf.Tensor):
            self._layer_tensor = layer
        else:
            raise TypeError("Layer must be of type `str` or `int`. Received '%s'", layer)

        # Check value of label for computing gradients
        if not (isinstance(label, (int, np.ndarray)) and np.max(label) in range(self.nb_classes)):
            raise ValueError('Only support int lable and current label %s is out of range.' % label)

        # Compute the normalized integrated gradient saliency for the target label
        b = x.shape[0]
        w = self._layer_tensor.shape[1].value
        h = self._layer_tensor.shape[2].value
        c = self._layer_tensor.shape[3].value

        # Compute the gradient and activations
        if logits == True:
            self._logit_gradcam_grads = [tf.gradients(self._logits[:, label[l]], self._layer_tensor)[0][l]
                                         for l in range(label.shape[0])]
        else:
            self._logit_gradcam_grads = [tf.gradients(self._probs[:, label[l]], self._layer_tensor)[0][l]
                                         for l in range(label.shape[0])]
        self._logit_gradcam_grads = tf.stack(self._logit_gradcam_grads)
        weights = tf.reduce_mean(self._logit_gradcam_grads, axis=(1,2))

        grad_cams = tf.reduce_sum(weights[:, None, None, :] * self._layer_tensor, axis = -1)

        if positive_only == True:
            saliency_unnormalized = tf.nn.relu(grad_cams)
            self._saliency = saliency_unnormalized/(tf.reduce_sum(saliency_unnormalized, axis=(1,2))+1e-32)[:,None,None]
        else:
            saliency_unnormalized = tf.abs(grad_cams)
            self._saliency = saliency_unnormalized/(tf.reduce_sum(saliency_unnormalized, axis=(1,2))+1e-32)[:,None,None]

        self._saliency_flatten = tf.reshape(self._saliency, [b, w*h])

        # Compute the top K features and the center mass of x: C(x)
        top_val, self._top_idx = tf.nn.top_k(self._saliency_flatten, k_top)
        y_mesh, x_mesh = np.meshgrid(np.arange(h), np.arange(w))
        self._mass_center = tf.stack([tf.reduce_sum(self._saliency * x_mesh, axis=(1,2)) / (w * h),
                                      tf.reduce_sum(self._saliency * y_mesh, axis=(1,2)) / (w * h)])
        self._mass_center = tf.transpose(self._mass_center)

        # Create feed_dict
        fd = {self._input_ph: x}

        ## get the saliency map and the top k faetures for the counterfactuals
        saliency, saliency_flatten, topK, mass_center = \
            self._sess.run([self._saliency, self._saliency_flatten,self._top_idx, self._mass_center], feed_dict=fd)

        ## get the gradient of each attack loss w.r.t. input
        elem1 = np.argsort(np.reshape(saliency, [b, w * h]), axis=-1)[:,-k_top:]
        elements1 = np.zeros((b, w * h))
        for ii in range(b):
            elements1[ii,elem1[ii]]=1
        topK_loss = -tf.reduce_sum(self._saliency_flatten *elements1, axis=1)
        self._topK_direction = tf.stack([tf.gradients(topK_loss[ll], self._input_ph)[0][ll] for ll in range(b)])

        mass_center_loss = tf.reduce_sum((self._mass_center - mass_center) ** 2, axis=1)
        self._mass_center_direction = tf.stack([tf.gradients(mass_center_loss[ll], self._input_ph)[0][ll] for ll in range(b)])

        if target_map is not None:
            target_loss = tf.reduce_sum((self._saliency * target_map[None,:,:]), axis=(1,2))
            self._target_direction = tf.stack([tf.gradients(target_loss[ll], self._input_ph)[0][ll] for ll in range(b)])
        return saliency, saliency_flatten, topK, mass_center

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

        if layer not in self._layer_tensor_map:
            # Get the computational graph
            with self._sess.graph.as_default():
                graph = tf.get_default_graph()

            if isinstance(layer, six.string_types):  # basestring for Python 2 (str, unicode) support
                if layer not in self._layer_names: raise ValueError("Layer name %s is not part of the graph." % layer)
                layer_tensor = graph.get_tensor_by_name(layer)
            elif isinstance(layer, (int, np.integer)):
                layer_tensor = graph.get_tensor_by_name(self._layer_names[layer])
            else:
                raise TypeError("Layer must be of type `str` or `int`. Received '%s'", layer)
            self._layer_tensor_map[layer] = layer_tensor;

        # Run prediction
        result = self._sess.run(self._layer_tensor_map[layer], feed_dict={self._input_ph: x})

        return result

    def save(self, filename, path=None):
        """
        Save a model to file in the format specific to the backend framework. For TensorFlow, .ckpt is used.

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

        saver = tf.train.Saver()
        saver.save(self._sess, full_path)
        logger.info('Model saved in path: %s.', full_path)

    def load(self, filename, path = None):
        """
        load a model to file in the format specific to the backend framework. For TensorFlow, .ckpt is used.

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
        saver = tf.train.Saver()
        saver.restore(self._sess, full_path)
        logger.info('Load model from path: %s.', full_path)