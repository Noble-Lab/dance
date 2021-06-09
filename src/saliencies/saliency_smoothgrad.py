from __future__ import absolute_import, division, print_function, unicode_literals

import os, sys, random
sys.path.append('..')

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import numpy as np

from saliencies.saliency import Saliency
from classifiers.classifier import Classifier


class SmoothGradSaliency(Saliency):

    """
    A Saliency class that computes saliency masks with the SmoothGrad method.
    """
    def __init__(self, classifier, stdev_spread=.15, nsamples=25, magnitude=True, normalize = True):
        """
        :param classifier: A trained model.
        :type classifier: :class:`Classifier`
        :param stdev_spread: Amount of noise to add to the input, as fraction of the
                    total spread (x_max - x_min).
        :type stdev_spread: `float`
        :param nsamples: Number of samples to average across to get the smooth gradient.
        :type nsamples: `int`
        :param magnitude: If true, computes the sum of squares of gradients instead of just the sum.
        :type magnitude: `bool`
        :param normalize: Normalize the obtained saliency map between 0 and 1.
        :type normalize: `bool`

        """
        super(SmoothGradSaliency, self).__init__(classifier, 'smoothgrad')

        self.stdev_spread = stdev_spread
        self.nsamples = nsamples
        self.magnitude = magnitude
        self.normalize = normalize
    def generate(self, x, y, **kwargs):
        """
        Generate saliency and return them as an array.

        :param x: Sample input with shape as expected by the classifier.
        :type x: `np.ndarray`
        :param y: Correct labels.
        :type y: `np.ndarray`
        :param kwargs: saliency-specific parameters used by child classes.
        :type kwargs: `dict`
        :return: An array holding the saliency masks.
        :rtype: `np.ndarray`
        """
        self.set_params(**kwargs)

        stdev = self.stdev_spread * (np.max(x) - np.min(x))
        total_gradients = np.zeros_like(x)

        for i in range(self.nsamples):
            noise = np.random.normal(0, stdev, x.shape)
            x_plus_noise = x + noise

            grads = self.classifier.class_gradient(x_plus_noise, label=y)
            grads = np.sum(grads, axis=1)

            if self.magnitude:total_gradients += (grads * grads)
            else: total_gradients += grads

        sal_x = total_gradients / self.nsamples
        if self.normalize == True:
            sal_x = np.abs(sal_x)

            percent_x = np.percentile(sal_x, 99, axis=range(1, len(sal_x.shape)));
            for _ in range(len(sal_x.shape) - 1): percent_x = percent_x[:, np.newaxis];
            sal_x = np.minimum(sal_x, percent_x);

            sal_x_max = np.max(sal_x, axis=(1,2,3))
            sal_x_max[sal_x_max==0] = 1e-16
            sal_x = sal_x / sal_x_max[:,None,None,None,]
        return sal_x

    # def normalized_saliency(sal_x):
    #     sal_x = np.abs(sal_x);
    #
    #     percent_x = np.percentile(sal_x, 99, axis=range(1, len(sal_x.shape)));
    #     for _ in range(len(sal_x.shape) - 1): percent_x = percent_x[:, np.newaxis];
    #     sal_x = np.minimum(sal_x, percent_x);
    #
    #     sal_x_max = np.max(np.max(np.max(sal_x, axis=1), axis=1), axis=1);
    #     sal_x = sal_x / (sal_x_max[:, None, None, None, ] + np.finfo(np.float).eps);
    #     return sal_x;

    def set_params(self, **kwargs):
        """
        Take in a dictionary of parameters and apply saliency-specific checks before saving them as attributes.

        :param stdev_spread: Amount of noise to add to the input, as fraction of the
                    total spread (x_max - x_min).
        :type stdev_spread: `float`
        :param nsamples: Number of samples to average across to get the smooth gradient.
        :type nsamples: `int`
        :param magnitude: If true, computes the sum of squares of gradients instead of just the sum.
        :type magnitude: `bool`
        """
        # Save attack-specific parameters
        super(SmoothGradSaliency, self).set_params(**kwargs)

        if self.stdev_spread <= 0:
            raise ValueError('The noise `stdev_spread` has to be positive.')

        if self.nsamples <= 0:
            raise ValueError('The sample number `nsamples` has to be positive.')
