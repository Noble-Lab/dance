from __future__ import absolute_import, division, print_function, unicode_literals

import os, sys, random
sys.path.append('..')

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import numpy as np

from saliencies.saliency import Saliency
from classifiers.classifier import Classifier


class IntegratedGradSaliency(Saliency):

    """
    A Saliency class that computes saliency masks with the integrated gradients method.
    https://arxiv.org/abs/1703.01365
    """
    def __init__(self, classifier, x_baseline=None, x_steps=25, normalize = True):
        """
        :param classifier: A trained model.
        :type classifier: :class:`Classifier`
        :param x_baseline: Baseline value used in integration.
        :type x_baseline: `np.ndarray`
        :param x_steps:Number of integrated steps between baseline and x.
        :type x_steps: `int`
        :param normalize: Normalize the obtained saliency map between 0 and 1.
        :type normalize: `bool`
        """
        super(IntegratedGradSaliency, self).__init__(classifier, 'intgrad')

        self.x_baseline = x_baseline
        self.x_steps = x_steps
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

        if self.x_baseline is None:
            self.x_baseline = np.zeros_like(x)
        else: assert self.x_baseline.shape == x.shape

        x_diff = x - self.x_baseline
        total_gradients = np.zeros_like(x)

        for alpha in np.linspace(0, 1, self.x_steps):
            x_step = self.x_baseline + alpha * x_diff

            grads = self.classifier.class_gradient(x_step, label=y)
            # TODO: adding the weights?
            grads = np.sum(grads, axis=1)
            total_gradients += grads
        sal_x = total_gradients * x_diff
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

        :param x_baseline: Baseline value used in integration.
        :type x_baseline: `np.ndarray`
        :param x_steps:Number of integrated steps between baseline and x.
        :type x_steps: `int`
        """
        # Save attack-specific parameters
        super(IntegratedGradSaliency, self).set_params(**kwargs);

        if self.x_steps <= 0:
            raise ValueError('The integrated steps `x_steps` has to be positive.');
