from __future__ import absolute_import, division, print_function, unicode_literals

import os, sys, random
sys.path.append('..')

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import numpy as np

from saliencies.saliency import Saliency
from classifiers.classifier import Classifier


class GradientSaliency(Saliency):

    """
    A Saliency class that computes saliency masks with a gradient.
    """
    def __init__(self, classifier, normalize = True):
        """
        :param classifier: A trained model.
        :type classifier: :class:`Classifier`
        :param normalize: Normalize the obtained saliency map between 0 and 1.
        :type normalize: `bool`
        """
        super(GradientSaliency, self).__init__(classifier, 'gradient')
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
        grads = self.classifier.class_gradient(x, label=y)
        sal_x = np.sum(grads, axis=1)
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
