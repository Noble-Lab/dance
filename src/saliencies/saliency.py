from __future__ import absolute_import, division, print_function, unicode_literals

import os, sys, abc

# Ensure compatibility with Python 2 and 3 when using ABCMeta
if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta(str('ABC'), (), {})


class Saliency(ABC):
    """
    Abstract base class for all saliency classes.
    """
    def __init__(self, classifier, name):
        """
        :param classifier: A trained model.
        :type classifier: :class:`Classifier`
        """
        self.classifier = classifier;
        self.name = name;

    def generate(self, x, y, **kwargs):
        """
        Generate saliency and return them as an array. This method should be overridden by all concrete implementations.

        :param x: Sample input with shape as expected by the classifier.
        :type x: `np.ndarray`
        :param y: Correct labels.
        :type y: `np.ndarray`
        :param kwargs: saliency-specific parameters used by child classes.
        :type kwargs: `dict`
        :return: An array holding the saliency masks.
        :rtype: `np.ndarray`
        """
        raise NotImplementedError

    def set_params(self, **kwargs):
        """
        Take in a dictionary of parameters and apply saliency-specific checks before saving them as attributes.

        :param kwargs: a dictionary of saliency-specific parameters
        :type kwargs: `dict`
        :return: `True` when parsing was successful
        """
        for key, value in kwargs.items():
            setattr(self, key, value)
        return True
