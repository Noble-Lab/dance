from __future__ import absolute_import, division, print_function, unicode_literals

import os, sys, abc

# Ensure compatibility with Python 2 and 3 when using ABCMeta
if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta(str('ABC'), (), {})


class Knockoff(ABC):
    """
    Abstract base class for all knockoff classes.
    """
    knockoff_params = ['classifier']

    def __init__(self, classifier, saliency_gen, knock_direction, partition_idx, partition_sum, layer):
        """
        :param classifier: A trained model.
        :type classifier: :class:`Classifier`
        :param saliency_gen: The saliency generator for knockoffs
        :type saliency_gen: `Saliency`
        """
        self.classifier = classifier;
        self.saliency_gen = saliency_gen;
        self.knock_direction = knock_direction
        self.partition_idx = partition_idx;
        self.partition_sum = partition_sum;
        self.layer = layer;

    def generate(self, x, y, **kwargs):
        """
        Generate knockoff examples and return them as an array. This method should be overridden by all concrete implementations.

        :param x: An array with the original inputs to generate knockoff from.
        :type x: `np.ndarray`
        :param y: Labels for generating saliency maps, not for generating knockoffs.
        :type y: `np.ndarray`
        :param kwargs: Knockoff-specific parameters used by child classes.
        :type kwargs: `dict`
        :return: One array holding the knockoff upper/lower bounds, and one array holding the saliency upper/lower bounds (depend on knock_direction)
        :rtype: `np.ndarray tuple`
        """
        raise NotImplementedError

    def set_params(self, **kwargs):
        """
        Take in a dictionary of parameters and apply knockoff-specific checks before saving them as attributes.

        :param kwargs: a dictionary of knockoff-specific parameters
        :type kwargs: `dict`
        :return: `True` when parsing was successful
        """
        for key, value in kwargs.items():
            if key in self.knockoff_params:
                setattr(self, key, value)
        return True

    def get_params_str(self):
        """
        Get knockoff params str.
        :return: Knockoff params str
        :rtype: `str`
        """
        raise NotImplementedError