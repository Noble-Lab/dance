from __future__ import absolute_import, division, print_function, unicode_literals

import os, sys, random, six, time
sys.path.append('..')

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import numpy as np

from knockoffs.knockoff import Knockoff
from classifiers.classifier import Classifier

class ConstKnockoff(Knockoff):
    """
    1. We generate the knockoff pixel by pixel.
    2. We use Differential Evolution, a population based optimization method to search the knockoff (without gradient descent),
        similar to "One Pixel Attack for Fooling Deep Neural Networks"
    3. We only consider the knockoff x to be a tunable range around x. In other words, we calculate the upper bound and lower bound of x
    4. We use mean-square-error to measure the distance between intermediate representation f(x) and f(x_ref).
    """

    def __init__(self, classifier, saliency_gen, knock_direction, partition_idx, partition_sum, layer, patch_size=2, stride=1, multi_size=10, ref_values=[]):
        """
        Initialization specifically for the ConstKnockoff.

        :param classifier: A trained model.
        :type classifier: :class:`Classifier`
        :param saliency_gen: The saliency generator for knockoffs
        :type saliency_gen: `Saliency`
        :param layer: Layer for the intermediate representation
        :type layer: `int` or `str`
        :param patch_size: The width/height of the patch
        :type patch_size: `int`
        :param stride: The stride length of the patch, which is no larger than the patch_size
        :type stride: `int`
        :param multi_size: The number of patches optimized at once
        :type multi_size: `int`
        :param ref_values: the reference values each pixel should be set to, same dimension as channels
        :type ref_values: `list`
        """
        super(ConstKnockoff, self).__init__(classifier, saliency_gen, knock_direction, partition_idx, partition_sum, layer);

        self.patch_size = patch_size;
        self.stride = stride;
        self.multi_size = multi_size;
        self.ref_values = ref_values;


    def generate(self, x, y, **kwargs):
        """Generate knockoff samples and return them in an array.

        :param x: An array with the original inputs.
        :type x: `np.ndarray`
        :param y: Labels for generating saliency maps, not for generating knockoffs.
        :type y: `np.ndarray`
        :return: One array holding the knockoff upper/lower bounds, and one array holding the saliency upper/lower bounds (depend on knock_direction)
        :rtype: `np.ndarray tuple`
        """
        self.set_params(**kwargs);
        assert x.shape[3] == len(self.ref_values);

        saliency_x = self.saliency_gen.generate(x, y);
        layer_x = self.classifier.get_activations(x, self.layer);

        knockoff_x_upper = x.copy(); knockoff_x_lower = x.copy();
        saliency_x_upper = saliency_x.copy(); saliency_x_lower = saliency_x.copy();

        rowcol_tuple_list = [];
        for rowidx in range(0, x.shape[1] - self.patch_size + 1, self.stride):
            for colidx in range(0, x.shape[2] - self.patch_size + 1, self.stride):
                rowcol_tuple_list.append((rowidx, colidx));

        np.random.shuffle(rowcol_tuple_list);
        logger.info('rowcol_tuple_list={}'.format(len(rowcol_tuple_list)));

        total_batch = int(np.ceil(len(rowcol_tuple_list) * 1.0 / self.multi_size)); t = time.time();
        for curr_batch in range(total_batch):
            if self.partition_idx >= 0 and (curr_batch) % self.partition_sum != self.partition_idx: continue;
            curr_rowcol_tuples = rowcol_tuple_list[curr_batch * self.multi_size:(curr_batch + 1) * self.multi_size];

            knockoff_x = x.copy();
            for channelIdx in range(knockoff_x.shape[3]):
                for rowidx, colidx in curr_rowcol_tuples:
                    knockoff_x[:, rowidx:rowidx + self.patch_size, colidx:colidx + self.patch_size, channelIdx] = self.ref_values[channelIdx];

            knockoff_saliency = self.saliency_gen.generate(knockoff_x, y);

            if self.knock_direction > 0:
                knockoff_x_upper = np.maximum(knockoff_x_upper, knockoff_x);
                saliency_x_upper = np.maximum(saliency_x_upper, knockoff_saliency);
            else:
                knockoff_x_lower = np.minimum(knockoff_x_lower, knockoff_x);
                saliency_x_lower = np.minimum(saliency_x_lower, knockoff_saliency);


        if self.knock_direction > 0: return knockoff_x_upper, saliency_x_upper;
        else: return knockoff_x_lower, saliency_x_lower;


    def set_params(self, **kwargs):
        """
        Take in a dictionary of parameters and apply knockoff-specific checks before saving them as attributes.

        :param layer: Layer for the intermediate representation
        :type layer: `int` or `str`

        """
        # Save attack-specific parameters
        super(ConstKnockoff, self).set_params(**kwargs);

        if self.patch_size <= 0:
            raise ValueError('The patch width/height `patch_size` has to be positive.');

        if self.stride <= 0:
            raise ValueError('The stride length `stride` has to be positive.');

        if self.patch_size < self.stride:
            raise ValueError('The patch width/height `patch_size` cannot be smaller than the stride length `stride`');

        if self.multi_size <= 0:
            raise ValueError('The patch number `multi_size` has to be positive.');

        return True;


    def get_params_str(self):
        """
        Get knockoff params str.
        :return: Knockoff params str
        :rtype: `str`
        """
        return 'const_layer{}_ref{}_multi{}_patch{}_stride{}_direct{}_partidx{}_partsum{}'.format(self.layer, ','.join([str(x) for x in self.ref_values]), self.multi_size, self.patch_size, self.stride, self.knock_direction, self.partition_idx, self.partition_sum);