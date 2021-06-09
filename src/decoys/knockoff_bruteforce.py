from __future__ import absolute_import, division, print_function, unicode_literals

import os, sys, random, six, time
sys.path.append('..')

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import numpy as np

from knockoffs.knockoff import Knockoff
from classifiers.classifier import Classifier

class BruteforceKnockoff(Knockoff):
    """
    1. We generate the knockoff pixel by pixel.
    2. We use Differential Evolution, a population based optimization method to search the knockoff (without gradient descent),
        similar to "One Pixel Attack for Fooling Deep Neural Networks"
    3. We only consider the knockoff x to be a tunable range around x. In other words, we calculate the upper bound and lower bound of x
    4. We use mean-square-error to measure the distance between intermediate representation f(x) and f(x_ref).
    """
    knockoff_params = Knockoff.knockoff_params + ['layer', 'tol', 'step_size']

    def __init__(self, classifier, saliency_gen, knock_direction, partition_idx, partition_sum, layer, tol=1e-3, step_size=5e-3):
        """
        Initialization specifically for the BruteforceKnockoff.

        :param classifier: A trained model.
        :type classifier: :class:`Classifier`
        :param saliency_gen: The saliency generator for knockoffs
        :type saliency_gen: `Saliency`
        :param layer: Layer for the intermediate representation
        :type layer: `int` or `str`
        :param tol: Tolarence for intermediate representation difference
        :type tol: `float`
        :param step_size: unit step_size for evolution searching
        :type step_size: `float`
        """
        super(BruteforceKnockoff, self).__init__(classifier, saliency_gen, knock_direction, partition_idx, partition_sum, layer);

        self.tol = tol;
        self.step_size = step_size;

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

        clip_min, clip_max = self.classifier.clip_values;
        saliency_x = self.saliency_gen.generate(x, y);
        layer_x = self.classifier.get_activations(x, self.layer);

        knockoff_x_upper = x.copy(); knockoff_x_lower = x.copy();
        saliency_x_upper = saliency_x.copy(); saliency_x_lower = saliency_x.copy();

        t = time.time();
        for channelIdx in range(x.shape[3]):
            for rowidx in range(x.shape[1]):
                for colidx in range(x.shape[2]):
                    if self.partition_idx >= 0 and (rowidx + colidx) % self.partition_sum != self.partition_idx: continue;

                    mask = np.zeros(x[0].shape);
                    mask[rowidx, colidx, channelIdx] = 1;
                    mask_x = np.repeat(mask[np.newaxis, :, :, :], x.shape[0], axis=0);

                    prev_x = x.copy();
                    prev_sum = np.sum(prev_x);
                    while True:
                        curr_x = prev_x + self.knock_direction * self.step_size * clip_max;
                        curr_x = mask_x * curr_x + (1 - mask_x) * x;
                        curr_x = np.clip(curr_x, clip_min, clip_max);

                        # loss_intermediate = self.classifier.layer_loss(curr_x, self.layer, layer_x);
                        layer_x_knockoff = self.classifier.get_activations(curr_x, self.layer); assert layer_x.shape == layer_x_knockoff.shape;
                        loss_layer = np.max(np.square(layer_x - layer_x_knockoff), axis=tuple(range(1, len(layer_x.shape))));

                        valid_sample_indices = np.where(loss_layer <= self.tol)[0];
                        if len(valid_sample_indices) <= 0: break;

                        prev_x[valid_sample_indices, rowidx, colidx, channelIdx] = curr_x[valid_sample_indices, rowidx, colidx, channelIdx];
                        if np.fabs(np.sum(prev_x) - prev_sum) <= 0: break;
                        prev_sum = np.sum(prev_x);

                    gain_x_mean = np.sum(np.maximum((prev_x - x) * self.knock_direction, 0), axis=tuple(range(1, len(x.shape))));

                    knockoff_saliency = self.saliency_gen.generate(prev_x, y);
                    if self.knock_direction > 0:
                        knockoff_x_upper = np.maximum(knockoff_x_upper, prev_x);
                        saliency_x_upper = np.maximum(saliency_x_upper, knockoff_saliency);
                    else:
                        knockoff_x_lower = np.minimum(knockoff_x_lower, prev_x);
                        saliency_x_lower = np.minimum(saliency_x_lower, knockoff_saliency);

                    logger.info('[t={:.2f}]channel={}\trowidx={}\tcolidx={}\txgain={}\tlayerloss={}'.format(time.time() - t, channelIdx, rowidx, colidx, np.mean(gain_x_mean), np.mean(loss_layer) ));

        if self.knock_direction > 0: return knockoff_x_upper, saliency_x_upper;
        else: return knockoff_x_lower, saliency_x_lower;


    def set_params(self, **kwargs):
        """
        Take in a dictionary of parameters and apply knockoff-specific checks before saving them as attributes.

        :param layer: Layer for the intermediate representation
        :type layer: `int` or `str`
        :param tol: Tolarence for intermediate representation difference
        :type tol: `float`
        :param step_size: step_size for evolution searching
        :type step_size: `float`
        """
        # Save attack-specific parameters
        super(BruteforceKnockoff, self).set_params(**kwargs);

        if self.tol <= 0:
            raise ValueError('The tolarence `tol` has to be positive.');

        if self.step_size <= 0:
            raise ValueError('The step size `step_size` has to be positive.');

        return True;


    def get_params_str(self):
        """
        Get knockoff params str.
        :return: Knockoff params str
        :rtype: `str`
        """
        return 'bf_layer{}_tol{}_step{}_direct{}_partidx{}_partsum{}'.format(self.layer, self.tol, self.step_size, self.knock_direction, self.partition_idx, self.partition_sum);