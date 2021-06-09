from __future__ import absolute_import, division, print_function, unicode_literals

import os, sys, random, six, time
sys.path.append('..')

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import numpy as np

from knockoffs.knockoff import Knockoff
from classifiers.classifier import Classifier

class TransformLInfMultiPatchKnockoff(Knockoff):
    """
    1. We generate the knockoff patch by patch.
    2. We use Hinge Loss to measure the distance between knockoff x and input x_ref, i.e. max(x-x_ref, 0) or max(x_ref-x, 0)
    3. We only consider the knockoff x to be a tunable range around x. In other words, we calculate the upper bound and lower bound of x
    4. We use mean-square-error to measure the distance between intermediate representation f(x) and f(x_ref).
    5. The objective function to minimize_{x}:
        upperbound: -max(x-x_ref, 0) + C*|| f(x)-f(x_ref) ||^Inf
        lowerbound: -max(x_ref-x, 0) + C*|| f(x)-f(x_ref) ||^Inf
        s.t. 0 <= x <= 255
    """
    knockoff_params = Knockoff.knockoff_params + ['layer', 'eps', 'C', 'tol', 'max_iter', 'patch_size', 'stride'];

    def __init__(self, classifier, saliency_gen, knock_direction, partition_idx, partition_sum, layer, init_C=1e4, tol=1e-1, delta=0,
                 max_iter=30,  patch_size=2, stride=1, multi_size=10, repeat=2):
        """
        Initialization specifically for the TransformLInfMultiPatchKnockoff.

        :param classifier: A trained model.
        :type classifier: :class:`Classifier`
        :param layer: Layer for the intermediate representation
        :type layer: `int` or `str`
        :param max_iter: Maximum iteration for gradient descent
        :type max_iter: `int`
        :param patch_size: The width/height of the patch
        :type patch_size: `int`
        :param stride: The stride length of the patch, which is no larger than the patch_size
        :type stride: `int`
        :param multi_size: The number of patches optimized at once
        :type multi_size: `int`
        :param repeat: The number of repeating epoches
        :type repeat: `int`
        """
        super(TransformLInfMultiPatchKnockoff, self).__init__(classifier, saliency_gen, knock_direction, partition_idx, partition_sum, layer);
        self._tanh_smoother = 0.999999;
        self.decrease_factor = 0.96;

        self.init_C = init_C;
        self.tol = tol;
        self.delta = delta;

        self.max_iter = max_iter;
        self.patch_size = patch_size;
        self.stride = stride;
        self.multi_size = multi_size;
        self.repeat = repeat;


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

        # clip_min, clip_max = self.classifier.clip_values;
        saliency_x = self.saliency_gen.generate(x, y);
        layer_x = self.classifier.get_activations(x, self.layer);

        knockoff_x_upper = x.copy(); knockoff_x_lower = x.copy();
        saliency_x_upper = saliency_x.copy(); saliency_x_lower = saliency_x.copy();

        rowcol_tuple_list = [];
        for rowidx in range(0, x.shape[1] - self.patch_size + 1, self.stride):
            for colidx in range(0, x.shape[2] - self.patch_size + 1, self.stride):
                for _ in range(self.repeat):
                    rowcol_tuple_list.append((rowidx, colidx));
        np.random.shuffle(rowcol_tuple_list);
        logger.info('rowcol_tuple_list={}'.format(len(rowcol_tuple_list)));

        total_batch = int(np.ceil(len(rowcol_tuple_list) * 1.0 / self.multi_size)); t = time.time();
        for curr_batch in range(total_batch):
            if self.partition_idx >= 0 and (curr_batch) % self.partition_sum != self.partition_idx: continue;

            curr_rowcol_tuples = rowcol_tuple_list[curr_batch * self.multi_size:(curr_batch + 1) * self.multi_size];

            mask = np.zeros(x[0].shape);
            for rowidx, colidx in curr_rowcol_tuples: mask[rowidx:rowidx + self.patch_size, colidx:colidx + self.patch_size, :] = 1;
            mask_x = np.repeat(mask[np.newaxis, :, :, :], x.shape[0], axis=0);

            knockoff_x = self.get_knockoff(x, layer_x, mask_x);
            knockoff_saliency = self.saliency_gen.generate(knockoff_x, y);

            if self.knock_direction > 0:
                knockoff_x_upper = np.maximum(knockoff_x_upper, knockoff_x);
                saliency_x_upper = np.maximum(saliency_x_upper, knockoff_saliency);
            else:
                knockoff_x_lower = np.minimum(knockoff_x_lower, knockoff_x);
                saliency_x_lower = np.minimum(saliency_x_lower, knockoff_saliency);

            logger.info('[t={:.2f}]curr_batch={}\tknockoff_diff_upper={}\tknockoff_diff_lower={}'.format(time.time() - t, curr_batch, np.sum(knockoff_x_upper-x), np.sum(knockoff_x_lower-x)));

        if self.knock_direction > 0: return knockoff_x_upper, saliency_x_upper;
        else: return knockoff_x_lower, saliency_x_lower;


    def get_knockoff(self, x, layer_x, mask_x):
        clip_min, clip_max = self.classifier.clip_values;
        c = self.init_C * np.ones(x.shape[0]); tau = 1.0;

        curr_x = x + 1e-4 * self.knock_direction * clip_max;
        curr_x = mask_x * curr_x + (1 - mask_x) * x;
        curr_x_tanh = self._original_to_tanh(curr_x, clip_min, clip_max);

        curr_xgain, curr_layerloss = self._loss(x, curr_x, layer_x, self.knock_direction, tau, self.delta); curr_xgain_mean = np.mean(curr_xgain); curr_layerloss_mean = np.mean(curr_layerloss);
        # logger.info('origin curr_x={}\txgain={}\tlayerloss={}'.format(np.sum(curr_x), curr_xgain_mean, curr_layerloss_mean ));

        for iter in range(self.max_iter):

            prev_x_tanh = curr_x_tanh.copy(); active_set = set();
            perturbation_tanh = -self._loss_gradient(x, curr_x, curr_x_tanh, layer_x, c, self.knock_direction, tau, self.delta);

            #for lr in np.linspace(-5,-1, 9):
            for lr in np.linspace(-3, -1, 5):
                lr_mult = np.float_power(10, lr) * np.ones(x.shape[0]);
                for _ in range(len(x.shape) - 1): lr_mult = lr_mult[:, np.newaxis];

                new_x_tanh = prev_x_tanh + lr_mult * perturbation_tanh;
                new_x = mask_x * self._tanh_to_original(new_x_tanh, clip_min, clip_max) + (1 - mask_x) * x;
                new_xgain, new_layerloss = self._loss(x, new_x, layer_x, self.knock_direction, tau, self.delta);

                active = (new_xgain-curr_xgain >= 0) & (new_layerloss <= self.tol);
                curr_xgain[active] = new_xgain[active];
                curr_layerloss[active] = new_layerloss[active];
                curr_x[active] = new_x[active];
                active_set = active_set.union(active);

            tau = max(tau*self.decrease_factor, self.tol);

            curr_x = mask_x * curr_x + (1 - mask_x) * x;
            curr_x_tanh = self._original_to_tanh(curr_x, clip_min, clip_max);
            if np.fabs(curr_xgain_mean - np.mean(curr_xgain)) <= 0: break;
            curr_xgain_mean = np.mean(curr_xgain);
            curr_layerloss_mean = np.mean(curr_layerloss);
        # logger.info('iter={} curr_x={}\txgain={}\tlayerloss={}'.format(iter, np.sum(curr_x), curr_xgain_mean, curr_layerloss_mean ));

        logger.info('curr_x={}\txgain={}\tlayerloss={}'.format(np.sum(curr_x), curr_xgain_mean*self.knock_direction, curr_layerloss_mean));
        return curr_x


    def _loss(self, x, x_knockoff, layer_x, direction, tau, delta):
        gain_x = np.sum(np.maximum((x_knockoff - x) * direction, delta), axis=tuple(range(1, len(x.shape))));
        layer_x_knockoff = self.classifier.get_activations(x_knockoff, self.layer); assert layer_x.shape == layer_x_knockoff.shape;
        loss_layer = np.max(np.maximum(np.square(layer_x - layer_x_knockoff) - tau, 0), axis=tuple(range(1,len(layer_x.shape)))); assert len(loss_layer) == x.shape[0];
        return gain_x, loss_layer;

    def _loss_gradient(self, x, x_knockoff, x_knockoff_tanh, layer_x, c, direction, tau, delta):
        grad_x = np.maximum((x_knockoff - x) * direction, delta); grad_x[grad_x > delta] = direction;
        grad_layer = self.classifier.layer_gradient_tau(x_knockoff, self.layer, layer_x, tau);
        # logger.info('grad_layer={}\t{}'.format(grad_layer.shape, np.sum(grad_layer)));

        c_mult = c;
        for _ in range(len(x.shape) - 1): c_mult = c_mult[:, np.newaxis];

        loss_gradient = -grad_x + c_mult * grad_layer;
        loss_gradient *= (1 - np.square(np.tanh(x_knockoff_tanh))) / (2 * self._tanh_smoother);
        return loss_gradient;


    def _original_to_tanh(self, x_original, clip_min, clip_max):
        x_tanh = np.clip(x_original, clip_min, clip_max)
        x_tanh = (x_tanh - clip_min) / (clip_max - clip_min)
        x_tanh = np.arctanh(((x_tanh * 2) - 1) * self._tanh_smoother)
        return x_tanh


    def _tanh_to_original(self, x_tanh, clip_min, clip_max):
        x_original = (np.tanh(x_tanh) / self._tanh_smoother + 1) / 2
        return x_original * (clip_max - clip_min) + clip_min


    def set_params(self, **kwargs):
        """
        Take in a dictionary of parameters and apply knockoff-specific checks before saving them as attributes.

        :param layer: Layer for the intermediate representation
        :type layer: `int` or `str`
        :param eps: Step size for gradient descent
        :type eps: `float`
        :param C: Coefficient of the || f(x)-f(x_ref) ||^2 term
        :type C: `float`
        :param tol: Tolarence for loss difference
        :type tol: `float`
        :param max_iter: Maximum iteration for gradient descent
        :type max_iter: `int`
        :param patch_size: The width/height of the patch
        :type patch_size: `int`
        :param stride: The stride length of the patch, which is no larger than the patch_size
        :type stride: `int`
        :param multi_size: The number of patches optimized at once
        :type multi_size: `int`
        :param repeat: The number of repeating epoches
        :type repeat: `int`
        """
        # Save attack-specific parameters
        super(TransformLInfMultiPatchKnockoff, self).set_params(**kwargs);

        if self.init_C <= 0:
            raise ValueError('The `init_C` has to be positive.');

        if self.tol <= 0:
            raise ValueError('The tolarence `tol` has to be positive.');

        # if self.delta > 0:
        #     raise ValueError('The `delta` has to be non-positive.');

        if self.max_iter <= 0:
            raise ValueError('The iteration `max_iter` has to be positive.');

        if self.patch_size <= 0:
            raise ValueError('The patch width/height `patch_size` has to be positive.');

        if self.stride <= 0:
            raise ValueError('The stride length `stride` has to be positive.');

        if self.patch_size < self.stride:
            raise ValueError('The patch width/height `patch_size` cannot be smaller than the stride length `stride`');

        if self.multi_size <= 0:
            raise ValueError('The patch number `multi_size` has to be positive.');

        if self.repeat <= 0:
            raise ValueError('The repeat number `repeat` has to be positive.');

        return True;


    def get_params_str(self):
        """
        Get knockoff params str.
        :return: Knockoff params str
        :rtype: `str`
        """
        return 'tlinf_layer{}_rep{}_multi{}_patch{}_stride{}_C{}_tol{}_delta{}_direct{}_partidx{}_partsum{}'.format(self.layer, self.repeat, self.multi_size, self.patch_size, self.stride, self.init_C, self.tol, self.delta, self.knock_direction, self.partition_idx, self.partition_sum);


