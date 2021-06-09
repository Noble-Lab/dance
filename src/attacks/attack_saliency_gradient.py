from __future__ import absolute_import, division, print_function, unicode_literals

import os, sys, random
sys.path.append('..')

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from scipy import stats
import numpy as np
from classifiers.tensorflow import TFClassifier
from attacks.attack_saliency import Attack_Saliency
from sklearn.metrics import accuracy_score

class Attack_Saliency_Gradient(Attack_Saliency):
    """
    A attack class that computes an attack image to gradient saliency method.
    """
    attacks_params = Attack_Saliency.attacks_parameters + ['method', 'dataset', 'epsilon', 'epochs', 'alpha', 'measure',
                                                           'target', 'logits', 'positive_only']

    def __init__(self, classifier, method, dataset, epsilon, k_top = 100, epochs=100, alpha=1,  measure="intersection",
                 target = None, logits = False, positive_only = False,):
        """
        :param classifier: A trained model.
        :type classifier: class:`Classifier`
        :param method: An attack method: "mass_center", "topK" or "random"
        :type method: `string`
        :param dataset: Dataset
        :type dataset: `string`
        :param epsilon: epsilon
        :type epsilon: `float`
        :param k_top: Top k features for the topK attack.
        :type k_top: `int`
        :param epsilon: Allowed maximum l_infty perturbation.
        :type epsilon: `int`
        :param epochs: Number of iteration.
        :type epochs: `int`
        :param alpha: Step size in each iteration.
        :type alpha: `int`
        :param measure: Measurement of the success of the attack: "correlation", "mass_center" or "intersection".
        :type measure: `string`
        :param target: target images for target attack.
        :type target: 'numpy.array'
        :param logits: Using the logits as outputs.
        :type logits: `bool`
        :param positive_only: Attacking only the positive features.
        :type positive_only: `bool`
        """
        super(Attack_Saliency_Gradient, self).__init__(classifier)
        self.k_top = k_top
        self.method = method
        self.dataset = dataset
        self.epsilon = epsilon
        self.clip_min, self.clip_max = self.classifier.clip_values
        self.epochs = epochs
        self.alpha = alpha
        self.measure = measure
        self.target = target
        self.logits = logits
        self.positive_only = positive_only
        if isinstance(self.classifier, TFClassifier):
            self.using_tf = True
        else:
            self.using_tf = False

    def generate(self, x, y, **kwargs):
        """
        Generate attacks on saliencies and return them as an array.

        :param x: Sample input with shape as expected by the classifier.
        :type x: `np.ndarray`
        :param y: Correct labels.
        :type y: `np.ndarray`
        :param method: An attack method: "mass_center", "topK" or "random"
        :type method: `string`
        :param epsilon: Allowed maximum l_infty perturbation.
        :type epsilon: `float`
        :param epochs: Number of iteration.
        :type epochs: `int`
        :param alpha: Step size in each iteration.
        :type alpha: `int`
        :param measure: Measurement of the success of the attack: "correlation", "mass_center" or "intersection".
        :type measure: `string`
        :param target: target images for target attack.
        :type target: 'numpy.array'
        :param logits: Using the logits as outputs.
        :type logits: `bool`
        :param positive_only: Attacking only the positive features.
        :type positive_only: `bool`
        :return: An array holding the attack inputs.
        :rtype: `np.ndarray`
        """
        if self.method == "mass_center":
            perturbed_x = x.copy() + 1e-4
        else:
            perturbed_x = x.copy()
        min_criterion = 1.
        perturb_size = 0.
        if self.using_tf == True:
            saliency_x, saliency_flatten_x, topK_x, mass_center_x = \
                self.classifier.attack_saliency_gradient(self.method, x, y, k_top=self.k_top,
                                                         target_map=self.target,logits=self.logits,
                                                         positive_only=self.positive_only)
        else:
            saliency_x, saliency_flatten_x, topK_x, mass_center_x, saliency_func, perturbation_func = \
                self.classifier.attack_saliency_gradient(self.method, x, y, k_top=self.k_top,
                                                         target_map=self.target,logits=self.logits,
                                                         positive_only=self.positive_only)
        for iter in range(self.epochs):
            if (iter+1) % int(self.epochs/10) == 0:
                print("Iteration : %d" %(iter+1))
            # Updating X
            if self.method == "topK":
                if self.using_tf == True:
                    fd = {self.classifier._input_ph: perturbed_x}
                    perturbation = self.classifier._sess.run(self.classifier._topK_direction, feed_dict=fd)
                else:
                    perturbation = perturbation_func([perturbed_x])
            elif self.method == "mass_center":
                if self.using_tf == True:
                    fd = {self.classifier._input_ph: perturbed_x}
                    perturbation = self.classifier._sess.run(self.classifier._mass_center_direction, feed_dict=fd)
                else:
                    perturbation = perturbation_func([perturbed_x])
            elif self.method == "target":
                if self.target is None:
                    raise ValueError("No target region determined!")
                else:
                    if self.using_tf == True:
                        fd = {self.classifier._input_ph: perturbed_x}
                        perturbation = self.classifier._sess.run(self.classifier._target_direction, feed_dict=fd)
                    else:
                        perturbation = perturbation_func([perturbed_x])
            else:
                print("Using random perturbation")
                perturbation = np.random.normal(size=x.shape)
            pert = np.reshape(perturbation, x.shape)
            perturbed_x = x + np.clip(perturbed_x + self.alpha * np.sign(pert) - x, -self.epsilon, self.epsilon)
            perturbed_x = np.clip(perturbed_x, self.clip_min, self.clip_max)

            # Checking prediction
            probs_idx = np.argmax(self.classifier.predict(perturbed_x), axis=1)
            acc = accuracy_score(probs_idx, y)
            probs_1 = self.classifier.predict(x)
            acc_1 = accuracy_score(np.argmax(probs_1, axis=1), y)

            flag = 0
            if self.dataset == 'imagenet':
                top5_index = np.argsort(probs_1, axis=1)[:,-5:]
                if (acc_1 - acc)/acc_1 < 0.5 and np.count_nonzero(top5_index - probs_idx[:,None]) <= top5_index.flatten().shape[0]-1:
                # if probs_idx[0] in top5_index[0]:
                    flag = 1
            else:
                if acc_1 == acc:
                    flag = 1

            if flag == 1:
                if self.measure == "intersection":
                    if self.using_tf == True:
                        fd = {self.classifier._input_ph: perturbed_x}
                        perturbed_topK = self.classifier._sess.run(self.classifier._top_idx, feed_dict = fd)
                    else:
                        _, _, perturbed_topK, _ = saliency_func([perturbed_x])
                    criterion = np.mean([float(len(np.intersect1d(topK_x[i], perturbed_topK[i])))
                                         / self.k_top for i in range(x.shape[0])])
                elif self.measure == "correlation":
                    if self.using_tf == True:
                        fd = {self.classifier._input_ph: perturbed_x}
                        perturbed_saliency_flatten = self.classifier._sess.run(self.classifier._saliency_flatten,
                                                                               feed_dict=fd)
                    else:
                        _, perturbed_saliency_flatten, _, _ = saliency_func([perturbed_x])
                    criterion = [stats.spearmanr(saliency_flatten_x[i,], perturbed_saliency_flatten[i,])[0]
                                 for i in range(x.shape[0])]
                    criterion = np.array(criterion)
                    criterion[np.isnan(criterion)] = 0

                elif self.measure == "mass_center":
                    if self.using_tf == True:
                        fd = {self.classifier._input_ph: perturbed_x}
                        perturbed_mass = self.classifier._sess.run(self.classifier._mass_center, feed_dict=fd)
                    else:
                        _, _, _, perturbed_mass = saliency_func([perturbed_x])
                    criterion = np.mean(-np.linalg.norm((mass_center_x - perturbed_mass), axis=1))
                else:
                    raise ValueError("Invalid measure!")
            else:
                criterion = 1

            if criterion < min_criterion:
                min_criterion = criterion
                perturbed_out = perturbed_x.copy()
                perturb_size = np.max(np.abs(x-perturbed_out))

        if min_criterion==1.:
            print("The attack was not successfull")
            perturbed_out = perturbed_x.copy()
            perturb_size = np.max(np.abs(x - perturbed_out))
            if self.using_tf == True:
                final_saliency, final_saliency_flatten, final_topK, final_mass_center  = \
                    self.classifier._sess.run([self.classifier._saliency, self.classifier._saliency_flatten,
                                               self.classifier._top_idx,self.classifier._mass_center],
                                              feed_dict={self.classifier._input_ph:perturbed_out})
            else:
                final_saliency, final_saliency_flatten, final_topK, final_mass_center = saliency_func([perturbed_out])
            correlation = [1]*x.shape[0]
            intersection = [1]*x.shape[0]
            center_dislocation = [0]*x.shape[0]
        else:
            final_prob = self.classifier.predict(perturbed_out)
            acc = accuracy_score(y, np.argmax(final_prob, axis = 1))
            if self.using_tf == True:
                final_saliency, final_saliency_flatten, final_topK, final_mass_center  = \
                    self.classifier._sess.run([self.classifier._saliency, self.classifier._saliency_flatten,
                                               self.classifier._top_idx,self.classifier._mass_center],
                                               feed_dict={self.classifier._input_ph:perturbed_out})
            else:
                final_saliency, final_saliency_flatten, final_topK, final_mass_center = saliency_func([perturbed_out])
            correlation = np.array([stats.spearmanr(saliency_flatten_x[i,], final_saliency_flatten[i,])[0] for i in range(x.shape[0])])
            correlation[np.isnan(correlation)] = 0
            intersection = [float(len(np.intersect1d(topK_x[i], final_topK[i]))) / self.k_top for i in range(x.shape[0])]
            center_dislocation = -np.linalg.norm((mass_center_x - final_mass_center), axis=1)
            print("Attack succeed with the classification accuracy = %.5f, perturbation = %.5f, mean correlation = %.5f,"
                  " mean intersection = %.5f, mean center_dislocation = %.5f" %(acc, perturb_size, np.mean(correlation),
                                                                               np.mean(intersection),
                                                                               np.mean(center_dislocation)))
        if self.positive_only == True:
            final_saliency = np.maximum(final_saliency, 0)
        else:
            final_saliency = np.abs(final_saliency)
        final_saliency[np.isnan(final_saliency)] = 1
        sal_x_max = np.max(final_saliency, axis=(1,2))
        sal_x_max[sal_x_max == 0] = 1e-16
        final_saliency = final_saliency / sal_x_max[:, None, None,]
        return perturbed_out, final_saliency, correlation, intersection, center_dislocation