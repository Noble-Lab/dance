import sys
sys.path.append(('..'))

from numpy.random import seed
import random
random.seed(1)
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score

from classifiers.tensorflow import TFClassifier
from saliencies.saliency_gradient import GradientSaliency
from saliencies.saliency_smoothgrad import SmoothGradSaliency
from saliencies.saliency_integratedgrad import IntegratedGradSaliency

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages


def normalized_saliency_by_topK(sal_x, K=0.2):
    sal_x_tmp = np.copy(sal_x)
    thred = int(83 * K)
    for idx in range(sal_x.shape[0]):
        tau = np.sort(sal_x[idx, ].flatten())[-thred]
        sal_x_tmp[idx, ][sal_x[idx, ] >= tau] = 1
        sal_x_tmp[idx, ][sal_x[idx, ] < tau] = 0

    return sal_x_tmp


def fidelity_new(X, y, classifier, saliency):
    predicted_arr = np.zeros(X.shape[0])
    predicted_mat = classifier.predict(saliency * X)
    for i in range(X.shape[0]):
        predicted_arr[i] = predicted_mat[i, y[i]]
    return -np.log(predicted_arr)


def fidelity(X, y, saliency, classifier):
    saliency = normalized_saliency_by_topK(saliency)
    fid = fidelity_new(X, y, classifier, saliency)
    return fid


def Noisyknockoff(x, y, saliency_gen, patch_size=1, stride=1, multi_size=1):

    saliency_x = saliency_gen.generate(x, y)

    knockoff_x_upper = x.copy()
    knockoff_x_lower = x.copy()
    saliency_x_upper = saliency_x.copy()
    saliency_x_lower = saliency_x.copy()

    idx_tuple_list = []
    for idx in range(0, x.shape[1] - patch_size + 1, stride):
        idx_tuple_list.append(idx)

    np.random.shuffle(idx_tuple_list)

    total_batch = int(np.ceil(len(idx_tuple_list) * 1.0 / multi_size))
    for curr_batch in range(total_batch):
        curr_rowcol_tuples = idx_tuple_list[curr_batch * multi_size:(curr_batch + 1) * multi_size]

        knockoff_x = x.copy()
        for idx in curr_rowcol_tuples:
            knockoff_x[:, idx] = np.random.normal(-3, 1)

        knockoff_saliency = saliency_gen.generate(knockoff_x, y)

        knockoff_x_upper = np.maximum(knockoff_x_upper, knockoff_x)
        saliency_x_upper = np.maximum(saliency_x_upper, knockoff_saliency)
        knockoff_x_lower = np.minimum(knockoff_x_lower, knockoff_x)
        saliency_x_lower = np.minimum(saliency_x_lower, knockoff_saliency)

    knockoff_saliency = saliency_x_upper - saliency_x_lower
    return knockoff_saliency


def Constknockoff(x, y, saliency_gen, ref_values, patch_size=1, stride=1, multi_size=1):

    saliency_x = saliency_gen.generate(x, y)

    knockoff_x_upper = x.copy()
    knockoff_x_lower = x.copy()
    saliency_x_upper = saliency_x.copy()
    saliency_x_lower = saliency_x.copy()

    idx_tuple_list = []
    for idx in range(0, x.shape[1] - patch_size + 1, stride):
        idx_tuple_list.append(idx)

    np.random.shuffle(idx_tuple_list)

    total_batch = int(np.ceil(len(idx_tuple_list) * 1.0 / multi_size))
    for curr_batch in range(total_batch):
        curr_rowcol_tuples = idx_tuple_list[curr_batch * multi_size:(curr_batch + 1) * multi_size]

        knockoff_x = x.copy()
        for idx in curr_rowcol_tuples:
            knockoff_x[:, idx] = ref_values[idx]

        knockoff_saliency = saliency_gen.generate(knockoff_x, y)

        knockoff_x_upper = np.maximum(knockoff_x_upper, knockoff_x)
        saliency_x_upper = np.maximum(saliency_x_upper, knockoff_saliency)
        knockoff_x_lower = np.minimum(knockoff_x_lower, knockoff_x)
        saliency_x_lower = np.minimum(saliency_x_lower, knockoff_saliency)

    knockoff_saliency = saliency_x_upper - saliency_x_lower
    return knockoff_saliency


def Optkonckoff(x, y, classifier, knock_direction, layer, saliency_gen, patch_size=1, stride=1,
                multi_size=1, init_C=1e4, tol=1e-2, delta=0, max_iter=30):

    _tanh_smoother = 0.999999
    decrease_factor = 0.96

    def _original_to_tanh(x_original, clip_min, clip_max):
        x_tanh = np.clip(x_original, clip_min, clip_max)
        x_tanh = (x_tanh - clip_min) / (clip_max - clip_min)
        x_tanh = np.arctanh(((x_tanh * 2) - 1) * _tanh_smoother)
        return x_tanh

    def _tanh_to_original(x_tanh, clip_min, clip_max):
        x_original = (np.tanh(x_tanh) / _tanh_smoother + 1) / 2
        return x_original * (clip_max - clip_min) + clip_min

    def _loss(x, x_knockoff, layer_x, direction, tau, delta):
        gain_x = np.sum(np.maximum((x_knockoff - x) * direction, delta), axis=tuple(range(1, len(x.shape))))
        layer_x_knockoff = classifier.get_activations(x_knockoff, layer)
        assert layer_x.shape == layer_x_knockoff.shape
        loss_layer = np.max(np.maximum(np.square(layer_x - layer_x_knockoff) - tau, 0),
                            axis=tuple(range(1, len(layer_x.shape))))
        assert len(loss_layer) == x.shape[0]
        return gain_x, loss_layer

    def _loss_gradient(x, x_knockoff, x_knockoff_tanh, layer_x, c, direction, tau, delta):
        grad_x = np.maximum((x_knockoff - x) * direction, delta)
        grad_x[grad_x > delta] = direction
        grad_layer = classifier.layer_gradient_tau(x_knockoff, layer, layer_x, tau)

        c_mult = c
        for _ in range(len(x.shape) - 1): c_mult = c_mult[:, np.newaxis]

        loss_gradient = -grad_x + c_mult * grad_layer
        loss_gradient *= (1 - np.square(np.tanh(x_knockoff_tanh))) / (2 * _tanh_smoother)
        return loss_gradient

    def get_knockoff(x, layer_x, mask_x):
        clip_min, clip_max = classifier.clip_values
        c = init_C * np.ones(x.shape[0])
        tau = 1.0

        curr_x = x + 1e-4 * knock_direction * clip_max
        curr_x = mask_x * curr_x + (1 - mask_x) * x
        curr_x_tanh = _original_to_tanh(curr_x, clip_min, clip_max)

        curr_xgain, curr_layerloss = _loss(x, curr_x, layer_x, knock_direction, tau, delta)
        curr_xgain_mean = np.mean(curr_xgain)
        curr_layerloss_mean = np.mean(curr_layerloss)

        for iter in range(max_iter):

            prev_x_tanh = curr_x_tanh.copy()
            active_set = set()
            perturbation_tanh = -_loss_gradient(x, curr_x, curr_x_tanh, layer_x, c, knock_direction, tau, delta)

            for lr in np.linspace(-3, -1, 5):
                lr_mult = np.float_power(10, lr) * np.ones(x.shape[0])
                for _ in range(len(x.shape) - 1): lr_mult = lr_mult[:, np.newaxis]

                new_x_tanh = prev_x_tanh + lr_mult * perturbation_tanh
                new_x = mask_x * _tanh_to_original(new_x_tanh, clip_min, clip_max) + (1 - mask_x) * x
                new_xgain, new_layerloss = _loss(x, new_x, layer_x, knock_direction, tau, delta)

                active = (new_xgain - curr_xgain >= 0) & (new_layerloss <= tol)
                curr_xgain[active] = new_xgain[active]
                curr_layerloss[active] = new_layerloss[active]
                curr_x[active] = new_x[active]
                active_set = active_set.union(active)

            tau = max(tau * decrease_factor, tol)

            curr_x = mask_x * curr_x + (1 - mask_x) * x
            curr_x_tanh = _original_to_tanh(curr_x, clip_min, clip_max)
            if np.fabs(curr_xgain_mean - np.mean(curr_xgain)) <= 0: break
            curr_xgain_mean = np.mean(curr_xgain)
            curr_layerloss_mean = np.mean(curr_layerloss)
        return curr_x

    saliency_x = saliency_gen.generate(x, y)
    layer_x = classifier.get_activations(x, layer)

    knockoff_x_upper = x.copy(); knockoff_x_lower = x.copy()
    saliency_x_upper = saliency_x.copy()
    saliency_x_lower = saliency_x.copy()

    idx_tuple_list = []
    for idx in range(0, x.shape[1] - patch_size + 1, stride):
        idx_tuple_list.append(idx)

    np.random.shuffle(idx_tuple_list)
    total_batch = int(np.ceil(len(idx_tuple_list) * 1.0 / multi_size))
    for curr_batch in range(total_batch):
        curr_rowcol_tuples = idx_tuple_list[curr_batch * multi_size:(curr_batch + 1) * multi_size]
        mask = np.zeros_like(x)

        for idx in curr_rowcol_tuples:
            mask[:, idx] = 1

        knockoff_x = get_knockoff(x, layer_x, mask)
        knockoff_saliency = saliency_gen.generate(knockoff_x, y)

        if knock_direction > 0:
            knockoff_x_upper = np.maximum(knockoff_x_upper, knockoff_x)
            saliency_x_upper = np.maximum(saliency_x_upper, knockoff_saliency)
        else:
            knockoff_x_lower = np.minimum(knockoff_x_lower, knockoff_x)
            saliency_x_lower = np.minimum(saliency_x_lower, knockoff_saliency)

    if knock_direction > 0:
        return knockoff_x_upper, saliency_x_upper
    else:
        return knockoff_x_lower, saliency_x_lower

# load data.
data = np.load("../../ids/data/ids_data.npz")
X_train, y_train, X_test, y_test = data['X_train'], data['y_train'], data['X_test'], data['y_test']
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# Define and train a tf classifier.
input_ph = tf.placeholder(tf.float32, shape=[None, 83])
output_ph = tf.placeholder(tf.int32, shape=[None, 4])

dense = tf.layers.dense(input_ph, 50, activation=tf.nn.sigmoid)
drop = tf.layers.Dropout(0.25)(dense)
dense = tf.layers.dense(drop, 30, activation=tf.nn.sigmoid)
drop = tf.layers.Dropout(0.25)(dense)
logits = tf.layers.dense(drop, 4)

loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=output_ph))
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

ids_classifier = TFClassifier((0, 1), input_ph, logits, output_ph, train=train, loss=loss, sess=sess)
sess.run(tf.global_variables_initializer())

ids_classifier.load('../../ids/model/ids')

y_pred = ids_classifier.predict(X_test)
print(accuracy_score(y_pred=np.argmax(y_pred, axis=1), y_true=np.argmax(y_test, axis=1)))

y_pred = ids_classifier.predict(X_train)
print(accuracy_score(y_pred=np.argmax(y_pred, axis=1), y_true=np.argmax(y_train, axis=1)))

# compute three types of saliency maps and conduct fidelity test.
X_test_tmp = np.load('../../ids/explain_data.npz')['x']
y_test_tmp = np.load('../../ids/explain_data.npz')['y']

X_test_tmp = X_test_tmp[np.where(np.max(X_test_tmp, axis=1)<2), ][0,][0:100,]
y_test_tmp = y_test_tmp[np.where(np.max(X_test_tmp, axis=1)<2)[0], ][0:100,]


## original saliency
saliency1 = GradientSaliency(ids_classifier, normalize=False)
saliency1_x = saliency1.generate(X_test_tmp, y_test_tmp)
fid = fidelity(X_test_tmp, y_test_tmp, saliency1_x, ids_classifier)

saliency2 = IntegratedGradSaliency(ids_classifier, normalize=False, x_steps=5)
saliency2_x = saliency2.generate(X_test_tmp, y_test_tmp)
fid_1 = fidelity(X_test_tmp, y_test_tmp, saliency2_x, ids_classifier)

saliency3 = SmoothGradSaliency(ids_classifier, normalize=False, nsamples=5, stdev_spread=1e-1, magnitude=False) # random mean: 0.5
saliency3_x = saliency3.generate(X_test_tmp, y_test_tmp)
fid_2 = fidelity(X_test_tmp, y_test_tmp, saliency3_x, ids_classifier)


## Constant knockoff
# ref_values = np.mean(X_test_tmp, axis=0)
ref_values = np.zeros((83,)) - 3

constsaliency1 = Constknockoff(X_test_tmp, y_test_tmp, saliency1, ref_values)
const_fid = fidelity(X_test_tmp, y_test_tmp, constsaliency1, ids_classifier)

constsaliency2 = Constknockoff(X_test_tmp, y_test_tmp, saliency2, ref_values)
const_fid_1 = fidelity(X_test_tmp, y_test_tmp, constsaliency2, ids_classifier)

constsaliency3 = Constknockoff(X_test_tmp, y_test_tmp, saliency3, ref_values)
const_fid_2 = fidelity(X_test_tmp, y_test_tmp, constsaliency3, ids_classifier)


## Noise knockoff
noisysaliency1 = Noisyknockoff(X_test_tmp, y_test_tmp, saliency1)
noisy_fid = fidelity(X_test_tmp, y_test_tmp, noisysaliency1, ids_classifier)

noisysaliency2 = Noisyknockoff(X_test_tmp, y_test_tmp, saliency2)
noisy_fid_1 = fidelity(X_test_tmp, y_test_tmp, noisysaliency2, ids_classifier)

noisysaliency3 = Noisyknockoff(X_test_tmp, y_test_tmp, saliency3)
noisy_fid_2 = fidelity(X_test_tmp, y_test_tmp, noisysaliency3, ids_classifier)


# Optimization knockoff
ref_values = np.zeros((83,))

_, knocksaliencyupper_1 = Optkonckoff(X_test_tmp, y_test_tmp, ids_classifier, 1, layer=1,
                                                 saliency_gen=saliency1, tol=1e-5)
_, knocksaliencylower_1 = Optkonckoff(X_test_tmp, y_test_tmp, ids_classifier, -1, layer=1,
                                                 saliency_gen=saliency1, tol=1e-5)
knocksaliency1 = knocksaliencyupper_1 - knocksaliencylower_1

k_fid = fidelity(X_test_tmp, y_test_tmp, knocksaliency1, ids_classifier)

_, knocksaliencyupper_2 = Optkonckoff(X_test_tmp, y_test_tmp, ids_classifier, 1, layer=2,
                                                 saliency_gen=saliency2, tol=1e-1)
_, knocksaliencylower_2 = Optkonckoff(X_test_tmp, y_test_tmp, ids_classifier, -1, layer=2,
                                                 saliency_gen=saliency2, tol=1e-1)
knocksaliency2 = knocksaliencyupper_2 - knocksaliencylower_2

k_fid_1 = fidelity(X_test_tmp, y_test_tmp, knocksaliency2, ids_classifier)

_, knocksaliencyupper_3 = Optkonckoff(X_test_tmp, y_test_tmp, ids_classifier, 1, layer=2,
                                                 saliency_gen=saliency3, tol=1e-1)
_, knocksaliencylower_3 = Optkonckoff(X_test_tmp, y_test_tmp, ids_classifier, -1, layer=2,
                                                 saliency_gen=saliency3, tol=1e-1)
knocksaliency3 = knocksaliencyupper_3 - knocksaliencylower_3

k_fid_2 = fidelity(X_test_tmp, y_test_tmp, knocksaliency3, ids_classifier)


# Draw figures
# label_list = []
# fidelity_list = []
# decoy_list = []
# nn = mean_fid.shape[0]

# for saliency_type in ['Grad', 'IntGrad', 'SGrad']:

#     for i in range(nn):
#         label_list.append(saliency_type)
#         if saliency_type == 'Grad':
#             fidelity_list.append(fid[i])
#         if saliency_type == 'IntGrad':
#             fidelity_list.append(fid_1[i])
#         if saliency_type == 'SGrad':
#             fidelity_list.append(fid_2[i])
#         decoy_list.append('Without decoys')

#     for i in range(nn):
#         label_list.append(saliency_type)
#         if saliency_type == 'Grad':
#             fidelity_list.append(k_fid[i])
#         if saliency_type == 'IntGrad':
#             fidelity_list.append(k_fid_1[i])
#         if saliency_type == 'SGrad':
#             fidelity_list.append(k_fid_2[i])
#         decoy_list.append('Decoys w/ range aggregation')

#     for i in range(nn):
#         label_list.append(saliency_type)
#         if saliency_type == 'Grad':
#             fidelity_list.append(const_fid[i])
#         if saliency_type == 'IntGrad':
#             fidelity_list.append(const_fid_1[i])
#         if saliency_type == 'SGrad':
#             fidelity_list.append(const_fid_2[i])
#         decoy_list.append('Constant-perturbation w/ range aggregation')

#     for i in range(nn):
#         label_list.append(saliency_type)
#         if saliency_type == 'Grad':
#             fidelity_list.append(noisy_fid[i])
#         if saliency_type == 'IntGrad':
#             fidelity_list.append(noisy_fid_1[i])
#         if saliency_type == 'SGrad':
#             fidelity_list.append(noisy_fid_2[i])
#         decoy_list.append('Noise w/ range aggregation')

#     for i in range(nn):
#         label_list.append(saliency_type)
#         if saliency_type == 'Grad':
#             fidelity_list.append(mean_fid[i])
#         if saliency_type == 'IntGrad':
#             fidelity_list.append(mean_fid_1[i])
#         if saliency_type == 'SGrad':
#             fidelity_list.append(mean_fid_2[i])
#         decoy_list.append('Decoys w/ mean aggregation')

# data_pd = pd.DataFrame({'Fidelity': fidelity_list, 'Label': label_list, 'Decoy': decoy_list})
# figure = plt.figure(figsize=(20, 6))
# ax = sns.boxplot(x="Label", y="Fidelity", hue="Decoy", data=data_pd, whis=2,
#                  hue_order=['Without decoys', 'Decoys w/ range aggregation',
#                             'Constant-perturbation w/ range aggregation', 'Noise w/ range aggregation',
#                             'Decoys w/ mean aggregation'])

# ax.legend(loc='upper left', bbox_to_anchor=(1, 0.5), prop={'size': 30})
# ax.set_ylabel('Fidelity', fontsize=35)
# ax.set_xlabel('')
# ax.tick_params(axis='both', which='major', labelsize=35)
# pp = PdfPages('../../ids/fidelity.pdf')
# pp.savefig(figure, bbox_inches='tight')
# pp.close()

