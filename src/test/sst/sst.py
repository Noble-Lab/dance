import os
from numpy.random import seed
import random
random.seed(1)
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from matplotlib.backends.backend_pdf import PdfPages

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from classifiers.tensorflow import TFClassifier
from saliencies.saliency_gradient import GradientSaliency
from saliencies.saliency_smoothgrad import SmoothGradSaliency
from saliencies.saliency_integratedgrad import IntegratedGradSaliency


def normalized_saliency(sal_x, use_abs=False, normalize=True ):
    if use_abs: sal_x = np.abs(sal_x);

    if normalize:
        percent_x = np.percentile(sal_x, 95, axis=range(1, len(sal_x.shape)))
        for _ in range(len(sal_x.shape) - 1): percent_x = percent_x[:, np.newaxis]
        sal_x = np.minimum(sal_x, percent_x)

        if len(sal_x.shape) == 3:sal_x = np.expand_dims(sal_x, 0)
        sal_x_max = np.max(sal_x, axis=1)
        sal_x_max[sal_x_max==0] = np.finfo(np.float).eps
        sal_x = sal_x / sal_x_max[:, None]

    return sal_x


def fidelity(X, y, saliency, whole_word=False, K=0.1 ):
    predicted_arr = np.zeros(X.shape[0]);

    if K <= 0: saliency_topK = saliency;
    else:
        saliency_topK = np.zeros_like(saliency);
        if whole_word:
            saliency_collapsed = np.mean(saliency, axis=(2, 3));
            thred = int(saliency_collapsed.shape[-1] * K);
            for sample_idx in range(X.shape[0]):
                tau = np.sort(saliency_collapsed[sample_idx])[-thred];
                saliency_topK[sample_idx, saliency_collapsed[sample_idx] >= tau, :, :] = 1;

        else:
            thred = int(saliency.shape[1] * saliency.shape[2] * K);
            for sample_idx in range(X.shape[0]):
                tau = np.sort(saliency[sample_idx].flatten())[-thred];
                saliency_topK[sample_idx][saliency_topK[sample_idx] >= tau] = 1;

    predicted_mat = ids_classifier.predict(saliency_topK * X);
    for i in range(X.shape[0]): predicted_arr[i] = predicted_mat[i, y[i]];
    return -np.log(predicted_arr);


def get_word_saliency(saliency, topK=3):
    normed_saliency = normalized_saliency(saliency, use_abs=True, normalize=True )
    normed_saliency_collapsed = np.mean(normed_saliency, axis=(2, 3));

    saliency_topK = np.zeros_like(saliency);
    saliency_collapsed = np.mean(saliency, axis=(2, 3));
    for sample_idx in range(saliency.shape[0]):
        tau = np.sort(saliency_collapsed[sample_idx])[-topK];
        saliency_topK[sample_idx, saliency_collapsed[sample_idx] >= tau, :, :] = 1;
    topk_saliency_collapsed = np.mean(saliency_topK, axis=(2, 3))

    return normed_saliency_collapsed, topk_saliency_collapsed;

def postprecess_decoys(saliency_original, saliency_upper, saliency_lower, type=0):
    saliency_upper_new = np.zeros_like(saliency_upper);
    saliency_lower_new = np.zeros_like(saliency_lower);

    for sample_idx in range(saliency_original.shape[0]):
        for word_idx in range(saliency_original.shape[1]):
            saliency_original_arr = saliency_original[sample_idx, word_idx, :, 0];
            saliency_upper_arr = saliency_upper[sample_idx, word_idx, :, 0];
            saliency_lower_arr = saliency_lower[sample_idx, word_idx, :, 0];

            if type == 0:
                shift_upper = saliency_upper_arr - saliency_original_arr;
                shift_lower = saliency_lower_arr - saliency_original_arr;
            elif type == 1:
                shift_upper = np.quantile(saliency_upper_arr - saliency_original_arr, q=0.5);
                shift_lower = np.quantile(saliency_lower_arr - saliency_original_arr, q=0.5);
            elif type == 2:
                shift_upper = np.minimum(saliency_upper_arr - saliency_original_arr, 0.005);
                shift_lower = np.maximum(saliency_lower_arr - saliency_original_arr, -0.005);
            elif type == 3:
               pass
            else: assert False;

            saliency_upper_new[sample_idx, word_idx, :, 0] = saliency_original[sample_idx, word_idx, :, 0] + shift_upper;
            saliency_lower_new[sample_idx, word_idx, :, 0] = saliency_original[sample_idx, word_idx, :, 0] + shift_lower;

    return saliency_upper_new, saliency_lower_new;



def visualize_sentence_saliency(word_saliency_list, fid_tuple, label, save_url, annotat=True):
    word_list = [x[0] for x in word_saliency_list];
    if len(word_list) > 15: return;
    # fid_orig, fid_decoy = fid_tuple;

    save_url1 = save_url.replace('.pdf', '_fid_{:.5f}_{:.5f}_label{}_annt{}.pdf'.format(fid_tuple[0], fid_tuple[1], label, int(annotat)));
    saliency_list_orig = np.asarray([x[1] for x in word_saliency_list]); print(saliency_list_orig)
    saliency_list_decoy = np.asarray([x[2] for x in word_saliency_list]); print(saliency_list_decoy)

    all_saliency_mat = np.zeros((2, len(word_list)), dtype=float);
    all_saliency_mat[0, :] = saliency_list_orig;
    all_saliency_mat[1, :] = saliency_list_decoy;
    print('all_saliency_mat={}'.format(all_saliency_mat.shape));

    # diff_mat = np.zeros((1, len(word_list)), dtype=float);
    # diff_mat[0, :] = saliency_list_decoy - saliency_list_orig;

    all_word_mat = np.asarray([word_list, word_list]);
    print('all_word_mat={}'.format(all_word_mat.shape));

    figure = plt.figure(figsize=(20, 8));
    ax = figure.add_subplot(1, 1, 1);  ax.autoscale();
    if annotat:  ax = sns.heatmap(all_saliency_mat, annot=all_word_mat, fmt='', annot_kws={"size": 15},  cmap="Blues");
    else: ax = sns.heatmap(all_saliency_mat, cmap="Blues");
    ax.set_xticks(range(len(word_saliency_list)));
    ax.set_xticklabels(word_list,  fontsize=20);
    ax.set_title('label={}'.format(label));

    # ax = figure.add_subplot(2, 1, 2); ax.autoscale();
    # ax = sns.heatmap(diff_mat, cmap="bwr");
    # ax.axes.xaxis.set_visible(False)
    # ax.axes.yaxis.set_visible(False)

    pp = PdfPages(save_url1);
    pp.savefig(figure, bbox_inches='tight');
    pp.close()
    figure.clear();

    # plt.imsave(save_url1.replace('.pdf', '_diff.pdf'), saliency_list_decoy - saliency_list_orig, cmap='bwr');


def Optkonckoff(x, y, classifier, knock_direction, layer, saliency_gen, radius=0.1, init_C=1e4, tol=1e-1, delta=0, max_iter=40):

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
        # clip_min, clip_max = classifier.clip_values
        clip_min = x - radius; clip_max = x + radius;
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

            print('iter={} curr_x={}\txgain={}\tlayerloss={}'.format(iter, np.sum(curr_x), curr_xgain_mean, curr_layerloss_mean));

        print('curr_x={}\txgain={}\tlayerloss={}'.format(np.sum(curr_x), curr_xgain_mean * knock_direction, curr_layerloss_mean));
        return curr_x

    saliency_x = saliency_gen.generate(x, y)
    layer_x = classifier.get_activations(x, layer)

    knockoff_x_upper = x.copy(); knockoff_x_lower = x.copy()
    saliency_x_upper = saliency_x.copy(); saliency_x_lower = saliency_x.copy()

    print('x={}\tlayer_x={}'.format(x.shape, layer_x.shape ));

    for word_idx in range(x.shape[1]):
        print('word_idx={}\tknock_direction={}'.format(word_idx, knock_direction));

        mask = np.zeros_like(x);
        mask[:, word_idx, :, :] = 1;

        knockoff_x = get_knockoff(x, layer_x, mask);
        knockoff_saliency = saliency_gen.generate(knockoff_x, y);

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


data = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'runs', "test_data.npz"), allow_pickle=True)
X_id, X_test, y_test = data['x_test'], data['x_embedded'], data['y_test']
X_orig = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'runs', "test_sentences.npy"), allow_pickle=True)
id_to_vocab_map = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'runs', 'id_to_vocab_map.npy'), allow_pickle=True).item();

x_min = np.min(X_test); x_max = np.max(X_test);

print('X_id={}\t'.format(X_id.shape));
print('X_orig={}\t'.format(len(X_orig)));
print('X_test={}\t'.format(X_test.shape));
print('y_test={}\t'.format(y_test.shape));
print('id_to_vocab_map={}\t'.format(len(id_to_vocab_map)));
print('x_min={}\tx_max={}\tx_std={}'.format(x_min, x_max, np.std(X_test, axis=(0,1,3)) ));


var_all = np.load('parameter.npy', allow_pickle=True)

# # Define and train a tf classifier.
input_ph = tf.placeholder(tf.float32, shape=[None, 51, 100, 1])
output_ph = tf.placeholder(tf.int32, shape=[None, 2])

filter_shape = [3, 100, 1, 100]
W = tf.get_variable("W", shape=filter_shape,
                    trainable=True,
                    initializer=tf.constant_initializer(var_all[0]))
b = tf.get_variable("b", shape=[100],
                    trainable=True,
                    initializer=tf.constant_initializer(var_all[1]))

conv = tf.nn.conv2d(
    input_ph,
    W, strides=[1, 1, 1, 1], padding="VALID",
    name="conv")
# declare non linearity
h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
# self.conv_loss += tf.nn.l2_loss(W)
# Max pool relu output
pool = tf.nn.max_pool(h, ksize=[1, 51 - 3 + 1, 1, 1],
                      strides=[1, 1, 1, 1], padding="VALID", name="pool") # (?, 1, 1, 100)

h_pool_flat = tf.reshape(pool, [-1, 100]) # (?, 100)

fully_con_W = tf.get_variable("W_fc", shape=[100, 2],
                    trainable=True,
                    initializer=tf.constant_initializer(var_all[2]))
fully_con_b = tf.get_variable("b_fc", shape=[2],
                    trainable=True,
                    initializer=tf.constant_initializer(var_all[3]))

logits = tf.nn.xw_plus_b(h_pool_flat, fully_con_W, fully_con_b, name="scores")

loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=output_ph))
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

ids_classifier = TFClassifier((x_min, x_max), input_ph, logits, output_ph, train=train, loss=loss, sess=sess)

y_pred = ids_classifier.predict(X_test)
print(accuracy_score(y_pred=np.argmax(y_pred, axis=1), y_true=np.argmax(y_test, axis=1)))


embedding_size = 100;
### calculate mean embedding
word_id_set = set(); word_embedding_mat = np.empty((0, embedding_size), float)
for sample_idx in range(X_id.shape[0]):
    for word_idx in range(X_id.shape[1]):
        word_id = X_id[sample_idx, word_idx];
        word_embedding = X_test[sample_idx, word_idx, :, 0];

        if word_id <= 0 or word_id in word_id_set: continue;
        word_id_set.add(word_id);
        word_embedding_mat = np.vstack((word_embedding_mat, word_embedding));

word_embedding_mean = np.mean(word_embedding_mat, axis=0);
print('word_embedding_mat={}\tword_embedding_mean={}'.format(word_embedding_mat.shape, word_embedding_mean.shape ));


# use_abs = False; saliency_portion=0.2; normalize = True;
for use_abs in [False]:
    for normalize in [True]:
        for saliency_portion in [0.2]: # 0.2, 0.1

            for saliency_gen in [GradientSaliency(ids_classifier, normalize=False), IntegratedGradSaliency(ids_classifier, normalize=False), SmoothGradSaliency(ids_classifier, normalize=False)]:
                X_orig_saliency = saliency_gen.generate(X_test, np.argmax(y_test, axis=1));

                X_orig_saliency = normalized_saliency(X_orig_saliency, use_abs=use_abs, normalize=normalize);
                X_orig_saliency_collapsed = np.mean(X_orig_saliency, axis=(2,3));
                print('X_orig_saliency_collapsed={}\tmean={}\tabs_mean={}'.format(X_orig_saliency_collapsed.shape, np.mean(X_orig_saliency), np.mean(np.fabs(X_orig_saliency)),  ));

                # knockoff_opt_saliency_url = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'runs', 'knockoff_saliency_opt_{}_{}.npz'.format(radius, saliency_gen.name));
                # if os.path.isfile(knockoff_opt_saliency_url):
                #     data_opt = np.load(knockoff_opt_saliency_url, allow_pickle=True)
                #     X_knockoff_saliency_opt_upper = data_opt['knockoff_upper'];
                #     X_knockoff_saliency_opt_lower = data_opt['knockoff_lower'];
                # else:
                #     X_opt_upper, X_knockoff_saliency_opt_upper = Optkonckoff(X_test, np.argmax(y_test, axis=1), ids_classifier, +1, 3, saliency_gen, radius=radius);
                #     X_opt_lower, X_knockoff_saliency_opt_lower = Optkonckoff(X_test, np.argmax(y_test, axis=1), ids_classifier, -1, 3, saliency_gen, radius=radius);
                #     np.savez(knockoff_opt_saliency_url, x_upper=X_opt_upper, x_lower=X_opt_lower, knockoff_upper=X_knockoff_saliency_opt_upper, knockoff_lower=X_knockoff_saliency_opt_lower);
                #
                # X_optrange_knockoff_saliency = normalized_saliency(X_knockoff_saliency_opt_upper - X_knockoff_saliency_opt_lower, use_abs=use_abs, normalize=normalize);
                # X_optrange_knockoff_saliency_collapsed = np.mean(X_optrange_knockoff_saliency, axis=(2,3));
                # print('X_optrange_knockoff_saliency={}\tX_optrange_knockoff_saliency_collapsed={}'.format(X_optrange_knockoff_saliency.shape, X_optrange_knockoff_saliency_collapsed.shape ));
                #
                # X_optmean_knockoff_saliency = normalized_saliency(0.5*(X_knockoff_saliency_opt_upper + X_knockoff_saliency_opt_lower), use_abs=use_abs, normalize=normalize);
                # X_optmean_knockoff_saliency_collapsed = np.mean(X_optmean_knockoff_saliency, axis=(2,3));
                # print('X_optmean_knockoff_saliency={}\tX_optmean_knockoff_saliency_collapsed={}'.format(X_optmean_knockoff_saliency.shape, X_optmean_knockoff_saliency_collapsed.shape));


                knockoff_decoy_saliency_url = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'runs', 'knockoff_saliency_mean_{}.npz'.format(saliency_gen.name));
                if os.path.isfile(knockoff_decoy_saliency_url):
                    data_decoy = np.load(knockoff_decoy_saliency_url, allow_pickle=True)
                    X_knockoff_saliency_decoy_upper = data_decoy['knockoff_upper'];
                    X_knockoff_saliency_decoy_lower = data_decoy['knockoff_lower'];
                else:
                    X_knockoff_saliency_decoy_upper = X_orig_saliency.copy(); X_knockoff_saliency_decoy_lower = X_orig_saliency.copy();

                    for word_idx in range(X_id.shape[1]):
                        print('word_idx={}'.format(word_idx));

                        X_knockoff = X_test.copy();
                        for sample_idx in range(X_id.shape[0]): X_knockoff[sample_idx, word_idx, :, 0] = word_embedding_mean;
                        X_knockoff_saliency_decoy = saliency_gen.generate(X_knockoff, np.argmax(y_test, axis=1));

                        X_knockoff_saliency_decoy_upper = np.maximum(X_knockoff_saliency_decoy_upper, X_knockoff_saliency_decoy);
                        X_knockoff_saliency_decoy_lower = np.minimum(X_knockoff_saliency_decoy_lower, X_knockoff_saliency_decoy);
                    np.savez(knockoff_decoy_saliency_url, knockoff_upper=X_knockoff_saliency_decoy_upper, knockoff_lower=X_knockoff_saliency_decoy_lower);

                X_decoy_knockoff_saliency = X_knockoff_saliency_decoy_upper - X_knockoff_saliency_decoy_lower; assert np.min(X_decoy_knockoff_saliency) >= 0;
                X_decoy_knockoff_saliency = normalized_saliency(X_decoy_knockoff_saliency, use_abs=use_abs, normalize=normalize);
                print('X_decoy_knockoff_saliency={}\t'.format(X_decoy_knockoff_saliency.shape,  ));

                X_decoymean_knockoff_saliency = normalized_saliency(0.5*(X_knockoff_saliency_decoy_upper + X_knockoff_saliency_decoy_lower), use_abs=use_abs, normalize=normalize);
                print('X_decoymean_knockoff_saliency={}\t'.format(X_decoymean_knockoff_saliency.shape, ));


                knockoff_const_saliency_url = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'runs', 'knockoff_saliency_const_{}.npz'.format(saliency_gen.name));
                if os.path.isfile(knockoff_const_saliency_url):
                    data_const = np.load(knockoff_const_saliency_url, allow_pickle=True)
                    X_knockoff_saliency_const_upper = data_const['knockoff_upper'];
                    X_knockoff_saliency_const_lower = data_const['knockoff_lower'];
                else:
                    X_knockoff_saliency_const_upper = X_orig_saliency.copy(); X_knockoff_saliency_const_lower = X_orig_saliency.copy();

                    rowcol_tuple_list = []; multi_size = 100;
                    for word_idx in range(X_test.shape[1]):
                        for embed_idx in range(X_test.shape[2]):
                            rowcol_tuple_list.append((word_idx, embed_idx));
                    np.random.shuffle(rowcol_tuple_list);
                    total_batch = int(np.ceil(len(rowcol_tuple_list) * 1.0 / multi_size));
                    for curr_batch in range(total_batch):
                        print('curr_batch={}/{}'.format(curr_batch, total_batch));
                        curr_rowcol_tuples = rowcol_tuple_list[curr_batch * multi_size:(curr_batch + 1) * multi_size];

                        X_knockoff = X_test.copy();
                        for word_idx, embed_idx in curr_rowcol_tuples: X_knockoff[:, word_idx, embed_idx, 0] = 0; # word_embedding_mean
                        X_knockoff_saliency_const = saliency_gen.generate(X_knockoff, np.argmax(y_test, axis=1));

                        X_knockoff_saliency_const_upper = np.maximum(X_knockoff_saliency_const_upper, X_knockoff_saliency_const);
                        X_knockoff_saliency_const_lower = np.minimum(X_knockoff_saliency_const_lower, X_knockoff_saliency_const);
                    np.savez(knockoff_const_saliency_url, knockoff_upper=X_knockoff_saliency_const_upper, knockoff_lower=X_knockoff_saliency_const_lower);

                X_const_knockoff_saliency = X_knockoff_saliency_const_upper - X_knockoff_saliency_const_lower; assert np.min(X_const_knockoff_saliency) >= 0;
                X_const_knockoff_saliency = normalized_saliency(X_const_knockoff_saliency, use_abs=use_abs, normalize=normalize);
                print('X_const_knockoff_saliency={}\t'.format(X_const_knockoff_saliency.shape,  ));


                knockoff_noise_saliency_url = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'runs', 'knockoff_saliency_noise_{}.npz'.format(saliency_gen.name));
                if os.path.isfile(knockoff_noise_saliency_url):
                    data_noise = np.load(knockoff_noise_saliency_url, allow_pickle=True)
                    X_knockoff_saliency_noise_upper = data_noise['knockoff_upper'];
                    X_knockoff_saliency_noise_lower = data_noise['knockoff_lower'];
                else:
                    X_knockoff_saliency_noise_upper = X_orig_saliency.copy(); X_knockoff_saliency_noise_lower = X_orig_saliency.copy();

                    for word_idx in range(X_id.shape[1]):
                        print('word_idx={}'.format(word_idx));

                        X_knockoff = X_test.copy();
                        for sample_idx in range(X_id.shape[0]): X_knockoff[sample_idx, word_idx, :, 0] += 5*np.random.rand(embedding_size);
                        X_knockoff_saliency_noise = saliency_gen.generate(X_knockoff, np.argmax(y_test, axis=1));

                        X_knockoff_saliency_noise_upper = np.maximum(X_knockoff_saliency_noise_upper, X_knockoff_saliency_noise);
                        X_knockoff_saliency_noise_lower = np.minimum(X_knockoff_saliency_noise_lower, X_knockoff_saliency_noise);
                    np.savez(knockoff_noise_saliency_url, knockoff_upper=X_knockoff_saliency_noise_upper, knockoff_lower=X_knockoff_saliency_noise_lower);

                X_noise_knockoff_saliency = X_knockoff_saliency_noise_upper - X_knockoff_saliency_noise_lower; assert np.min(X_noise_knockoff_saliency) >= 0;
                X_noise_knockoff_saliency = normalized_saliency(X_noise_knockoff_saliency, use_abs=use_abs, normalize=normalize);
                print('X_noise_knockoff_saliency={}\t'.format(X_noise_knockoff_saliency.shape,  ));

                whole_word = True;
                ### fidelity
                fid1 = fidelity(X_test, np.argmax(y_test, axis=1), X_orig_saliency, whole_word=whole_word, K=saliency_portion);
                print('Origin fid={}'.format(len(fid1)))
                print('mean_fid={}\tmedian_fid={}'.format(np.mean(fid1), np.median(fid1)))
                print('==================')

                fid2 = fidelity(X_test, np.argmax(y_test, axis=1), X_decoy_knockoff_saliency, whole_word=whole_word, K=saliency_portion);
                print('Decoy_range fid={}'.format(len(fid2)))
                print('mean_fid={}\tmedian_fid={}'.format(np.mean(fid2), np.median(fid2)))
                print('==================')

                fid3 = fidelity(X_test, np.argmax(y_test, axis=1), X_decoymean_knockoff_saliency, whole_word=whole_word, K=saliency_portion);
                print('Decoy_mean fid={}'.format(len(fid3)))
                print('mean_fid={}\tmedian_fid={}'.format(np.mean(fid3), np.median(fid3)))
                print('==================')

                fid4 = fidelity(X_test, np.argmax(y_test, axis=1), X_const_knockoff_saliency, whole_word=whole_word, K=saliency_portion);
                print('ConstDecoy fid={}'.format(len(fid4)))
                print('mean_fid={}\tmedian_fid={}'.format(np.mean(fid4), np.median(fid4)))
                print('==================')

                fid5 = fidelity(X_test, np.argmax(y_test, axis=1), X_noise_knockoff_saliency, whole_word=whole_word, K=saliency_portion);
                print('NoiseDecoy fid={}'.format(len(fid5)))
                print('mean_fid={}\tmedian_fid={}'.format(np.mean(fid5), np.median(fid5)))
                print('==================')



                output_url = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'runs', 'sst_fidelity_{}_norm{}_whole{}_abs{}_K{}.csv'.format(saliency_gen.name, int(normalize),  int(whole_word), int(use_abs), saliency_portion )); print(output_url);
                f = open(output_url, 'w');
                f.write('saliency,raw_fidelity_original,raw_fidelity_decoy_range,raw_fidelity_decoy_mean,raw_fidelity_const,raw_fidelity_noise\n');
                for value1, value2, value3, value4, value5 in zip(fid1, fid2, fid3, fid4, fid5): f.write('{},{},{},{},{},{}\n'.format(saliency_gen.name, value1, value2, value3, value4, value5 ));
                f.close();


                # valid_sample_indices = [0,4,7,20,34,53,79,100,108,123,143,157,159,168,236,337,354];
                valid_sample_indices = list(range(200));
                for sample_idx in valid_sample_indices:
                    print('sample_idx={}\t{}'.format(sample_idx, X_orig[sample_idx] ));
                    save_url = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'runs', 'saliency','sst_sentence_{}_norm{}_whole{}_abs{}_K{}_sample{}.pdf'.format(saliency_gen.name, int(normalize),  int(whole_word), int(use_abs), saliency_portion, sample_idx ));

                    orig_normed_saliency_collapsed, orig_topk_saliency_collapsed = get_word_saliency(X_orig_saliency);
                    decoy_normed_saliency_collapsed, decoy_topk_saliency_collapsed = get_word_saliency(X_decoy_knockoff_saliency);


                    word_saliency_list_normed = [(id_to_vocab_map[id], saliency_orig, saliency_decoy) for id, saliency_orig, saliency_decoy in
                                          zip(X_id[sample_idx], orig_normed_saliency_collapsed[sample_idx], decoy_normed_saliency_collapsed[sample_idx]) if id > 0];

                    word_saliency_list_topk = [(id_to_vocab_map[id], saliency_orig, saliency_decoy) for id, saliency_orig, saliency_decoy in
                                          zip(X_id[sample_idx], orig_topk_saliency_collapsed[sample_idx], decoy_topk_saliency_collapsed[sample_idx]) if id > 0];

                    label = np.argmax(y_test[sample_idx]);
                    visualize_sentence_saliency(word_saliency_list_normed, (fid1[sample_idx], fid2[sample_idx]), label, save_url, annotat=False);
                    visualize_sentence_saliency(word_saliency_list_normed, (fid1[sample_idx], fid2[sample_idx]), label, save_url, annotat=True);
