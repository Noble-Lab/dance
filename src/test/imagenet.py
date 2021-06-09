from __future__ import absolute_import, division, print_function, unicode_literals


import os, sys, unittest, argparse;

parent_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'));

sys.path.append(parent_dir);
sys.path.append(os.path.join(parent_dir, 'models'));
sys.path.append(os.path.join(parent_dir, 'classifiers'));
sys.path.append(os.path.join(parent_dir, 'knockoffs'));
sys.path.append(os.path.join(parent_dir, 'viz'));
sys.path.append(os.path.join(parent_dir, 'results'));

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import numpy as np
import tensorflow as tf
slim = tf.contrib.slim

import models.model_zoo as model_zoo
from classifiers.tensorflow import TFClassifier

from saliencies.saliency_gradient import GradientSaliency;
from saliencies.saliency_smoothgrad import SmoothGradSaliency;
from saliencies.saliency_integratedgrad import IntegratedGradSaliency;

from knockoffs.knockoff_bruteforce import BruteforceKnockoff
from knockoffs.knockoff_transform_linf import TransformLInfMultiPatchKnockoff
from knockoffs.knockoff_baseline_const import ConstKnockoff
from knockoffs.knockoff_baseline_blur import BlurKnockoff

# import viz.visualization as visualization;

from sklearn.metrics import accuracy_score
# from matplotlib import pyplot as plt
from scipy import io;

NB_TRAIN = 60000
NB_TEST = 10000
BATCH_SIZE = 128
IMG_INDEX = [0,1,2,3]


class TestPixelKnockoff(unittest.TestCase):
    """
    This class tests the functionalities of pixel-level knockoffs
    """

    @classmethod
    def setUpClass(cls):
        pass

    def setUp(self):
        logger.info('setUp')
        model_zoo.master_seed(1234);

        # Loading data0
        parent_path = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'));
        # pparent_path = os.path.join(os.path.abspath(os.path.join(parent_path, '..')), 'results', 'attacks', 'imagenet');

        pparent_path = os.path.abspath(os.path.join(parent_path, 'data'));
        self.data = io.loadmat(os.path.join(pparent_path, 'imagenet_data.mat'));

        self.x = self.data['X']; logger.info('x={}'.format(self.x.shape));
        self.y = self.data['y'].flatten(); logger.info('y={}'.format(self.y.shape));
        # self.labels = self.data['label']; logger.info('labels={}'.format(self.labels.shape));

        self.mean_image = np.zeros(self.x[0].shape);
        self.mean_image[:, :, 0] = 103.939; self.mean_image[:, :, 1] = 116.779; self.mean_image[:, :, 2] = 123.68;


        input_ph = tf.placeholder(tf.float32, shape=[None, 227, 227, 3], name='input_x');
        output_ph = tf.placeholder(tf.float32, shape=[None, 1000], name='output_y');

        num_classes = 1000; is_training=False; dropout_keep_prob=0.5; spatial_squeeze=True; scope='vgg_16'; fc_conv_padding='VALID'; global_pool=False;

        with tf.variable_scope(scope, 'vgg_16', [input_ph]) as sc:
            end_points_collection = sc.original_name_scope + '_end_points'
            # Collect outputs for conv2d, fully_connected and max_pool2d.
            with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                                outputs_collections=end_points_collection):
                net = slim.repeat(input_ph, 2, slim.conv2d, 64, [3, 3], scope='conv1')
                net = slim.max_pool2d(net, [2, 2], scope='pool1')
                net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
                net = slim.max_pool2d(net, [2, 2], scope='pool2')
                net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
                net = slim.max_pool2d(net, [2, 2], scope='pool3')
                net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
                net = slim.max_pool2d(net, [2, 2], scope='pool4')
                conv_last = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
                net = slim.max_pool2d(conv_last, [2, 2], scope='pool5')

                # Use conv2d instead of fully_connected layers.
                net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
                net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                                   scope='dropout6')
                net = slim.conv2d(net, 4096, [1, 1], scope='fc7')

                # Convert end_points_collection into a end_point dict.
                end_points = slim.utils.convert_collection_to_dict(end_points_collection)
                if global_pool:
                    net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
                    end_points['global_pool'] = net
                if num_classes:
                    net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                                       scope='dropout7')
                    net = slim.conv2d(net, num_classes, [1, 1],
                                      activation_fn=None,
                                      normalizer_fn=None,
                                      scope='fc8')
                    if spatial_squeeze:
                        logits = tf.squeeze(net, [1, 2], name='fc8/squeezed')
                    end_points[sc.name + '/fc8'] = logits

        # Tensorflow session and initialization
        self.sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(self.sess, os.path.join(pparent_path, 'vgg_16.ckpt'));

        # Train operator
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=output_ph))
        train = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(loss)

        # Create classifier
        self.classifier = TFClassifier((-self.mean_image, 255.0 - self.mean_image), input_ph = input_ph, logits = logits,
                                       output_ph= output_ph, train=train, loss=loss, sess=self.sess)

        # preds = self.classifier.predict(self.x, batch_size=BATCH_SIZE);
        # logger.info('Accuracy after fitting: %.2f%%', (accuracy_score(self.y, np.argmax(preds, axis=1)) * 100))

        self.using_attack = True
        if int(os.environ['TOP']) > 0: sub_dir = 'topK{}'.format(os.environ['TOP']);
        elif int(os.environ['TOP']) == 0: sub_dir = 'mass_center';
        elif int(os.environ['TOP']) == -1: sub_dir = 'target';
        else: self.using_attack = False

        if self.using_attack == True:
            if os.environ['SALIENCY'].lower() == 'gradient':
                self.data_attack = io.loadmat(os.path.join(pparent_path, sub_dir, 'new/attack_results_tf_gradient.mat'));
            elif os.environ['SALIENCY'].lower() == 'smoothgrad':
                self.data_attack = io.loadmat(os.path.join(pparent_path, sub_dir, 'new/attack_results_tf_smoothgrad.mat'));
            elif os.environ['SALIENCY'].lower() == 'integratedgrad':
                self.data_attack = io.loadmat(os.path.join(pparent_path, sub_dir, 'new/ttack_results_tf_integratedgrad.mat'));
            else: assert False;


    def tearDown(self):
        logger.info('tearDown')
        self.sess.close();

    def test_knockoff(self):
        logger.info('test_knockoff');

        selected_img_indices = IMG_INDEX;
        subx = self.x[selected_img_indices]; logger.info('subx={}'.format(subx.shape));
        suby = self.y[selected_img_indices]; logger.info('suby={}'.format(suby.shape));
        if self.using_attack == True:
            print('Using attack data...')
            x_attack = self.data_attack['X_attack'][selected_img_indices]; logger.info('x_attack={}'.format(x_attack.shape));
        else:
            print('Using original data...')
            x_attack = subx

        assert os.environ['SALIENCY'].lower() in ['gradient', 'smoothgrad', 'integratedgrad'];
        if os.environ['SALIENCY'].lower() == 'gradient':
            saliency_gen = GradientSaliency(self.classifier, normalize=False);
        elif os.environ['SALIENCY'].lower() == 'smoothgrad':
            saliency_gen = SmoothGradSaliency(self.classifier, normalize=False);
        elif os.environ['SALIENCY'].lower() == 'integratedgrad':
            saliency_gen = IntegratedGradSaliency(self.classifier, normalize=False);
        else: assert False;
        logger.info('saliency_gen={}'.format(saliency_gen.name));

        def normalized_saliency(sal_x):
            sal_x = np.abs(sal_x);

            percent_x = np.percentile(sal_x, 99, axis=range(1, len(sal_x.shape)));
            for _ in range(len(sal_x.shape) - 1): percent_x = percent_x[:, np.newaxis];
            sal_x = np.minimum(sal_x, percent_x);

            sal_x_max = np.max(np.max(np.max(sal_x, axis=1), axis=1), axis=1);
            sal_x = sal_x / (sal_x_max[:, None, None, None, ] + np.finfo(np.float).eps);
            return sal_x;

        layer_names = self.classifier.layer_names;
        logger.info('layer_names={}'.format(layer_names));
        valid_layer_indices = [idx for idx, name in enumerate(layer_names) if 'MaxPool' in name];
        logger.info('query_layer={}\tvalid_layer_indices={}\tvalid_layer_names={}'.format(os.environ['LAYER'], valid_layer_indices, [layer_names[idx] for idx in valid_layer_indices]));

        knockoff_type = int(os.environ['KNOCKOFF_TYPE']);
        logger.info('knockoff_type={}\tsaliency={}\tlayer={}\teps={}\tC={}\ttol={}\tdelta={}\tmulti_size={}\tpatch_size={}\tstride={}'.format(
                os.environ['KNOCKOFF_TYPE'],
                os.environ['SALIENCY'],
                os.environ['LAYER'],
                os.environ['EPS'],
                os.environ['C'],
                os.environ['TOL'],
                os.environ['DELTA'],
                os.environ['MULTI_SIZE'],
                os.environ['PATCH_SIZE'],
                os.environ['STRIDE']));


        if knockoff_type == 1:
            knockoff_gen = BruteforceKnockoff(self.classifier, saliency_gen,
                                              knock_direction=int(os.environ['DIRECTION']),
                                              partition_idx=int(os.environ['PARTITIONIDX']),
                                              partition_sum=int(os.environ['PARTITIONSUM']),
                                              layer=int(os.environ['LAYER']),
                                              tol=float(os.environ['TOL']));
        elif knockoff_type == 3:
            knockoff_gen = TransformLInfMultiPatchKnockoff(self.classifier, saliency_gen,
                                                           knock_direction=int(os.environ['DIRECTION']),
                                                           partition_idx=int(os.environ['PARTITIONIDX']),
                                                           partition_sum=int(os.environ['PARTITIONSUM']),
                                                           layer=int(os.environ['LAYER']),
                                                           init_C=float(os.environ['C']),
                                                           tol=float(os.environ['TOL']),
                                                           patch_size=int(os.environ['PATCH_SIZE']),
                                                           stride=int(os.environ['STRIDE']),
                                                           multi_size=int(os.environ['MULTI_SIZE']),
                                                           delta=float(os.environ['DELTA']),
                                                           repeat=1);

        elif knockoff_type == 4:
            knockoff_gen = ConstKnockoff(self.classifier, saliency_gen,
                                                           knock_direction=int(os.environ['DIRECTION']),
                                                           partition_idx=int(os.environ['PARTITIONIDX']),
                                                           partition_sum=int(os.environ['PARTITIONSUM']),
                                                           layer=int(os.environ['LAYER']),
                                                           patch_size=int(os.environ['PATCH_SIZE']),
                                                           stride=int(os.environ['STRIDE']),
                                                           multi_size=int(os.environ['MULTI_SIZE']),
                                                           ref_values=[0,0,0] );

        elif knockoff_type == 5:
            knockoff_gen = ConstKnockoff(self.classifier, saliency_gen,
                                                           knock_direction=int(os.environ['DIRECTION']),
                                                           partition_idx=int(os.environ['PARTITIONIDX']),
                                                           partition_sum=int(os.environ['PARTITIONSUM']),
                                                           layer=int(os.environ['LAYER']),
                                                           patch_size=int(os.environ['PATCH_SIZE']),
                                                           stride=int(os.environ['STRIDE']),
                                                           multi_size=int(os.environ['MULTI_SIZE']),
                                                           ref_values=[-103.939,-116.779,-123.68] );

        elif knockoff_type == 6:
            knockoff_gen = BlurKnockoff(self.classifier, saliency_gen,
                                                           knock_direction=int(os.environ['DIRECTION']),
                                                           partition_idx=int(os.environ['PARTITIONIDX']),
                                                           partition_sum=int(os.environ['PARTITIONSUM']),
                                                           layer=int(os.environ['LAYER']),
                                                           patch_size=int(os.environ['PATCH_SIZE']),
                                                           stride=int(os.environ['STRIDE']),
                                                           multi_size=int(os.environ['MULTI_SIZE']),
                                                           blur_radius=1 );

        elif knockoff_type == 7:
            knockoff_gen = BlurKnockoff(self.classifier, saliency_gen,
                                                           knock_direction=int(os.environ['DIRECTION']),
                                                           partition_idx=int(os.environ['PARTITIONIDX']),
                                                           partition_sum=int(os.environ['PARTITIONSUM']),
                                                           layer=int(os.environ['LAYER']),
                                                           patch_size=int(os.environ['PATCH_SIZE']),
                                                           stride=int(os.environ['STRIDE']),
                                                           multi_size=int(os.environ['MULTI_SIZE']),
                                                           blur_radius=3 );

        elif knockoff_type == 8:
            knockoff_gen = BlurKnockoff(self.classifier, saliency_gen,
                                                           knock_direction=int(os.environ['DIRECTION']),
                                                           partition_idx=int(os.environ['PARTITIONIDX']),
                                                           partition_sum=int(os.environ['PARTITIONSUM']),
                                                           layer=int(os.environ['LAYER']),
                                                           patch_size=int(os.environ['PATCH_SIZE']),
                                                           stride=int(os.environ['STRIDE']),
                                                           multi_size=int(os.environ['MULTI_SIZE']),
                                                           blur_radius=5 );


        else: assert False;

        # visualization.save_image(visualization.create_sprite(subx+self.mean_image), 'imagenet_x.png', cmap='');
        # visualization.save_image(visualization.create_sprite(x_attack+self.mean_image), 'imagenet_x_attackTop{}_{}.png'.format(os.environ['TOP'], os.environ['SALIENCY']), cmap='');
        # visualization.save_image(visualization.create_sprite(normalized_saliency(saliency_gen.generate(subx, suby)), normalize=True), 'imagenet_saliency_{}.png'.format(os.environ['SALIENCY']));
        # visualization.save_image(visualization.create_sprite(normalized_saliency(saliency_gen.generate(x_attack, suby))), 'imagenet_saliency_attackTop{}_{}.png'.format(os.environ['TOP'], os.environ['SALIENCY']));

        if self.using_attack:
            saved_saliency_url = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')), 'data', 'imagenet_attackTop{}_knockoff_{}_{}.npy'.format(os.environ['TOP'], saliency_gen.name, knockoff_gen.get_params_str()));
        else:
            saved_saliency_url = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')), 'data', 'imagenet_knockoff_{}_{}.npy'.format(saliency_gen.name, knockoff_gen.get_params_str()));

        logger.info(saved_saliency_url)
        if os.path.isfile(saved_saliency_url):
            knockoff_x, saliency_x = np.load(saved_saliency_url);
        else:
            knockoff_x, saliency_x = knockoff_gen.generate(x_attack, suby);
            np.save(saved_saliency_url, (knockoff_x, saliency_x));
        logger.info('finished saliency {}'.format(saliency_gen.name));
        #
        # # label = 'imagenet'; x = subx; y = suby;
        # label = 'imagenet'; x = x_attack; y = suby;
        #
        #
        # saliency_url = '{}_attackTop{}_saliency_0_{}_{}.png'.format(label, os.environ['TOP'], saliency_gen.name, knockoff_gen.get_params_str());
        # visualization.save_image(visualization.create_sprite(normalized_saliency(saliency_x), normalize=True), saliency_url);
        #
        # saliency_url = '{}_attackTop{}_saliency_1_{}_{}.png'.format(label, os.environ['TOP'], saliency_gen.name, knockoff_gen.get_params_str());
        # visualization.save_image(visualization.create_sprite(normalized_saliency(saliency_x_upper - saliency_x_lower) , normalize=True), saliency_url);
        #
        # saliency_url = '{}_attackTop{}_saliency_2_{}_{}.png'.format(label, os.environ['TOP'], saliency_gen.name, knockoff_gen.get_params_str());
        # visualization.save_image(visualization.create_sprite(normalized_saliency(saliency_x_upper + saliency_x_lower) , normalize=True), saliency_url);
        #
        # saliency_url = '{}_attackTop{}_saliency_3_{}_{}.png'.format(label, os.environ['TOP'], saliency_gen.name, knockoff_gen.get_params_str());
        # visualization.save_image(visualization.create_sprite(normalized_saliency(saliency_x_upper - saliency_x_mean) , normalize=True), saliency_url);
        #
        # saliency_url = '{}_attackTop{}_saliency_4_{}_{}.png'.format(label, os.environ['TOP'], saliency_gen.name, knockoff_gen.get_params_str());
        # visualization.save_image(visualization.create_sprite(normalized_saliency(saliency_x_mean - saliency_x_lower) , normalize=True), saliency_url);
        #
        # saliency_url = '{}_attackTop{}_saliency_5_{}_{}.png'.format(label, os.environ['TOP'], saliency_gen.name, knockoff_gen.get_params_str());
        # visualization.save_image(visualization.create_sprite(normalized_saliency(saliency_x_mean), normalize=True), saliency_url);
        #
        # saliency_url = '{}_attackTop{}_saliency_6_{}_{}.png'.format(label, os.environ['TOP'], saliency_gen.name, knockoff_gen.get_params_str());
        # visualization.save_image(visualization.create_sprite(normalized_saliency(saliency_x_upper), normalize=True), saliency_url);
        #
        # saliency_url = '{}_attackTop{}_saliency_7_{}_{}.png'.format(label, os.environ['TOP'], saliency_gen.name, knockoff_gen.get_params_str());
        # visualization.save_image(visualization.create_sprite(normalized_saliency(saliency_x_lower), normalize=True), saliency_url);

        tf.reset_default_graph()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optional app description');
    parser.add_argument('--knockoff_type', type=int, help='knockoff_type');
    parser.add_argument('--saliency', type=str, help='saliency');
    parser.add_argument('--layer', type=int, help='layer', default=0);
    parser.add_argument('--eps', type=float, help='eps', default=2e3);
    parser.add_argument('--C', type=float, help='C', default=5e-7);
    parser.add_argument('--tol', type=float, help='tol', default=1e-3);
    parser.add_argument('--multi_size', type=int, help='multi_size', default=10);
    parser.add_argument('--patch_size', type=int, help='patch_size', default=2);
    parser.add_argument('--stride', type=int, help='stride', default=1);
    parser.add_argument('--top', type=int, help='top');
    parser.add_argument('--delta', type=float, help='delta');
    parser.add_argument('--direction', type=int, help='direction');
    parser.add_argument('--partitionidx', type=int, help='partitionidx');
    parser.add_argument('--partitionsum', type=int, help='partitionsum');

    args = parser.parse_args();

    os.environ['KNOCKOFF_TYPE'] = str(args.knockoff_type);
    os.environ['SALIENCY'] = str(args.saliency);
    os.environ['LAYER'] = str(args.layer);
    os.environ['EPS'] = str(args.eps);
    os.environ['C'] = str(args.C);
    os.environ['TOL'] = str(args.tol);
    os.environ['MULTI_SIZE'] = str(args.multi_size);
    os.environ['PATCH_SIZE'] = str(args.patch_size);
    os.environ['STRIDE'] = str(args.stride);
    os.environ['TOP'] = str(args.top);
    os.environ['DELTA'] = str(args.delta);
    os.environ['DIRECTION'] = str(args.direction);
    os.environ['PARTITIONIDX'] = str(args.partitionidx);
    os.environ['PARTITIONSUM'] = str(args.partitionsum);

    unittest.main(argv=[sys.argv[1]]);


### Normalization
# def normalized_saliency_by_topK(sal_x, K=0.2):
#     sal_x_tmp = np.copy(sal_x)
#     thred = int(227 * 227 * K)
#     for idx in range(sal_x.shape[0]):
#         for i in range(3):
#             tau = np.sort(sal_x[idx, :, :, i].flatten())[-thred]
#             sal_x_tmp[idx, :, :, i][sal_x[idx, :, :, i] >= tau] = 1
#             sal_x_tmp[idx, :, :, i][sal_x[idx, :, :, i] < tau] = 0

#     return sal_x_tmp


### Fidelity evaluation.
## x_tmp: input sample;
## y_tmp: label of x_tmp;
## classifier: pretrained neural network -- TFClassifier;
## k_sal: decoy saliency map of x_tmp.

# k_sal = normalized_saliency(knockoff_saliencies)[0,]
# x_eval_k = np.expand_dims(x_tmp * k_sal, 0)

# pred_eval = classifier.predict(x_eval_k)[0, y_tmp]

# print('Fidelity score:')
# print(-np.log(pred_eval))


### Robustness evaluation.
## x_attack: attack image of x_tmp;
## x_k_sal: decoy saliency map of x_tmp;
## x_attack_k_sal: decoy saliency map of x_attack.

# x_k_sal = normalized_saliency(knockoff_saliencies)[0,]
# x_attack = attack_sample[i, idx]
# x_attack_k_sal = normalized_saliency(attack_knockoff_saliency)[0,]
# diff_x = np.sqrt(np.sum(np.square((x_tmp - x_attack)/255)))
# diff_x_k_sal = np.sqrt(np.sum(np.square((x_k_sal - x_attack_k_sal))))
# stab_x_k = diff_x_k_sal/diff_x
# print('Adversarial robustness score:')
# print(stab_x_k)

