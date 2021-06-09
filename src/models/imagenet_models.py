from __future__ import absolute_import, division, print_function, unicode_literals

import os
import tensorflow as tf
from tensorflow.contrib.slim.nets import resnet_v1
from tensorflow.contrib.slim.nets import inception
import numpy as np

slim = tf.contrib.slim
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)

def imagenet_vgg16(save_path,
                   model_name,
                   num_classes = 1000,
                   is_training=False,
                   dropout_keep_prob=0.5,
                   spatial_squeeze=True,
                   scope='vgg_16',
                   fc_conv_padding='VALID',
                   global_pool=False):
    """Oxford Net VGG 16-Layers version D.
    :param save_path: checkpoint save path.
    :type save_path: 'str'.
    :param model_name: name of the checkpoint.
    :type model_name: 'str'.
    :param num_classes: number of classes.
    :type num_classes: 'int'.
    :param is_training: Training model or not.
    :type is_training: 'bool'
    :param dropout_keep_prob: drop out probability.
    :type dropout_keep_prob: 'float'
    :param spatial_squeeze: whether or not should squeeze the spatial dimensions of the
           outputs. Useful to remove unnecessary dimensions for classification.
    :type spatial_squeeze: 'bool'
    :param scope: Optional scope for the variables.
    :type scope: 'str;
    :param fc_conv_padding: the type of padding to use for the fully connected layer
           that is implemented as a convolutional layer. Use 'SAME' padding if you
           are applying the network in a fully convolutional manner and want to
           get a prediction map downsampled by a factor of 32 as an output.
           Otherwise, the output prediction map will be (input / 32) - 6 in case of 'VALID' padding.
    :type fc_conv_padding: 'str'
    :param global_pool: Optional boolean flag. If True, the input to the classification
           layer is avgpooled to size 1x1, for any input size. (This is not part
           of the original VGG architecture.)
    :type 'bool'
    :return input_ph: input placeholder.
    :type input_ph: 'tf.placeholder'
    :return logits: pre-softmax layer output
    :type logits: 'tf.tensor'
    :return output_ph: output placehoder.
    :type output_ph: 'tf.placeholder'
    :return train: training optimizer.
    :type train: tf.train.optimizer
    :return loss: model training loss
    :type loss: 'tf.tensor'
    :return sess: tensorflow session.
    :type sess: 'tf.session'
    """

    tf.reset_default_graph()
    input_ph = tf.placeholder(tf.float32, shape=[None, 227, 227, 3], name='input_x')
    output_ph = tf.placeholder(tf.float32, shape=[None, 1000], name='output_y')

    with tf.variable_scope(scope, 'vgg_16', [input_ph]) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                            outputs_collections=end_points_collection):
            net_1 = slim.repeat(input_ph, 2, slim.conv2d, 64, [3, 3], scope='conv1')
            net = slim.max_pool2d(net_1, [2, 2], scope='pool1')
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
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, os.path.join(save_path, model_name))

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=output_ph))
    train = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(loss)
    return input_ph, logits, output_ph, train, loss, sess

def imagenet_resnet_v1_50(save_path, model_name, num_classes = 1000, is_training = False):
    """resnet_v1 architecture
    :param save_path: checkpoint save path.
    :type save_path: 'str'.
    :param model_name: name of the checkpoint.
    :type model_name: 'str'.
    :param num_classes: number of classes.
    :type num_classes: 'int'.
    :param is_training: Training model or not.
    :type is_training: 'bool'
    :return input_ph: input placeholder.
    :type input_ph: 'tf.placeholder'
    :return logits: pre-softmax layer output
    :type logits: 'tf.tensor'
    :return output_ph: output placehoder.
    :type output_ph: 'tf.placeholder'
    :return train: training optimizer.
    :type train: tf.train.optimizer
    :return loss: model training loss
    :type loss: 'tf.tensor'
    :return sess: tensorflow session.
    :type sess: 'tf.session'
    """
    tf.reset_default_graph()
    input_ph = tf.placeholder(tf.float32, shape=[None, 227, 227, 3])
    output_ph = tf.placeholder(tf.float32, shape=[None, num_classes], name='output_y')

    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        logits, end_points = resnet_v1.resnet_v1_50(input_ph, num_classes=num_classes, is_training=is_training)

    # Run computation
    saver = tf.train.Saver()
    # Execute graph
    sess = tf.Session()
    saver.restore(sess, os.path.join(save_path, model_name))

    logits = tf.layers.Flatten()(logits)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=output_ph))
    train = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(loss)
    return input_ph, logits, output_ph, train, loss, sess

def imagenet_alexnet(save_path,
                     model_name,
                     num_classes=1000,
                     is_training = False):
    """alexnet architecture
    :param save_path: checkpoint save path.
    :type save_path: 'str'.
    :param model_name: name of the checkpoint.
    :type model_name: 'str'.
    :param num_classes: number of classes.
    :type num_classes: 'int'.
    :param is_training: Training model or not.
    :type is_training: 'bool'
    :return input_ph: input placeholder.
    :type input_ph: 'tf.placeholder'
    :return logits: pre-softmax layer output
    :type logits: 'tf.tensor'
    :return output_ph: output placehoder.
    :type output_ph: 'tf.placeholder'
    :return train: training optimizer.
    :type train: tf.train.optimizer
    :return loss: model training loss
    :type loss: 'tf.tensor'
    :return sess: tensorflow session.
    :type sess: 'tf.session'
    """

    tf.reset_default_graph()
    input_ph = tf.placeholder(tf.float32, shape=[None, 227, 227, 3], name='input_x')
    output_ph = tf.placeholder(tf.float32, shape=[None, num_classes], name='output_y')

    # 1st Layer: Conv (w ReLu) -> Lrn -> Pool
    conv1 = conv(input_ph, 11, 11, 96, 4, 4, padding='VALID', name='conv1')
    norm1 = lrn(conv1, 2, 1e-05, 0.75, name='norm1')
    pool1 = max_pool(norm1, 3, 3, 2, 2, padding='VALID', name='pool1')

    # 2nd Layer: Conv (w ReLu)  -> Lrn -> Pool with 2 groups
    conv2 = conv(pool1, 5, 5, 256, 1, 1, groups=2, name='conv2')
    norm2 = lrn(conv2, 2, 1e-05, 0.75, name='norm2')
    pool2 = max_pool(norm2, 3, 3, 2, 2, padding='VALID', name='pool2')

    # 3rd Layer: Conv (w ReLu)
    conv3 = conv(pool2, 3, 3, 384, 1, 1, name='conv3')

    # 4th Layer: Conv (w ReLu) splitted into two groups
    conv4 = conv(conv3, 3, 3, 384, 1, 1, groups=2, name='conv4')

    # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
    conv5 = conv(conv4, 3, 3, 256, 1, 1, groups=2, name='conv5')
    pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID', name='pool5')

    # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
    flattened = tf.reshape(pool5, [-1, 6 * 6 * 256])
    fc6 = fc(flattened, 6 * 6 * 256, 4096, name='fc6')
    dropout6 = dropout(fc6, 0.5, is_training)

    # 7th Layer: FC (w ReLu) -> Dropout
    fc7 = fc(dropout6, 4096, 4096, name='fc7')
    dropout7 = dropout(fc7, 0.5, is_training)

    # 8th Layer: FC and return unscaled activations
    logits = fc(dropout7, 4096, num_classes, relu=False, name='fc8')

    # Load the weights into memory
    weights_dict = np.load(os.path.join(save_path, model_name), encoding='bytes').item()

    # list of all assignment operators
    assign_list = []

    # Loop over all layer names stored in the weights dict
    for op_name in weights_dict:
        with tf.variable_scope(op_name, reuse=True):
            # Assign weights/biases to their corresponding tf variable
            for data in weights_dict[op_name]:
                # Biases
                if len(data.shape) == 1:
                   var = tf.get_variable('biases', trainable=False)
                   assign_list.append(var.assign(data))

                # Weights
                else:
                   var = tf.get_variable('weights', trainable=False)
                   assign_list.append(var.assign(data))

    ret = tf.group(assign_list, name="load_weights")
    sess = tf.Session()
    sess.run(ret)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=output_ph))
    train = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(loss)
    return input_ph, logits, output_ph, train, loss, sess


def imagenet_inception_v1(save_path,
                          model_name,
                          num_classes=1001,
                          is_training=False):
    """Inception V1 architecture (http://arxiv.org/pdf/1409.4842v1.pdf).
    :param save_path: checkpoint save path.
    :type save_path: 'str'.
    :param model_name: name of the checkpoint.
    :type model_name: 'str'.
    :param num_classes: number of classes.
    :type num_classes: 'int'.
    :param is_training: Training model or not.
    :type is_training: 'bool'
    :param dropout_keep_prob: drop out probability.
    :type dropout_keep_prob: 'float'
    :param prediction_fn: a function to get predictions out of logits.
    :type prediction_fn: tf.Activation
    :param spatial_squeeze: whether or not should squeeze the spatial dimensions of the
           outputs. Useful to remove unnecessary dimensions for classification.
    :type spatial_squeeze: 'bool'
    :param reuse: whether or not the network and its variables should be reused. To be
          able to reuse 'scope' must be given.
    :type reuse: 'bool'
    :param scope: Optional scope for the variables.
    :type scope: 'str;
    :param global_pool: Optional boolean flag. If True, the input to the classification
           layer is avgpooled to size 1x1, for any input size.
    :type 'bool'
    :return input_ph: input placeholder.
    :type input_ph: 'tf.placeholder'
    :return logits: pre-softmax layer output
    :type logits: 'tf.tensor'
    :return output_ph: output placehoder.
    :type output_ph: 'tf.placeholder'
    :return train: training optimizer.
    :type train: tf.train.optimizer
    :return loss: model training loss
    :type loss: 'tf.tensor'
    :return sess: tensorflow session.
    :type sess: 'tf.session'
    """
    tf.reset_default_graph()
    input_ph = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name='input_x')
    output_ph = tf.placeholder(tf.float32, shape=[None, num_classes], name='output_y')

    with slim.arg_scope(inception.inception_v1_arg_scope()):
        logits, end_points = inception.inception_v1(input_ph, num_classes=num_classes, is_training=False)
    predictions = end_points["Predictions"]
    saver = tf.train.Saver()

    # Execute graph
    sess = tf.Session()
    saver.restore(sess, os.path.join(save_path, model_name))

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=output_ph))
    train = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(loss)
    return input_ph, logits, output_ph, train, loss, sess



    # with tf.variable_scope(scope, 'InceptionV1', [input_ph], reuse=reuse) as scope:
    #     with slim.arg_scope([slim.batch_norm, slim.dropout],is_training=is_training):
    #         with tf.variable_scope('InceptionV1', 'InceptionV1', [input_ph]):
    #             with slim.arg_scope([slim.conv2d, slim.fully_connected],weights_initializer=trunc_normal(0.01)):
    #                 with slim.arg_scope([slim.conv2d, slim.max_pool2d],stride=1, padding='SAME'):
    #                     net = input_ph
    #                     end_point = 'Conv2d_1a_7x7'
    #                     net = slim.conv2d(input_ph, 64, [7, 7], stride=2, scope=end_point)
    #                     end_point = 'MaxPool_2a_3x3'
    #                     net = slim.max_pool2d(net, [3, 3], stride=2, scope=end_point)
    #                     end_point = 'Conv2d_2b_1x1'
    #                     net = slim.conv2d(net, 64, [1, 1], scope=end_point)
    #                     end_point = 'Conv2d_2c_3x3'
    #                     net = slim.conv2d(net, 192, [3, 3], scope=end_point)
    #                     end_point = 'MaxPool_3a_3x3'
    #                     net = slim.max_pool2d(net, [3, 3], stride=2, scope=end_point)
    #
    #                     end_point = 'Mixed_3b'
    #                     with tf.variable_scope(end_point):
    #                         with tf.variable_scope('Branch_0'):
    #                             branch_0 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
    #                         with tf.variable_scope('Branch_1'):
    #                             branch_1 = slim.conv2d(net, 96, [1, 1], scope='Conv2d_0a_1x1')
    #                             branch_1 = slim.conv2d(branch_1, 128, [3, 3], scope='Conv2d_0b_3x3')
    #                         with tf.variable_scope('Branch_2'):
    #                             branch_2 = slim.conv2d(net, 16, [1, 1], scope='Conv2d_0a_1x1')
    #                             branch_2 = slim.conv2d(branch_2, 32, [3, 3], scope='Conv2d_0b_3x3')
    #                         with tf.variable_scope('Branch_3'):
    #                             branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
    #                             branch_3 = slim.conv2d(branch_3, 32, [1, 1], scope='Conv2d_0b_1x1')
    #                         net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
    #
    #                     end_point = 'Mixed_3c'
    #                     with tf.variable_scope(end_point):
    #                         with tf.variable_scope('Branch_0'):
    #                             branch_0 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
    #                         with tf.variable_scope('Branch_1'):
    #                             branch_1 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
    #                             branch_1 = slim.conv2d(branch_1, 192, [3, 3], scope='Conv2d_0b_3x3')
    #                         with tf.variable_scope('Branch_2'):
    #                             branch_2 = slim.conv2d(net, 32, [1, 1], scope='Conv2d_0a_1x1')
    #                             branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0b_3x3')
    #                         with tf.variable_scope('Branch_3'):
    #                             branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
    #                             branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
    #                         net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
    #
    #                     end_point = 'MaxPool_4a_3x3'
    #                     net = slim.max_pool2d(net, [3, 3], stride=2, scope=end_point)
    #
    #                     end_point = 'Mixed_4b'
    #                     with tf.variable_scope(end_point):
    #                         with tf.variable_scope('Branch_0'):
    #                             branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
    #                         with tf.variable_scope('Branch_1'):
    #                             branch_1 = slim.conv2d(net, 96, [1, 1], scope='Conv2d_0a_1x1')
    #                             branch_1 = slim.conv2d(branch_1, 208, [3, 3], scope='Conv2d_0b_3x3')
    #                         with tf.variable_scope('Branch_2'):
    #                             branch_2 = slim.conv2d(net, 16, [1, 1], scope='Conv2d_0a_1x1')
    #                             branch_2 = slim.conv2d(branch_2, 48, [3, 3], scope='Conv2d_0b_3x3')
    #                         with tf.variable_scope('Branch_3'):
    #                             branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
    #                             branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
    #                         net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
    #
    #                     end_point = 'Mixed_4c'
    #                     with tf.variable_scope(end_point):
    #                         with tf.variable_scope('Branch_0'):
    #                             branch_0 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
    #                         with tf.variable_scope('Branch_1'):
    #                             branch_1 = slim.conv2d(net, 112, [1, 1], scope='Conv2d_0a_1x1')
    #                             branch_1 = slim.conv2d(branch_1, 224, [3, 3], scope='Conv2d_0b_3x3')
    #                         with tf.variable_scope('Branch_2'):
    #                             branch_2 = slim.conv2d(net, 24, [1, 1], scope='Conv2d_0a_1x1')
    #                             branch_2 = slim.conv2d(branch_2, 64, [3, 3], scope='Conv2d_0b_3x3')
    #                         with tf.variable_scope('Branch_3'):
    #                             branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
    #                             branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
    #                         net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
    #
    #                     end_point = 'Mixed_4d'
    #                     with tf.variable_scope(end_point):
    #                         with tf.variable_scope('Branch_0'):
    #                             branch_0 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
    #                         with tf.variable_scope('Branch_1'):
    #                             branch_1 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
    #                             branch_1 = slim.conv2d(branch_1, 256, [3, 3], scope='Conv2d_0b_3x3')
    #                         with tf.variable_scope('Branch_2'):
    #                             branch_2 = slim.conv2d(net, 24, [1, 1], scope='Conv2d_0a_1x1')
    #                             branch_2 = slim.conv2d(branch_2, 64, [3, 3], scope='Conv2d_0b_3x3')
    #                         with tf.variable_scope('Branch_3'):
    #                             branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
    #                             branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
    #                         net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
    #
    #                     end_point = 'Mixed_4e'
    #                     with tf.variable_scope(end_point):
    #                         with tf.variable_scope('Branch_0'):
    #                             branch_0 = slim.conv2d(net, 112, [1, 1], scope='Conv2d_0a_1x1')
    #                         with tf.variable_scope('Branch_1'):
    #                             branch_1 = slim.conv2d(net, 144, [1, 1], scope='Conv2d_0a_1x1')
    #                             branch_1 = slim.conv2d(branch_1, 288, [3, 3], scope='Conv2d_0b_3x3')
    #                         with tf.variable_scope('Branch_2'):
    #                             branch_2 = slim.conv2d(net, 32, [1, 1], scope='Conv2d_0a_1x1')
    #                             branch_2 = slim.conv2d(branch_2, 64, [3, 3], scope='Conv2d_0b_3x3')
    #                         with tf.variable_scope('Branch_3'):
    #                             branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
    #                             branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
    #                         net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
    #
    #                     end_point = 'Mixed_4f'
    #                     with tf.variable_scope(end_point):
    #                         with tf.variable_scope('Branch_0'):
    #                             branch_0 = slim.conv2d(net, 256, [1, 1], scope='Conv2d_0a_1x1')
    #                         with tf.variable_scope('Branch_1'):
    #                             branch_1 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
    #                             branch_1 = slim.conv2d(branch_1, 320, [3, 3], scope='Conv2d_0b_3x3')
    #                         with tf.variable_scope('Branch_2'):
    #                             branch_2 = slim.conv2d(net, 32, [1, 1], scope='Conv2d_0a_1x1')
    #                             branch_2 = slim.conv2d(branch_2, 128, [3, 3], scope='Conv2d_0b_3x3')
    #                         with tf.variable_scope('Branch_3'):
    #                             branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
    #                             branch_3 = slim.conv2d(branch_3, 128, [1, 1], scope='Conv2d_0b_1x1')
    #                         net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
    #
    #                     end_point = 'MaxPool_5a_2x2'
    #                     net = slim.max_pool2d(net, [2, 2], stride=2, scope=end_point)
    #
    #                     end_point = 'Mixed_5b'
    #                     with tf.variable_scope(end_point):
    #                         with tf.variable_scope('Branch_0'):
    #                             branch_0 = slim.conv2d(net, 256, [1, 1], scope='Conv2d_0a_1x1')
    #                         with tf.variable_scope('Branch_1'):
    #                             branch_1 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
    #                             branch_1 = slim.conv2d(branch_1, 320, [3, 3], scope='Conv2d_0b_3x3')
    #                         with tf.variable_scope('Branch_2'):
    #                             branch_2 = slim.conv2d(net, 32, [1, 1], scope='Conv2d_0a_1x1')
    #                             branch_2 = slim.conv2d(branch_2, 128, [3, 3], scope='Conv2d_0a_3x3')
    #                         with tf.variable_scope('Branch_3'):
    #                             branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
    #                             branch_3 = slim.conv2d(branch_3, 128, [1, 1], scope='Conv2d_0b_1x1')
    #                         net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
    #
    #                     end_point = 'Mixed_5c'
    #                     with tf.variable_scope(end_point):
    #                         with tf.variable_scope('Branch_0'):
    #                             branch_0 = slim.conv2d(net, 384, [1, 1], scope='Conv2d_0a_1x1')
    #                         with tf.variable_scope('Branch_1'):
    #                             branch_1 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
    #                             branch_1 = slim.conv2d(branch_1, 384, [3, 3], scope='Conv2d_0b_3x3')
    #                         with tf.variable_scope('Branch_2'):
    #                             branch_2 = slim.conv2d(net, 48, [1, 1], scope='Conv2d_0a_1x1')
    #                             branch_2 = slim.conv2d(branch_2, 128, [3, 3], scope='Conv2d_0b_3x3')
    #                         with tf.variable_scope('Branch_3'):
    #                             branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
    #                             branch_3 = slim.conv2d(branch_3, 128, [1, 1], scope='Conv2d_0b_1x1')
    #                         net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
    #         with tf.variable_scope('Logits'):
    #             if global_pool:
    #                 # Global average pooling.
    #                 net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
    #             else:
    #                 # Pooling with a fixed kernel size.
    #                 net = slim.avg_pool2d(net, [7, 7], stride=1, scope='AvgPool_0a_7x7')
    #             net = slim.dropout(net, dropout_keep_prob, scope='Dropout_0b')
    #             logits = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='Conv2d_0c_1x1')
    #
    #             if spatial_squeeze:
    #               logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
    #
    #               predictions = prediction_fn(logits, scope='Predictions')
    #
    # sess = tf.Session()
    # saver = tf.train.Saver()
    # saver.restore(sess, os.path.join(save_path, model_name))

def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name,
         padding='SAME', groups=1):
    """Create a convolution layer.

    Adapted from: https://github.com/ethereon/caffe-tensorflow
    """
    # Get number of input channels
    input_channels = int(x.get_shape()[-1])

    # Create lambda function for the convolution
    convolve = lambda i, k: tf.nn.conv2d(i, k,
                                         strides=[1, stride_y, stride_x, 1],
                                         padding=padding)

    with tf.variable_scope(name) as scope:
        # Create tf variables for the weights and biases of the conv layer
        weights = tf.get_variable('weights', shape=[filter_height,
                                                    filter_width,
                                                    input_channels/groups,
                                                    num_filters])
        biases = tf.get_variable('biases', shape=[num_filters])

    if groups == 1:
        conv = convolve(x, weights)

    # In the cases of multiple groups, split inputs & weights and
    else:
        # Split input and weights and convolve them separately
        input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
        weight_groups = tf.split(axis=3, num_or_size_splits=groups,
                                 value=weights)
        output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]

        # Concat the convolved output together again
        conv = tf.concat(axis=3, values=output_groups)

    # Add biases
    bias = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))

    # Apply relu function
    relu = tf.nn.relu(bias, name=scope.name)

    return relu


def fc(x, num_in, num_out, name, relu=True):
    """Create a fully connected layer."""
    with tf.variable_scope(name) as scope:

        # Create tf variables for the weights and biases
        weights = tf.get_variable('weights', shape=[num_in, num_out],
                                  trainable=True)
        biases = tf.get_variable('biases', [num_out], trainable=True)

        # Matrix multiply weights and inputs and add bias
        act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

    if relu:
        # Apply ReLu non linearity
        relu = tf.nn.relu(act)
        return relu
    else:
        return act


def max_pool(x, filter_height, filter_width, stride_y, stride_x, name,
             padding='SAME'):
    """Create a max pooling layer."""
    return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                          strides=[1, stride_y, stride_x, 1],
                          padding=padding, name=name)


def lrn(x, radius, alpha, beta, name, bias=1.0):
    """Create a local response normalization layer."""
    return tf.nn.local_response_normalization(x, depth_radius=radius,
                                              alpha=alpha, beta=beta,
                                              bias=bias, name=name)


def dropout(x, keep_prob, is_training):
    """Create a dropout layer."""
    return slim.dropout(x, keep_prob, is_training=is_training)
