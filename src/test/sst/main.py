import tensorflow as tf
from conf import config
from datasets import Dataset
from word_vectors import WordVectors
import train
import evaluate
import sys
import numpy as np

dataset = Dataset("SST_phrase")
# dataset = Dataset("IMDB", preprocess=True)
# dataset = Dataset("MR", preprocess=True)

# load pretrained word vectors
if not config['eval']:
    pretrained_vectors = []
    for index, type_ in enumerate(config['word_vector_type']):
        pretrained_vectors.append(WordVectors(type_, config["pretrained_vectors"][index]))
        print ("loaded vectors {}".format(config['word_vector_type'][index]))
        # print('dictionary={}'.format(len(pretrained_vectors[index].dictionary)));
        # print('word_to_index={}'.format(len(pretrained_vectors[index].word_to_index)));
        # print('vectors={}'.format(pretrained_vectors[index].vectors.shape));


# SST splits
# 1 = train
# 2 = test
# 3 = dev
t1 = dataset.cv_split(index=2)
t2 = dataset.cv_split(index=1)
data = [t2[2], t2[3], t1[2], t1[3]]


dx_train, y_train, dx_dev, y_dev = data
max_document_length = max([len(x.split(" ")) for x in dx_train])

from tensorflow.contrib import learn
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
vocab_processor.fit(dx_train + dx_dev)

vocab_dict = vocab_processor.vocabulary_._mapping
id_to_vocab_map = {};
for vocab in vocab_dict: id_to_vocab_map[vocab_dict[vocab]] = vocab;
print(id_to_vocab_map);
np.save('id_to_vocab_map.npy', id_to_vocab_map);



#
# # init and run tf graph
# g = tf.Graph()
# with g.as_default():
#     sess = tf.Session()
#     with sess.as_default():
#         if config['eval']:
#             evaluate.eval_model(
#                 sess, g, config["load_last_checkpoint"], data, config,
#             )
#         else:
#             train.set_train(
#                 sess, config, data,
#                 pretrained_embeddings=pretrained_vectors)
