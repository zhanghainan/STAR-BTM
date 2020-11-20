# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
from models.layers.rnn_encoder import RNNEncoder

x = tf.placeholder(tf.int32, [None, None])
length = tf.placeholder(tf.int32, [None])
embs = tf.get_variable("embeddings", [2000, 10])

emb_inp = tf.nn.embedding_lookup(embs, x)

encoder = RNNEncoder('lstm', 'uni', 10, num_layers=2)
outputs, state = encoder(emb_inp, length)
outputs2, state2 = encoder(emb_inp, length)

print(outputs, state)
# print(outputs.shape, state.shape)

bi_encoder = RNNEncoder('lstm', 'bi', 20, num_layers=2)
bi_outputs, bi_state = bi_encoder(emb_inp, length)
bi_outputs2, bi_state2 = bi_encoder(emb_inp, length)
# print(bi_outputs.shape, bi_state.shape)
print(bi_outputs, bi_state)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    np_x = np.asarray([[3, 4, 5, 0, 0], [4, 5, 1, 2, 0]])
    np_len = np.asarray([3, 4], dtype='int32')
    res = sess.run([outputs, state, outputs2, state2, bi_outputs, bi_state], {x: np_x, length: np_len})
    for r in res:
        print(r)
