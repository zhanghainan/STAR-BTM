# -*- coding:utf-8 -*-
import tensorflow as tf
from models.model_base import ModelMode
from models.HRED import HREDModel
from configs import HREDConfig
from utils.vocab import load_vocabulary

config = HREDConfig()
vocab = load_vocabulary('../data/ubuntu/vocab.dialog.txt')

config.vocab_size = vocab.size
config.sos_idx = vocab.sos_idx
config.eos_idx = vocab.eos_idx

config_proto = tf.ConfigProto()
with tf.Session(config=config_proto) as sess:
    initializer = tf.random_uniform_initializer(-1.0 * config.init_w, config.init_w)
    scope = config.model
    with tf.variable_scope(scope, reuse=None, initializer=initializer):
        train_model = HREDModel(config=config, mode=ModelMode.train, scope=scope)