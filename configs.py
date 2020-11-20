# - * - coding:utf-8 -*-


class BaseConfig(object):
    model = "BaseModel"

    # vocab size
    vocab_size = None
    sos_idx = None
    eos_idx = None

    # training options
    max_epoch = 200  # max training epochs
    batch_size = 32  # training batch size
    max_len = 50  # sentence max length
    infer_batch_size = 32  # infer batch size
    display_frequency = 200  # display_frequency
    checkpoint_frequency = 2000  # checkpoint frequency

    # optimizer configs
    optimizer = "adam"  # adam or sgd
    max_gradient_norm = 5.0  # gradient abs max cut
    learning_rate = 0.001  # initial learning rate
    start_decay_step = 10000
    decay_steps = 5000  # How frequent we decay
    decay_factor = 0.9  # How much we decay.

    # checkpoint max to keep
    max_to_keep = 20


class HREDConfig(BaseConfig):
    model = "HRED"

    prefix = "dialog"

    # model configs
    unit_type = "gru"  # gru or lstm
    enc_type = 'bi' # uni, bi
    emb_size = 256  # word embedding size
    topic_size = 10
    enc_hidden_size = 512  # encoder hidden size
    dec_hidden_size = 512  # decoder hidden size
    num_layers = 1  # number of RNN layers
    dropout_keep_prob = 0.9  # Dropout keep_prob rate (not drop_prob)
    init_w = 0.1  # init weight scale

    max_turn = 15
    batch_size = 32

    # infer options
    beam_size = 5
    infer_batch_size = 32
    infer_max_len = 50
    length_penalty_weight = 0.0

    buckets = [(2, 128), (4, 128), (6, 100), (8, 80), (10, 80)]  # buckets config(turn_size, batch_size)


class HREDTestConfig(HREDConfig):
    model = "HRED"

    prefix = "dialog"
    # model configs
    unit_type = "lstm"  # gru or lstm
    emb_size = 256  # word embedding size
    topic_size=10
    enc_hidden_size = 512  # encoder hidden size
    dec_hidden_size = 512  # decoder hidden size
    num_layers = 1  # number of RNN layers
    dropout_keep_prob = 0.9  # Dropout keep_prob rate (not drop_prob)
    init_w = 0.1  # init weight scale

    max_turn = 15

    batch_size = 32
    display_frequency = 200  # display_frequency
    checkpoint_frequency = 2000  # checkpoint frequency
