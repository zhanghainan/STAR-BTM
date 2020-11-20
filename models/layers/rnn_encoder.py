# -*- coding:utf-8 -*-
import tensorflow as tf
import tensorflow.contrib as tc


def get_rnn_cell(unit_type, hidden_size, num_layers=1, dropout_keep_prob=None):
    """
    Gets the RNN Cell
    Args:
        unit_type: 'lstm', 'gru' or 'rnn'
        hidden_size: The size of hidden units
        num_layers: MultiRNNCell are used if num_layers > 1
        dropout_keep_prob: dropout in RNN
    Returns:
        An RNN Cell
    """
    if unit_type.endswith('lstm'):
        cell = tc.rnn.LSTMCell(num_units=hidden_size)
    elif unit_type.endswith('gru'):
        cell = tc.rnn.GRUCell(num_units=hidden_size)
    elif unit_type.endswith('rnn'):
        cell = tc.rnn.BasicRNNCell(num_units=hidden_size)
    else:
        raise NotImplementedError('Unsuported rnn type: {}'.format(unit_type))
    if dropout_keep_prob is not None:
        cell = tc.rnn.DropoutWrapper(cell,
                                     input_keep_prob=dropout_keep_prob,
                                     output_keep_prob=dropout_keep_prob)
    if num_layers > 1:
        cell = tc.rnn.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
    return cell


class RNNEncoder(object):
    def __init__(self, unit_type, enc_type, hidden_size, num_layers=1, dropout_keep_prob=None):
        self.enc_type = enc_type
        self.num_layers = num_layers
        self.unit_type = unit_type
        self.hidden_size = hidden_size
        if enc_type == 'uni':
            self._enc_cell = get_rnn_cell(unit_type=unit_type,
                                          hidden_size=hidden_size,
                                          num_layers=num_layers,
                                          dropout_keep_prob=dropout_keep_prob)
        elif enc_type == 'bi':
            bi_hidden_size = int(hidden_size / 2)
            self.hidden_size = bi_hidden_size * 2
            self._enc_fw_cell = get_rnn_cell(unit_type=unit_type,
                                             hidden_size=bi_hidden_size,
                                             num_layers=num_layers,
                                             dropout_keep_prob=dropout_keep_prob)

            self._enc_bw_cell = get_rnn_cell(unit_type=unit_type,
                                             hidden_size=bi_hidden_size,
                                             num_layers=num_layers,
                                             dropout_keep_prob=dropout_keep_prob)
        else:
            raise NotImplementedError("Not Implemented Encode Type: %s" % enc_type)

    def __call__(self, inputs, sequence_length=None, initial_state=None, *args, **kwargs):
        if self.enc_type == 'uni':
            enc_outputs, enc_state = tf.nn.dynamic_rnn(cell=self._enc_cell,
                                                       inputs=inputs,
                                                       sequence_length=sequence_length,
                                                       dtype=tf.float32,
                                                       initial_state=initial_state,
                                                       swap_memory=True)
            return enc_outputs, enc_state
        elif self.enc_type == 'bi':
            bi_enc_outputs, bi_enc_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=self._enc_fw_cell,
                                                                            cell_bw=self._enc_bw_cell,
                                                                            inputs=inputs,
                                                                            sequence_length=sequence_length,
                                                                            dtype=tf.float32,
                                                                            swap_memory=True)
            encoder_fw_state = bi_enc_states[0]
            encoder_bw_state = bi_enc_states[1]

            if isinstance(encoder_fw_state, tc.rnn.LSTMStateTuple):  # LstmCell
                state_c = tf.concat(
                    (encoder_fw_state.c, encoder_bw_state.c), 1, name="bidirectional_concat_c")
                state_h = tf.concat(
                    (encoder_fw_state.h, encoder_bw_state.h), 1, name="bidirectional_concat_h")
                encoder_final_state = tc.rnn.LSTMStateTuple(c=state_c, h=state_h)
            elif isinstance(encoder_fw_state, tuple) \
                    and isinstance(encoder_fw_state[0], tc.rnn.LSTMStateTuple):  # MultiLstmCell
                encoder_final_state = tuple(map(
                    lambda fw_state, bw_state: tc.rnn.LSTMStateTuple(
                        c=tf.concat((fw_state.c, bw_state.c), 1,
                                    name="bidirectional_concat_c"),
                        h=tf.concat((fw_state.h, bw_state.h), 1,
                                    name="bidirectional_concat_h")),
                    encoder_fw_state, encoder_bw_state))
            else:
                encoder_final_state = tf.concat(
                    (encoder_fw_state, encoder_bw_state), 1,
                    name="bidirectional_state_concat")

            return tf.concat(bi_enc_outputs, axis=-1), encoder_final_state
