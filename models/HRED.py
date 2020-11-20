# -*- coding:utf-8 -*-

import tensorflow.contrib.seq2seq as tc_seq2seq
from tensorflow.python.layers import core as layers_core
from models.model_base import *
from models.model_helper import *


class HREDModel(BaseTFModel):
    def __init__(self, config, mode, scope=None):
        super(HREDModel, self).__init__(config, mode, scope)

    def _build_graph(self):
        self._build_placeholders()
        self._build_embeddings()
        self._build_encoder()
        self._build_decoder()
        if self.mode != ModelMode.infer:
            self._compute_loss()
            if self.mode == ModelMode.train:
                self.create_optimizer(self.loss)
                # Training Summary
                self.train_summary = tf.summary.merge([tf.summary.scalar("lr", self.learning_rate),
                                                       tf.summary.scalar("train_loss", self.loss)]
                                                      + self.grad_norm_summary)

    def _build_placeholders(self):
        with tf.variable_scope("placeholders"):
            batch_size = None
            dialog_turn_size = None
            dialog_sent_size = None

            self.source = tf.placeholder(tf.int32,
                                         shape=[batch_size, dialog_turn_size, dialog_sent_size],
                                         name="dialog_inputs")

            self.source_length = tf.placeholder(tf.int32,
                                                shape=[batch_size, dialog_turn_size],
                                                name='dialog_input_lengths')
            self.source_topic  = tf.placeholder(tf.int32,
                                                shape=[batch_size, dialog_turn_size],
                                                name='dialog_input_topic')

            self.target_input = tf.placeholder(tf.int32,
                                               shape=[batch_size, None],
                                               name="response_input_sent")
            self.target_output = tf.placeholder(tf.int32,
                                                shape=[batch_size, None],
                                                name="response_output_sent")
            self.target_length = tf.placeholder(tf.int32,
                                                shape=[None],
                                                name="target_length")

            self.dropout_keep_prob = tf.placeholder(tf.float32)

            self.batch_size = tf.shape(self.source)[0]
            self.turn_size = tf.shape(self.source)[1]
            self.sent_size = tf.shape(self.source)[2]

            if self.mode != ModelMode.infer:
                self.predict_count = tf.reduce_sum(self.target_length)

    def _build_embeddings(self):
        with tf.variable_scope("dialog_embeddings"):
            dialog_embeddings = tf.get_variable("dialog_embeddings",
                                                shape=[self.config.vocab_size, self.config.emb_size],
                                                dtype=tf.float32,
                                                trainable=True)
            # share encoder and decoder vocabulary
            self.encoder_embeddings = dialog_embeddings
            self.decoder_embeddings = dialog_embeddings

            topic_embeddings = tf.get_variable("topic_embeddings",
                                                shape=[self.config.topic_size, self.config.emb_size],
                                                dtype=tf.float32,
                                                trainable=True)
            # share encoder and decoder vocabulary
            self.encoder_topic_embeddings = topic_embeddings

    def _build_encoder(self):
        with tf.variable_scope("dialog_encoder"):
            with tf.variable_scope('utterance_rnn'):
                uttn_hidden_size = self.config.emb_size * 2
                uttn_encoder = RNNEncoder(unit_type='gru',
                                          enc_type='uni',
                                          hidden_size=uttn_hidden_size,
                                          num_layers=self.config.num_layers,
                                          dropout_keep_prob=self.dropout_keep_prob)

                uttn_emb_inp = tf.nn.embedding_lookup(self.encoder_embeddings,
                                                      tf.reshape(self.source, [-1, self.sent_size]))
                print('utterance input embs shape', uttn_emb_inp.shape)

                _, uttn_states = uttn_encoder(uttn_emb_inp, tf.reshape(self.source_length, [-1]))

                uttn_states = tf.reshape(uttn_states, [self.batch_size,
                                                       self.turn_size,
                                                       uttn_hidden_size])
                print('utterance shape', uttn_states.shape)

            with tf.variable_scope("context_rnn"):
                context_encoder = RNNEncoder(unit_type=self.config.unit_type,
                                             enc_type='bi',
                                             hidden_size=self.config.enc_hidden_size,
                                             num_layers=self.config.num_layers,
                                             dropout_keep_prob=self.dropout_keep_prob)

                context_turn_length = tf.reduce_sum(tf.sign(self.source_length), axis=1)
                uttn_topic_inp = tf.nn.embedding_lookup(self.encoder_topic_embeddings,
                                                      tf.reshape(self.source_topic, [-1, self.turn_size]))
                print('utterance topic embs shape', uttn_topic_inp.shape)
                uttn_states_new = tf.concat([uttn_topic_inp,uttn_states],2)
                ctx_outputs, ctx_state = context_encoder(uttn_states_new,  context_turn_length)

                self.encoder_outputs = ctx_outputs
                self.encoder_state = ctx_state

    def _build_decoder_cell(self, enc_outputs, enc_state):
        beam_size = self.config.beam_size
        context_length = self.source_length
        memory = enc_outputs

        if self.mode == ModelMode.infer and beam_size > 0:
            enc_state = tc_seq2seq.tile_batch(enc_state,
                                              multiplier=beam_size)

            memory = tc_seq2seq.tile_batch(memory,
                                           multiplier=beam_size)

            context_length = tc_seq2seq.tile_batch(context_length,
                                                   multiplier=beam_size)

            batch_size = self.batch_size * beam_size

        else:
            enc_state = enc_state
            batch_size = self.batch_size

        dec_cell = get_rnn_cell(self.config.unit_type,
                                hidden_size=self.config.dec_hidden_size,
                                num_layers=self.config.num_layers,
                                dropout_keep_prob=self.dropout_keep_prob)

        return dec_cell, enc_state

    def _build_decoder(self):
        with tf.variable_scope("dialog_decoder"):
            with tf.variable_scope("decoder_output_projection"):
                output_layer = layers_core.Dense(
                    self.config.vocab_size, use_bias=False, name="output_projection")

            with tf.variable_scope("decoder_rnn"):
                dec_cell, dec_init_state = self._build_decoder_cell(enc_outputs=self.encoder_outputs,
                                                                    enc_state=self.encoder_state)

                # Training or Eval
                if self.mode != ModelMode.infer:  # not infer, do decode turn by turn
                    resp_emb_inp = tf.nn.embedding_lookup(self.decoder_embeddings, self.target_input)
                    helper = tc_seq2seq.TrainingHelper(resp_emb_inp, self.target_length)
                    decoder = tc_seq2seq.BasicDecoder(
                        cell=dec_cell,
                        helper=helper,
                        initial_state=dec_init_state,
                        output_layer=output_layer
                    )

                    dec_outputs, dec_state, _ = tc_seq2seq.dynamic_decode(decoder)
                    sample_id = dec_outputs.sample_id
                    logits = dec_outputs.rnn_output

                else:
                    beam_width = self.config.beam_size
                    length_penalty_weight = self.config.length_penalty_weight
                    maximum_iterations = tf.to_int32(self.config.infer_max_len)
                    start_tokens = tf.fill([self.batch_size], self.config.sos_idx)
                    end_token = self.config.eos_idx

                    # beam size
                    decoder = tc_seq2seq.BeamSearchDecoder(
                        cell=dec_cell,
                        embedding=self.decoder_embeddings,
                        start_tokens=start_tokens,
                        end_token=end_token,
                        initial_state=dec_init_state,
                        beam_width=beam_width,
                        output_layer=output_layer,
                        length_penalty_weight=length_penalty_weight)

                    dec_outputs, dec_state, _ = tc_seq2seq.dynamic_decode(
                        decoder,
                        maximum_iterations=maximum_iterations,
                    )
                    logits = tf.no_op()
                    sample_id = dec_outputs.predicted_ids

                self.logits = logits
                self.sample_id = sample_id

    def _compute_loss(self):
        with tf.variable_scope('loss'):
            """Compute optimization loss."""
            batch_size = tf.shape(self.target_output)[0]
            max_time = tf.shape(self.target_output)[1]

            crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.target_output, logits=self.logits)
            target_weights = tf.sequence_mask(self.target_length, maxlen=max_time, dtype=self.logits.dtype)
            loss = tf.reduce_sum(crossent * target_weights) / tf.to_float(batch_size)
            self.loss = loss
            self.exp_ppl = tf.reduce_sum(crossent * target_weights)
        pass

    def train(self, sess, batch_input):
        assert self.mode == ModelMode.train
        feed_dict = {
            self.source: batch_input.source,
            self.source_length: batch_input.source_length,
            self.source_topic : batch_input.source_topic,
            self.target_input: batch_input.target_input,
            self.target_output: batch_input.target_output,
            self.target_length: batch_input.target_length,
            self.dropout_keep_prob: self.config.dropout_keep_prob
        }
        res = sess.run([self.update_opt,
                        self.loss,
                        self.exp_ppl,
                        self.predict_count,
                        self.batch_size,
                        self.train_summary,
                        self.global_step], feed_dict)
        return res[1:]
        pass

    def eval(self, sess, batch_input):
        assert self.mode == ModelMode.eval
        feed_dict = {
            self.source: batch_input.source,
            self.source_length: batch_input.source_length,
            self.source_topic : batch_input.source_topic,
            self.target_input: batch_input.target_input,
            self.target_output: batch_input.target_output,
            self.target_length: batch_input.target_length,
            self.dropout_keep_prob: 1.0
        }
        res = sess.run([self.loss,
                        self.exp_ppl,
                        self.predict_count,
                        self.batch_size,
                        self.global_step], feed_dict)
        return res

    def infer(self, sess, batch_data):
        assert self.mode == ModelMode.infer
        feed_dict = {
            self.source: batch_data.source,
            self.source_length: batch_data.source_length,
            self.source_topic: batch_data.source_topic,
            self.dropout_keep_prob: 1.0
        }
        res = sess.run([self.sample_id,
                        self.batch_size], feed_dict)
        return res
