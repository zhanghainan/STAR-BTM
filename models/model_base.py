# -*- coding:utf-8 -*-
import tensorflow as tf

"""
Base TensorFlow Model 
"""

from enum import Enum, unique


@unique
class ModelMode(Enum):  # three modes for models
    train = 1,
    eval = 2,
    infer = 3


class BaseTFModel(object):
    def __init__(self, config, mode, scope=None):
        self.scope = scope
        assert type(mode) == ModelMode

        self.mode = mode
        self.config = config

        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = None
        self.update_opt = None
        self.grad_norm_summary = None

        self._build_graph()
        self.saver = tf.train.Saver(max_to_keep=config.max_to_keep)
        pass

    def _get_learning_rate_decay(self):
        """Get learning rate decay."""
        start_decay_step = self.config.start_decay_step
        decay_steps = self.config.decay_steps
        decay_factor = self.config.decay_factor
        return tf.cond(
            self.global_step < start_decay_step,
            lambda: self.learning_rate,
            lambda: tf.train.exponential_decay(
                self.learning_rate,
                (self.global_step - start_decay_step),
                decay_steps, decay_factor, staircase=True),
            name="learning_rate_decay_cond")

    def _get_optimizer(self):
        # Optimizer
        if self.config.optimizer == "sgd":
            opt = tf.train.GradientDescentOptimizer(self.learning_rate)
            tf.summary.scalar("lr", self.learning_rate)

        elif self.config.optimizer == "adam":
            assert float(
                self.config.learning_rate
            ) <= 0.001, "! High Adam learning rate %g" % self.config.learning_rate
            opt = tf.train.AdamOptimizer(self.learning_rate)
        else:
            raise NotImplementedError("Not Implemented Optimizer")
        return opt

    @staticmethod
    def _gradient_clip(gradients, max_gradient_norm):
        """Clipping gradients of a model."""
        clipped_gradients, gradient_norm = tf.clip_by_global_norm(
            gradients, max_gradient_norm)

        gradient_norm_summary = [tf.summary.scalar("grad_norm", gradient_norm),
                                 tf.summary.scalar("clipped_gradient", tf.global_norm(clipped_gradients))]

        return clipped_gradients, gradient_norm_summary

    def create_optimizer(self, loss):
        """Creates the training operation"""
        """Creates the optimizer"""
        # learning rate
        self.learning_rate = tf.constant(self.config.learning_rate)
        # decay
        self.learning_rate = self._get_learning_rate_decay()
        # optimizer
        optimizer = self._get_optimizer()
        # variables and gradients
        tvars = tf.trainable_variables()
        # Gradients
        gradients = tf.gradients(loss, tvars)

        # Clip gradients
        clipped_gradients, gradient_norm_summary = self._gradient_clip(
            gradients, max_gradient_norm=self.config.max_gradient_norm)

        update_opt = optimizer.apply_gradients(
            zip(clipped_gradients, tvars), global_step=self.global_step)

        self.update_opt = update_opt
        self.grad_norm_summary = gradient_norm_summary

    def _build_graph(self):
        raise NotImplementedError

    def _build_placeholders(self):
        raise NotImplementedError

    def _compute_loss(self):
        raise NotImplementedError

    def train(self, sess, data_iter):
        raise NotImplementedError

    def eval(self, sess, data_iter):
        raise NotImplementedError

    def save(self, sess, saved_path):
        self.saver.save(sess, saved_path, global_step=self.global_step)
