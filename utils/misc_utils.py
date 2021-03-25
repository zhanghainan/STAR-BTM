# -*- coding:utf-8 -*-
import os
import time
import sys
import math
import tensorflow as tf


def load_lines(path):
    lines = []
    with open(path, 'r', encoding='utf-8') as fr:
        for line in fr:
            lines.append(line.strip())
        return lines

def load_dialog_data(path, split_str='</d>'):
    dialog_data = []
    with open(path, 'r', encoding='utf-8') as fr:
        for line in fr:
            sents = line.strip().split(split_str)
            sents = [sent.split() for sent in sents]
            dialog_data += [sents]
    return dialog_data


def check_file_exist(path):
    return os.path.exists(path)


def mkdir_not_exists(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)


def save_sentences(sents, path):
    with open(path, 'w', encoding='utf-8') as fw:
        for sent in sents:
            fw.write(' '.join(sent) + '\n')


def get_config_proto(log_device_placement=False, allow_soft_placement=True):
    # GPU options:
    config_proto = tf.ConfigProto(
        log_device_placement=log_device_placement,
        allow_soft_placement=allow_soft_placement)
    config_proto.gpu_options.allow_growth = True
    return config_proto


def print_time(s, start_time):
    """Take a start time, print elapsed duration, and return a new time."""
    print("%s, time %ds, %s." % (s, (time.time() - start_time), time.ctime()))
    sys.stdout.flush()
    return time.time()


def safe_exp(value):
    """Exponentiation with catching of overflow error."""
    try:
        ans = math.exp(value)
    except OverflowError:
        ans = float("inf")
    return ans


def add_summary(summary_writer, global_step, tag, value):
    """Add a new summary to the current summary_writer.
  Useful to log things that are not part of the training graph, e.g., tag=BLEU.
  """
    summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
    summary_writer.add_summary(summary, global_step)
