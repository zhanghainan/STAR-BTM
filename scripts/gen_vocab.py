# -*- coding:utf-8 -*-
import os
os.sys.path.append(os.path.join(os.path.dirname(__file__), "./../"))
from utils.vocab import *


def gen_vocab(input_file, saved_path, vocab_size=None):
    vocab = Vocabulary()
    with open(input_file, 'r', encoding='utf-8') as fr:
        for line in fr:
            words = line.strip().split()
            for word in words:
                vocab.add(word)
    origin_size = vocab.size
    if vocab_size is not None:
        vocab = vocab.prune(vocab_size)
        print("prune vocab from %d to %d" % (origin_size, vocab.size))
    print('save vocabulary into %s' % saved_path)
    vocab.write_file(saved_path)


if __name__ == '__main__':
    gen_vocab('../data/ubuntu/train.dialog.txt', "../data/ubuntu/vocab.dialog.txt", vocab_size=20000)
    gen_vocab('../data/ubuntu-10k/train.dialog.txt', "../data/ubuntu-10k/vocab.dialog.txt", vocab_size=10000)
    pass
