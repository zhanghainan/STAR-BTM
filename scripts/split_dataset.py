# -*- coding:utf-8 -*-
import numpy as np

def split(dialog_file, saved_file):
    dialogs = []
    turns = []
    lengths = []
    with open(dialog_file, 'r', encoding='utf-8') as fr:
        for line in fr:
            line = line.replace("**unknown**", "<unk>")
            sents = line.strip().split('__eot__')
            turns.append(len(sents))
            lengths += [len(s.split()) for s in sents]
            dialogs.append(sents)
    print("samples:\t%d" % len(dialogs))
    print('average turns:\t%d; max turn:\t%d' % (np.mean(turns), max(turns)))
    print('average lengths:\t%d; max length:\t%d' % (np.mean(lengths), max(lengths)))

    with open(saved_file, 'w', encoding='utf-8') as fw:
        for sents in dialogs:
            fw.write('</d>'.join(sents) + '\n')


def split_small(dialog_file, saved_dialog_file, num_samples=10000):
    dialogs = [line.strip() for line in open(dialog_file, 'r', encoding='utf-8')]

    with open(saved_dialog_file, 'w', encoding='utf-8') as fw:
        for line in dialogs[:num_samples]:
            fw.write(line + '\n')


def split_context_response_pairs(dialog_file, ctx_file=None, resp_file=None):
    ctx_file = dialog_file.replace("dialog", "context") if ctx_file is None else ctx_file
    resp_file = dialog_file.replace("dialog", "response") if resp_file is None else resp_file
    dialog_lines = [line.strip().split('</d>') for line in open(dialog_file, 'r', encoding='utf-8')]
    with open(ctx_file, 'w', encoding='utf-8') as ctx_fw, \
            open(resp_file, 'w', encoding='utf-8') as resp_fw:
        for dialog_line in dialog_lines:
            assert len(dialog_line) >= 2
            ctx_fw.write('</d>'.join(dialog_line[:-1]) + '\n')
            resp_fw.write(dialog_line[-1] + '\n')


if __name__ == '__main__':
    # split('./raw_training_text.txt', "./ubuntu/train.dialog.txt")
    # split_small("./ubuntu/train.dialog.txt", './ubuntu-10k/train.dialog.txt', 10000)
    # split('./raw_valid_text.txt', "./ubuntu/valid.dialog.txt")
    # split_small("./ubuntu/valid.dialog.txt", './ubuntu-10k/valid.dialog.txt', 1000)
    # split('./raw_test_text.txt', "./ubuntu/test.dialog.txt")
    # split_small("./ubuntu/test.dialog.txt", './ubuntu-10k/test.dialog.txt', 1000)

    # split_context_response_pairs("./ubuntu/valid.dialog.txt")
    # split_context_response_pairs("./ubuntu/test.dialog.txt")
    split_context_response_pairs("../data/ubuntu-10k/valid.dialog.txt")
    split_context_response_pairs("../data/ubuntu-10k/test.dialog.txt")
