# -*- coding:utf-8 -*-
from utils.vocab import *
from utils.iterator import *

vocab = load_vocabulary('../data/ubuntu/vocab.dialog.txt')
print('vocab size', vocab.size)

data_iter = get_dialog_data_iter(vocab=vocab,
                                 dialog_file='../data/ubuntu/valid.dialog.txt',
                                 batch_size=16,
                                 max_turn=10,
                                 max_len=100,
                                 infer=False,
                                 shuffle=False,
                                 bucket_config=[(4, 16), (6, 16), (8, 16), (10, 16)])

num_samples = 0
for batch_data in data_iter.next_batch():
    assert batch_data.target_input.shape == batch_data.target_output.shape
    # print(batch_data.target_output.shape)
    num_samples += batch_data.source.shape[0]
    pass
print("num samples:", num_samples)

for batch_data in data_iter.next_batch():
    print(batch_data.source.shape)
    print(batch_data.source)
    print(batch_data.target_input)
    print(batch_data.target_output)
    print(batch_data.target_length)
    break

infer_iter = get_dialog_data_iter(vocab=vocab,
                                  dialog_file='../data/ubuntu/test.context.txt',
                                  batch_size=16,
                                  max_turn=10,
                                  max_len=100,
                                  infer=True,
                                  shuffle=False)

print(infer_iter.num_samples)
count = 0
for batch_input in infer_iter.next_batch():
    count += batch_input.source.shape[0]
    # print(batch_input.source.shape)
print(count)
for batch_input in infer_iter.next_batch():
    print(batch_input.source)
    print(batch_input.source_length)
    break

