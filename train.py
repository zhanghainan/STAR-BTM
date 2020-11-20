# -*- coding:utf-8 -*-
import os
import time
import argparse
from utils.vocab import *
from models.HRED import HREDModel
from models.model_base import ModelMode
from configs import *
from utils.iterator import *
import utils.misc_utils as utils
from utils.eval_utils import evaluate

import nltk
import numpy as np
def run_evaluate(sess, eval_model, infer_model,
                 vocab, config, global_step, data_dir, pred_dir,
                 data_iter, mode):
    assert mode in ["valid", "test"]
    loss, ppl = _run_internal_eval(sess, eval_model, data_iter)

    ctx_file = os.path.join(data_dir, "%s.context.txt" % mode)
    resp_file = os.path.join(data_dir, "%s.response.txt" % mode)

    infer_iter = get_infer_iter(context_file=ctx_file, vocab=vocab, config=config)
    pred_sents = _run_external_eval(sess, infer_model, infer_iter, vocab)

    pred_tgt_file = os.path.join(pred_dir, "%s_e%d_ppl_%.2f_loss_%.2f.pred.txt" %
                                 (mode, global_step, ppl, loss))

    utils.save_sentences(pred_sents, pred_tgt_file)

    resp_file = open(resp_file)
    resp=[]
    for sen in resp_file:
        resp.append(sen.strip())
    resp_file.close()
    bleu=[]
    for res,pred in zip(resp,pred_sents):
        res=res.split()[1:]
        #pred=pred.split()
        bleu.append(nltk.translate.bleu_score.sentence_bleu([pred],res,(0.25, 0.25, 0.25, 0.25),nltk.translate.bleu_score.SmoothingFunction().method1))

    #bleu = evaluate(resp_file, pred_tgt_file)
    return loss, ppl, 100*np.mean(bleu)


def _run_internal_eval(sess, eval_model, eval_iter):
    eval_loss, eval_predict_count, eval_samples = 0.0, 0, 0
    for batch_data in eval_iter.next_batch():
        step_loss, step_word_count, step_predict_count, batch_size, _ = eval_model.eval(sess, batch_data)

        eval_samples += batch_size
        eval_loss += step_loss * batch_size
        eval_predict_count += step_predict_count

    return eval_loss / eval_samples, utils.safe_exp(eval_loss / eval_predict_count)


def _run_external_eval(sess, infer_model, infer_iter, tgt_vocab, num_response_per_input=1):
    predict_sents = []
    for infer_batch_data in infer_iter.next_batch():
        batch_ids, batch_size = infer_model.infer(sess, infer_batch_data)

        for sent_id in range(batch_size):
            for beam_id in range(num_response_per_input):
                predict_id = batch_ids[sent_id, :, beam_id].tolist()
                predict_sent = tgt_vocab.convert2words(predict_id)
                predict_sents.append(predict_sent)
    return predict_sents


def load_vocab_setup_config(args):
    # load vocab from precessed vocab file
    vocab_file = os.path.join(args.data_dir, "vocab.dialog.txt")
    vocab = load_vocabulary(vocab_file)

    # get config
    config = eval(args.config)
    # setup vocab related config
    config.vocab_size = vocab.size
    config.sos_idx = vocab.sos_idx
    config.eos_idx = vocab.eos_idx

    return vocab, config


def train(args):
    start_time = time.time()
    vocab, model_config = load_vocab_setup_config(args)
    print('... load vocab and setup model config over, cost:\t%.2f s' % (time.time() - start_time))
    print('... vocab size:\t%d' % vocab.size)

    start_time = time.time()
    train_iter, valid_iter, test_iter = get_train_iter(args.data_dir, vocab=vocab, config=model_config)
    print('-' * 100)
    print('... load train and valid data iterator over, cost:\t%.2f s' % (time.time() - start_time))
    print('... train iterator samples:\t%d' % train_iter.num_samples)
    print('... valid iterator samples:\t%d' % valid_iter.num_samples)
    print('... test iterator samples:\t%d' % test_iter.num_samples)

    # prepare output dir
    output_dir = args.output_dir
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    ckpt_dir2 = os.path.join(output_dir, "checkpoints2")
    ckpt_dir3 = os.path.join(output_dir, "checkpoints3")
    log_dir = os.path.join(output_dir, "train_log")
    pred_dir = os.path.join(output_dir, "pred")

    utils.mkdir_not_exists(output_dir)
    utils.mkdir_not_exists(ckpt_dir)
    utils.mkdir_not_exists(ckpt_dir2)
    utils.mkdir_not_exists(ckpt_dir3)
    utils.mkdir_not_exists(log_dir)
    utils.mkdir_not_exists(pred_dir)
    ckpt_path = os.path.join(ckpt_dir, model_config.model)
    ckpt_path2 = os.path.join(ckpt_dir2, model_config.model)
    ckpt_path3 = os.path.join(ckpt_dir3, model_config.model)
    print('=' * 100)
    print('... building model')
    start_time = time.time()
    if model_config.model == 'HRED':
        model = HREDModel
    else:
        raise NotImplementedError("No such model")

    config_proto = tf.ConfigProto()
    #config_proto.gpu_options.per_process_gpu_memory_fraction = 0.9
    config_proto.gpu_options.allow_growth=True
    with tf.Session(config=config_proto) as sess:
        initializer = tf.random_uniform_initializer(-1.0 * model_config.init_w, model_config.init_w)
        scope = model_config.model
        with tf.variable_scope(scope, reuse=None, initializer=initializer):
            train_model = model(config=model_config, mode=ModelMode.train, scope=scope)
        with tf.variable_scope(scope, reuse=True, initializer=initializer):
            eval_model = model(config=model_config, mode=ModelMode.eval, scope=scope)
        with tf.variable_scope(scope, reuse=True, initializer=initializer):
            infer_model = model(config=model_config, mode=ModelMode.infer, scope=scope)

        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        ckpt2 = tf.train.get_checkpoint_state(ckpt_dir2)
        ckpt3 = tf.train.get_checkpoint_state(ckpt_dir3)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            train_model.saver.restore(sess, ckpt.model_checkpoint_path)
            print("Reading model parameters from %s" % ckpt2.model_checkpoint_path)
            eval_model.saver.restore(sess, ckpt2.model_checkpoint_path)
            print("Reading model parameters from %s" % ckpt3.model_checkpoint_path)
            infer_model.saver.restore(sess, ckpt3.model_checkpoint_path)
        else:
            print('... create %s model over, time cost: %.2fs' % (model_config.model, time.time() - start_time))
            print('=' * 100)
            sess.run(tf.global_variables_initializer())
        # Summary writer
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

        early_stop=0
        pre_valid_loss = 10000
        start_time = time.time()
        ckpt_loss, ckpt_ppl, ckpt_predict_count, ckpt_samples = 0.0, 0.0, 0, 0
        for epoch in range(model_config.max_epoch):
            train_samples = 0
            if early_stop>=4:
                break
            for batch_data in train_iter.next_batch():
                # print("batch data shape:", batch_data.dialog.shape, "batch length shape:", batch_data.dialog_length.shape)
                # print(batch_data.dialog_length)
                step_loss, step_ppl, step_predict_count, batch_size, step_summary, global_step = \
                    train_model.train(sess, batch_data)

                ckpt_samples += batch_size
                ckpt_loss += step_loss * batch_size
                ckpt_ppl += step_ppl
                ckpt_predict_count += step_predict_count
                train_samples += batch_size

                summary_writer.add_summary(step_summary, global_step)

                if global_step % model_config.display_frequency == 0:
                    train_loss = ckpt_loss / ckpt_samples
                    train_ppl = safe_exp(ckpt_ppl / ckpt_predict_count)
                    print('Epoch: %d/%d; Samples: %d/%d; Step: %d; Train Loss: %.2f; Train PPL: %.2f; Time Cost: %.2fs' %
                          (epoch + 1,
                           model_config.max_epoch,
                           train_samples,
                           train_iter.num_samples,
                           global_step,
                           train_loss,
                           train_ppl,
                           time.time() - start_time))
                    utils.add_summary(summary_writer, global_step, "train_ppl", train_ppl)

                    ckpt_loss, ckpt_ppl, ckpt_predict_count, ckpt_samples = 0.0, 0.0, 0, 0
                    start_time = time.time()

                if global_step % model_config.checkpoint_frequency == 0:
                    print("--------- evaluate model ------------")
                    start_time = time.time()
                    valid_loss, valid_ppl, valid_bleu = run_evaluate(sess, eval_model, infer_model,
                                                                     vocab, model_config,
                                                                     global_step,
                                                                     args.data_dir,
                                                                     pred_dir,
                                                                     valid_iter, "valid")

                    print('Epoch: %d/%d; Step: %d; Valid Loss: %.2f; Valid PPL: %.2f; Valid Bleu:%.2f; Time Cost: %.2fs' %
                          (epoch + 1,
                           model_config.max_epoch,
                           global_step,
                           valid_loss,
                           valid_ppl,
                           valid_bleu,
                           time.time() - start_time))
                    if valid_loss> pre_valid_loss:
                        early_stop+=1
                    else:
                        early_stop=0
                    if early_stop>=4:
                        break
                    pre_valid_loss=valid_loss
                    start_time = time.time()

                    test_loss, test_ppl, test_bleu = run_evaluate(sess, eval_model, infer_model,
                                                                  vocab, model_config,
                                                                  global_step,
                                                                  args.data_dir,
                                                                  pred_dir,
                                                                  test_iter, "test")

                    print('Epoch: %d/%d; Step: %d; Test Loss: %.2f; Test PPL: %.2f; Test Bleu:%.2f; Time Cost: %.2fs' %
                          (epoch + 1,
                           model_config.max_epoch,
                           global_step,
                           test_loss,
                           test_ppl,
                           test_bleu,
                           time.time() - start_time))

                    pred_tgt_file = os.path.join(pred_dir, "%s_e%d_ppl_%.2f_loss_%.2f.pred.txt" %
                                 ("valid", global_step, valid_ppl, valid_loss))
                    out = open(pred_tgt_file,"a")
                    # save summary and checkpoints
                    out.write('Train Loss: %.2f Train PPL: %.2f Valid Loss: %.2f Valid PPL: %.2lf Valid Bleu: %.5lf Test Loss: %.2f Test PPL: %.2lf Test Bleu: %.5lf\n' %
                          (train_loss,
                           train_ppl,
                           valid_loss,
                           valid_ppl,
                           valid_bleu,
                           test_loss,
                           test_ppl,
                           test_bleu,
                           ))
                    out.close()
                    utils.add_summary(summary_writer, global_step, "valid_ppl", valid_ppl)
                    utils.add_summary(summary_writer, global_step, "valid_bleu", valid_bleu)
                    utils.add_summary(summary_writer, global_step, "test_ppl", test_ppl)
                    utils.add_summary(summary_writer, global_step, "test_bleu", test_bleu)
                    summary_writer.flush()

                    train_model.save(sess, ckpt_path)
                    eval_model.save(sess, ckpt_path2)
                    infer_model.save(sess, ckpt_path3)
                    start_time = time.time()

    # done training
    summary_writer.close()

    pass


def parse_args():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser('Dialog Generation')

    parser.add_argument("--config", type=str,
                        default="HREDTestConfig", help="model config")
    parser.add_argument("--data_dir", type=str,
                        default="./data/ubuntu-10k/", help="training input dir")
    parser.add_argument("--output_dir", type=str,
                        default="./data/ubuntu_10k_output", help="training output dir")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
