import codecs
import os
import tensorflow as tf
from utils.metrics import bleu
from utils.metrics import rouge


def evaluate(ref_file, hyp_file, metric="bleu"):
    # BLEU scores for translation task
    if metric.lower() == "bleu":
        evaluation_score = _bleu(ref_file, hyp_file)
    # ROUGE scores for summarization tasks
    elif metric.lower() == "rouge":
        evaluation_score = _rouge(ref_file, hyp_file)
    elif metric.lower() == "accuracy":
        evaluation_score = _accuracy(ref_file, hyp_file)
    elif metric.lower() == "word_accuracy":
        evaluation_score = _word_accuracy(ref_file, hyp_file)
    else:
        raise ValueError("Unknown metric %s" % metric)

    return evaluation_score


def _clean(sentence):
    """Clean and handle BPE or SPM outputs."""
    sentence = sentence.strip()
    return sentence


# Follow //transconsole/localization/machine_translation/metrics/bleu_calc.py
def _bleu(ref_file, hyp_file, max_order=4):
    """Compute BLEU scores and handling BPE."""
    smooth = False

    ref_files = [ref_file]
    reference_text = []
    for reference_filename in ref_files:
        with codecs.getreader("utf-8")(
                tf.gfile.GFile(reference_filename, "rb")) as fh:
            reference_text.append(fh.readlines())

    per_segment_references = []
    for references in zip(*reference_text):
        reference_list = []
        for reference in references:
            reference = _clean(reference)
            reference_list.append(reference.split(" "))
        per_segment_references.append(reference_list)

    translations = []
    with codecs.getreader("utf-8")(tf.gfile.GFile(hyp_file, "rb")) as fh:
        for line in fh:
            line = _clean(line)
            translations.append(line.split(" "))

    # bleu_score, precisions, bp, ratio, translation_length, reference_length
    bleu_score, _, _, _, _, _ = bleu.compute_bleu(
        per_segment_references, translations, max_order, smooth)
    return 100 * bleu_score


def _rouge(ref_file, summarization_file):
    """Compute ROUGE scores and handling BPE."""

    references = []
    with codecs.getreader("utf-8")(tf.gfile.GFile(ref_file, "rb")) as fh:
        for line in fh:
            references.append(_clean(line))

    hypotheses = []
    with codecs.getreader("utf-8")(
            tf.gfile.GFile(summarization_file, "rb")) as fh:
        for line in fh:
            hypotheses.append(_clean(line))

    rouge_score_map = rouge.rouge(hypotheses, references)
    return 100 * rouge_score_map["rouge_l/f_score"]


def _accuracy(label_file, pred_file):
    """Compute accuracy, each line contains a label."""

    with codecs.getreader("utf-8")(tf.gfile.GFile(label_file, "rb")) as label_fh:
        with codecs.getreader("utf-8")(tf.gfile.GFile(pred_file, "rb")) as pred_fh:
            count = 0.0
            match = 0.0
            for label in label_fh:
                label = label.strip()
                pred = pred_fh.readline().strip()
                if label == pred:
                    match += 1
                count += 1
    return 100 * match / count


def _word_accuracy(label_file, pred_file):
    """Compute accuracy on per word basis."""

    with codecs.getreader("utf-8")(tf.gfile.GFile(label_file, "r")) as label_fh:
        with codecs.getreader("utf-8")(tf.gfile.GFile(pred_file, "r")) as pred_fh:
            total_acc, total_count = 0., 0.
            for sentence in label_fh:
                labels = sentence.strip().split(" ")
                preds = pred_fh.readline().strip().split(" ")
                match = 0.0
                for pos in range(min(len(labels), len(preds))):
                    label = labels[pos]
                    pred = preds[pos]
                    if label == pred:
                        match += 1
                total_acc += 100 * match / max(len(labels), len(preds))
                total_count += 1
    return total_acc / total_count
