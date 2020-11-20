# -*- coding:utf-8 -*-
import os

os.sys.path.append(os.path.join(os.path.dirname(__file__), "./../"))

import argparse
from utils.eval_utils import evaluate


def parse_args():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser('Dialog Generation Evaluation')

    parser.add_argument("--hyp_file", type=str,
                        default="../data/ubuntu/raw_testing_responses.txt", help="response predict file")
    parser.add_argument("--ref_file", type=str,
                        default="../data/ubuntu/ModelPredictions/VHRED/First_VHRED_BeamSearch_5_GeneratedTestResponses"
                                ".txt_First.txt", help="response gold file")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    bleu = evaluate(args.ref_file, args.hyp_file)
    print("Bleu@4:\t%.3f" % bleu)
