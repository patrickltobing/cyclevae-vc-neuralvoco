#!/usr/bin/env python

# Copyright 2021 Patrick Lumban Tobing (Nagoya University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import torch

import logging


def main():
    parser = argparse.ArgumentParser(
        description="making feature file argsurations.")

    parser.add_argument("--expdir", required=True,
                        type=str, help="directory to save log")
    parser.add_argument("--confdir", required=True,
                        type=str, help="directory of model config.")
    parser.add_argument("--verbose", default=1,
                        type=int, help="log message level")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # set log level
    if args.verbose == 1:
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S',
                            filename=args.expdir + "/get_model_indices.log")
        logging.getLogger().addHandler(logging.StreamHandler())
    elif args.verbose > 1:
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S',
                            filename=args.expdir + "/get_model_indices.log")
        logging.getLogger().addHandler(logging.StreamHandler())
    else:
        logging.basicConfig(level=logging.WARN,
                            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S',
                            filename=args.expdir + "/get_model_indices.log")
        logging.getLogger().addHandler(logging.StreamHandler())
        logging.warn("logging is disabled.")

    checkpoint = torch.load(os.path.join(args.expdir, "checkpoint-last.pkl"), map_location=torch.device("cpu"))
    last_epoch = checkpoint["iterations"]
    min_idx_epoch = checkpoint["min_idx"]+1
    logging.info(args.expdir)
    logging.info(f'{last_epoch} {min_idx_epoch}')

    out_file = args.confdir+".idx"
    logging.info(out_file)
    f = open(out_file, 'w')
    f.write('%d %d\n' % (last_epoch, min_idx_epoch))
    f.close()


if __name__ == "__main__":
    main()
