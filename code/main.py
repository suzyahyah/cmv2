#!/usr/bin/python
# Author: Suzanna Sia

# Import Testing conditions

# set all random seed here
import random
import numpy as np
import os
import torch
import pdb
import sys
import full_task
import tfidf_baseline
import bert_baseline
#import perplex_task
#import pair_task


# ARGPARSE
import argparse
from distutils.util import strtobool

argparser = argparse.ArgumentParser()

# Seed
argparser.add_argument('--new_task', type=int, default=0)
argparser.add_argument('--seed', type=int, default=0)
argparser.add_argument('--confign', type=int, default=0)
argparser.add_argument('--dataset', type=str, default="all")
# filenames
argparser.add_argument('--old_emb', type=str)
argparser.add_argument('--new_emb',  type=str)
argparser.add_argument('--vocab_fn', type=str)
argparser.add_argument('--title_embed_fn', type=str)
argparser.add_argument('--train_fn', type=str)
argparser.add_argument('--valid_fn', type=str)
argparser.add_argument('--test_fn', type=str)
argparser.add_argument('--test_fn2', type=str)
argparser.add_argument('--savedir', type=str)
# mode
argparser.add_argument('--freeze', type=lambda x: bool(strtobool(x)), default=False)
argparser.add_argument('--cuda', dest='cuda', type=str, default="-1")
argparser.add_argument('--num_epochs', dest='num_epochs', type=int)
argparser.add_argument('--batch_size', dest='batch_size', type=int)
#argparser.add_argument('--lr', dest='lr', type=float)
argparser.add_argument('--nwords', dest='nwords', type=int)
argparser.add_argument('--l_epoch', dest='l_epoch', type=int)
argparser.add_argument('--framework', dest='framework', type=str)
argparser.add_argument('--encoder', dest='encoder', type=str)
argparser.add_argument('--hidden_dim', dest='h_dim', type=int)
argparser.add_argument('--latent_dim', dest='z_dim', type=int)
argparser.add_argument('--rnngate', dest='rnngate', type=str)
argparser.add_argument('--n_layers', dest='n_layers', type=int)
argparser.add_argument('--max_seq_len', dest='max_seq_len', type=int, default=25)
argparser.add_argument('--ss_recon_loss', dest='ss_recon_loss', type=int, default=0)
argparser.add_argument('--eval_metric', dest='eval_metric', type=str, default="acc")
argparser.add_argument('--balanced', dest='balanced', type=int, default=0)

#z
argparser.add_argument('--triplet_thresh', type=float, default=0.0)
argparser.add_argument('--contrast_thresh', type=float, default=0.0)
argparser.add_argument('--margin_thresh', type=float, default=0.0)
argparser.add_argument('--universal_embed', type=lambda x: bool(strtobool(x)), default=False)
argparser.add_argument('--z_combine', type=str)
argparser.add_argument('--zsum', type=str)
argparser.add_argument('--word_dropout', type=float, default=0.0)
argparser.add_argument('--scale_pzvar', type=float, default=1.0)
argparser.add_argument('--update_itr', type=int)
argparser.add_argument('--stopwords', type=int) # type=lambda x: bool(strtobool(x)), default=False)
argparser.add_argument('--use_prior_mu', type=lambda x: bool(strtobool(x)), default=False)
argparser.add_argument('--hyp', type=int, default=1)


args = argparser.parse_args()
device = torch.device("cuda:{}".format(args.cuda) if int(args.cuda)>=0 else "cpu")
print("device:", device)

def set_seed(seed):
    print("> set SEED:", seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":

    if args.framework == "tfidf_baseline":
        tfidf_baseline.run(args, device)
    elif args.framework == "bert_baseline":
        bert_baseline.run(args, device)

    if args.dataset == "all":
        results_dict = full_task.run(args, device)
        results_dict['ID_score'] = results_dict.pop('test_score')
        results_dict['CD_score'] = results_dict.pop('test_score2')
        args = vars(args)
        args.update(results_dict)

        keys = "train_score valid_score ID_score CD_score train_epochs encoder framework zsum \
        triplet_thresh hyp ss_recon_loss seed balanced".split()

        result = ""
        for k in keys:
            result += "{}\t".format(args[k])

        result = result.strip()
        if not os.path.exists('results/full_task.txt'):
            with open('results/full_task.txt', 'w') as f:
                f.write(",".join(keys))


        with open('results/full_task.txt', 'a') as f:
            f.write(result+"\n")
            

    elif args.dataset == "pair":
        pair_task.run(args, device)

    elif args.dataset == "IQ2":

        # fix this
        results_dict = full_task.run(args, device)
        fn = f"{args.encoder}-{args.triplet_thresh}-{args.zsum}-{args.framework}-{args.seed}.txt"

        fns_test = os.listdir('data/IQ2_corpus/test')
        fn_test = fns_test[0]
        threadID = fn_test[:fn_test.find('-')]

        args = vars(args)
        args.update(results_dict)

        keys = "train_score valid_score test_score train_epochs encoder framework zsum \
        triplet_thresh hyp ss_recon_loss seed balanced".split()

        leave_one_out = True

        if leave_one_out:
            keys.insert(0, "tID")
            args['tID'] = threadID

        result = ""
        for k in keys:
            result += "{}\t".format(args[k])

        result = result.strip()

        with open('results/debates_task_leave5_contrast0.1_h256_z128.txt', 'a') as f:
            f.write(result+"\n")
 
