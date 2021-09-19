#!/usr/bin/python
# Author: Suzanna Sia
import utils
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import accuracy_score
import pdb

def run(args):
    train_ds = utils.JSONDataset(json_fn=args.train_fn, nwords=args.nwords, 
            universal_embed=args.universal_embed, max_seq_len=args.max_seq_len)

    train_ds.make_ix_dicts(args.old_emb, args.new_emb, args.vocab_fn)
    train_xs , train_ys = train_ds.proc_tfidf()

    test_ds = utils.JSONDataset(json_fn=args.test_fn, nwords=args.nwords, 
            universal_embed=args.universal_embed, max_seq_len=args.max_seq_len)
    test_ds.make_ix_dicts(args.old_emb, args.new_emb, args.vocab_fn)
    test_xs , test_ys = test_ds.proc_tfidf()

    logreg = LR()
    logreg.fit(train_xs, train_ys)

    probs = logreg.predict_proba(test_xs)
    pairwise_predict= []
    for i in range(len(test_ys)):
        if probs[i][0]>probs[i][1]:
            pairwise_predict.append(0)
        else:
            pairwise_predict.append(1)

    print("train:", len(train_ys), "test:", len(test_ys))
    acc = accuracy_score(test_ys, pairwise_predict)
    print("accuracy:", acc)
    print('log reg coef:', logreg.coef_) 


