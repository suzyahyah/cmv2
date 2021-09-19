#!/usr/bin/python
# Author: Suzanna Sia

# from the project: https://pypi.org/project/sentence-transformers/
from sentence_transformers import SentenceTransformer


import utils
import torch
from torch import optim
import vae_model
import numpy as np
import pdb

embedding_dim = 768
h_dim = 1056

def run(args):

    device = torch.device("cuda:{}".format(args.cuda) if int(args.cuda)>=0 else "cpu")
    print("device:", device)
    train_ds = utils.JSONDataset(json_fn=args.train_fn, nwords=args.nwords,
            universal_embed=args.universal_embed, max_seq_len=args.max_seq_len)

    valid_ds = utils.JSONDataset(json_fn=args.valid_fn, nwords=args.nwords,
            universal_embed=args.universal_embed, max_seq_len=args.max_seq_len)

    test_ds = utils.JSONDataset(json_fn=args.test_fn, nwords=args.nwords, 
            universal_embed=args.universal_embed, max_seq_len=args.max_seq_len)


    transf = SentenceTransformer('bert-base-nli-mean-tokens', device=device)
    delta_nn = vae_model.DeltaPredictor(z_dim=embedding_dim, h_dim=h_dim, device=device)
    delta_nn.to(device)

    start_e = 0
    end_e = args.num_epochs
    optimizer = optim.Adam(delta_nn.parameters(), lr=0.001)

    for epoch in range(start_e, end_e):
        print("epoch:", epoch)

        delta_nn.train()
        run_epoch_(args, train_ds, optimizer, transf, delta_nn, device=device, mode="train")

        delta_nn.eval()
        run_epoch_(args, valid_ds, optimizer, transf, delta_nn, device=device, mode="valid")
        run_epoch_(args, test_ds, optimizer, transf, delta_nn, device=device, mode="test")


def run_epoch_(args, ds, optimizer, transf, delta_nn, device="cpu", mode="train"):

    delta_loss = 0
    total_n = 0
    score = 0
    eq = 0


    for i in range(len(ds.data)):
        thread_ID, OH, CO_pos, CO_neg, CO_irr = ds.data[i]
        OH_sent_emb = torch.mean(torch.tensor(transf.encode(OH)), dim=0).to(device)
        CO_neg_sent_emb = torch.mean(torch.tensor(transf.encode(CO_neg)), dim=0).to(device)
        CO_pos_sent_emb = torch.mean(torch.tensor(transf.encode(CO_pos)), dim=0).to(device)

        # DElTA LOSS
        pos_out = delta_nn(OH_sent_emb, CO_pos_sent_emb)
        neg_out = delta_nn(OH_sent_emb, CO_neg_sent_emb)

        delta1_loss = delta_nn.delta_loss(pos_out, delta=1)
        delta0_loss = delta_nn.delta_loss(neg_out, delta=0)

        dloss = (delta1_loss + delta0_loss)/2
        delta_loss += dloss

    
        if mode=="train":
            if i%args.update_itr==0:
                optimizer.zero_grad()
                delta_loss.backward()
                optimizer.step()
                delta_loss = 0
        else:
            delta_loss = 0

        
        item_score, eq = delta_nn.pairwise_predict(pos_out, neg_out)
        score += item_score
        total_n += 2

    acc = score/total_n
    print("--{}-- acc:{:.3f}".format(mode.upper(), acc))
