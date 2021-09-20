import re
import pickle
import utils
import vae_model
import torch
from torch import optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_value_
import json
import os
import pdb
import sys
import pandas as pd
import numpy as np
import random
import logger_utils
import loss_utils
import yaml

from sklearn.metrics import roc_auc_score, accuracy_score
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False

global MODEL_SAVE_PATH 

def run(args, device):
    if not os.path.exists(f'{args.savedir}/hidden_states'):
        os.makedirs(f'{args.savedir}/hidden_states')



    print("gpu device:", device)
    MODEL_SAVE_PATH = logger_utils.get_save_dir(args)
    log_err_train, log_err_valid, log_err_test = logger_utils.get_all_logger(typ="err", args=args)
    log_samp_train, log_samp_valid, log_samp_test = logger_utils.get_all_logger(typ="samp", args=args)

    ### Loading Data
    # we start with train so that if the vocab f is missing, it is constructed from train
    train_dataset, train_dataloader = utils.prep_datasets(args, device, json_fn=args.train_fn, mode='train')
    _, test_dataloader = utils.prep_datasets(args, device, json_fn=args.test_fn, mode="test")
    _, valid_dataloader = utils.prep_datasets(args, device, json_fn=args.valid_fn, mode="valid")

    loss_names = ['train','valid','test']
    if args.dataset=="all":
        # this is for reddit-cmv 
        loss_names.append('test2')
        _, test_dataloader2 = utils.prep_datasets(args, device, json_fn=args.test_fn2, mode="test")

    if args.framework == "bert":
        print("--- Running bert baseline delta predictor ---")
        model = vae_model.DeltaPredictor(z_dim=768, h_dim=768,
                z_combine=args.z_combine,device=device)

    else:
        model = vae_model.RNNVAE(nwords=train_dataset.vocab_size+1,
                                encoder=args.encoder,
                                framework=args.framework,
                                z_dim=args.z_dim,
                                h_dim=args.h_dim,
                                rnngate=args.rnngate,
                                device=device,
                                freeze_weights=args.freeze,
                                z_combine=args.z_combine,
                                scale_pzvar=args.scale_pzvar, 
                                z_summary=args.zsum)
    
        new_emb_fn = args.new_emb + f".{args.nwords}"
        model.load_embeddings(fn=new_emb_fn)
    try:
        save_fp = f'{MODEL_SAVE_PATH}-{args.l_epoch}.pt'
        model.load_state_dict(torch.load(save_fp))
        print("loaded from:", args.l_epoch)
    except:
        print("Could not load file or missing model path, train from scratch.")
        
    start_e = args.l_epoch
    end_e = args.l_epoch+args.num_epochs
    start_e = 0
    end_e = args.num_epochs

    with open(f'configs/config{args.confign}.yml', 'r') as f:
        config = yaml.safe_load(f)
    opt = config['optm']
    
    model.cuda() #
    optimizer = optim.Adam(model.parameters(), lr=opt['lr'], weight_decay=opt['wd'])
    logdir=f"runs/{args.dataset}{args.seed}/{args.framework}-{args.zsum}-{args.confign}-{args.ss_recon_loss}-{args.hyp}"
    stoploss = loss_utils.StopLoss(loss_names, logdir=logdir) 
    anneal_w = utils.frange_cycle_sigmoid(config['cycle'])
    
    steps = 0
    for epoch in range(start_e, end_e):
        if epoch >= len(anneal_w):
            pass
        else:
            kl_w = anneal_w[epoch]
    #    steps, kl_w = utils.anneal(steps, incr, 20000)
            
        print("epoch:", epoch, "steps:", steps, "kl_old:", np.around(kl_w, 8)) #, 'anneal_new:', anneal_w[epoch])
        ### TRAIN
        model.train()
        train_metricsD, train_lossD, _ = run_epoch_(train_dataloader, log_err_train, 
                                                log_samp_train, model, 
                                                optimizer, train_dataset.ix2w, epoch, kl_w, 
                                                args, mode="train", device=device,
                                                word_dropout=args.word_dropout)
        
        ### SAVE
        #if epoch%10==0 and epoch!=0:
        torch.save(model.state_dict(), f'{MODEL_SAVE_PATH}-{epoch}.pt')
        model.eval()

        ### VALID, TEST
        # refactor such that I can see valid and test metrics. 
        #
        valid_metricsD, valid_lossD, _ = run_epoch_(valid_dataloader, log_err_valid, log_samp_valid, model, 
                                                optimizer, train_dataset.ix2w, epoch, kl_w, 
                                                args, mode="valid", device=device,
                                                word_dropout=0)

        test_metricsD, test_lossD, hidden_states  = run_epoch_(test_dataloader, log_err_test, log_samp_test, model, 
                                                optimizer, train_dataset.ix2w, epoch, kl_w, 
                                                args, mode="test(ID)", device=device, word_dropout=0)
        metrics = {'train':np.mean(train_metricsD[args.eval_metric]),
                'valid': np.mean(valid_metricsD[args.eval_metric]),
                'test':np.mean(test_metricsD[args.eval_metric])}


        if args.dataset == "all":
            test_metricsD2, test_lossD2, hidden_states2 = run_epoch_(test_dataloader2, log_err_test, log_samp_test, 
                                    model, optimizer, train_dataset.ix2w, epoch, kl_w, 
                                    args, mode="test(CD)", device=device, word_dropout=1)

            metrics['test2'] = np.mean(test_metricsD2[args.eval_metric])
        
        # may introduce probs later
            stoploss.write_tb(epoch=epoch, typ='metrics', train=train_metricsD, 
                          valid=valid_metricsD, test=test_metricsD, test2=test_metricsD2)

            stoploss.write_tb(epoch=epoch, typ='loss', train=train_lossD, 
                        valid=valid_lossD, test=test_lossD, test2=test_lossD2)

        else:
            stoploss.write_tb(epoch=epoch, typ='metrics', train=train_metricsD, 

                          valid=valid_metricsD, test=test_metricsD)

            stoploss.write_tb(epoch=epoch, typ='loss', train=train_lossD, 
                        valid=valid_lossD, test=test_lossD)

        stoploss.update(metrics)
        if stoploss.check(epoch):
            save_fp = f'{MODEL_SAVE_PATH}-{stoploss.max_epoch}.pt'
            model.load_state_dict(torch.load(save_fp))

            test_metricsD, test_lossD, hidden_states  = run_epoch_(test_dataloader, log_err_test, log_samp_test, model, 
                                                optimizer, train_dataset.ix2w, epoch, kl_w, 
                                                args, mode="test(ID)", device=device, word_dropout=0)

            os.makedirs(f'{args.savedir}/hidden_states_ID', exist_ok=True)      
            with open(f'{args.savedir}/hidden_states_ID/val_hyp{args.hyp}_seed{args.seed}.p', 'wb') as f:
                pickle.dump(hidden_states, f)

            if args.dataset == "all":

                test_metricsD2, test_lossD2, hidden_states2 = run_epoch_(test_dataloader2, log_err_test, log_samp_test, 
                                        model, optimizer, train_dataset.ix2w, epoch, kl_w, 
                                        args, mode="test(CD)", device=device, word_dropout=1)

                metrics['test2'] = np.mean(test_metricsD2[args.eval_metric])
                
                os.makedirs(f'{args.savedir}/hidden_states_CD', exist_ok=True)
                with open(f'{args.savedir}/hidden_states_CD/val_hyp{args.hyp}_seed{args.seed}.p', 'wb') as f:
                    pickle.dump(hidden_states2, f)

                print("hidden states written to:", f'{args.savedir}/hidden_states')

            return stoploss.report_max()
    
    stoploss.check(epoch, stop=True)
    return stoploss.report_max()



def run_epoch_(dataloader, log_err, log_samp, model, optimizer, ix2w,
        epoch, kl_w, args, mode=None, device="cpu", word_dropout=1):

    # The losses and weights we care about for multi-task learning
    lossD, weightsD = loss_utils.construct_lossD(f'configs/config{args.confign}.yml', type_=0)
    lossD_epoch, _ = loss_utils.construct_lossD(f'configs/config{args.confign}.yml', type_=[])

    # The metrics we care about
    # These do not get considered in backpropagation loss
    metricsD_epoch = {"acc":[], "roc":[], "acc_indiv":[]}
    hidden_states = {}


    for i, (threadID, OP, all_CO_pos, all_CO_neg, all_CO_irr) in enumerate(dataloader):
        # OP is always a dictionary
        # OP['xx'].shape = [batch_size, seq_len, dim] = [1, n_seq, dim]
        # if use unsupervised. then we extract the relevant unsupervised data for additional
        # training
        hidden_states[threadID[0]] = {}

        if len(all_CO_pos)==0 and len(all_CO_neg)==0:
            continue

        if args.framework == "bert":
            if len(all_CO_pos)>0:
                pos_out, lossD = model_bert_delta(model, OP, all_CO_pos, lossD, delta=1)
                #pass

            if len(all_CO_neg)>0:
                neg_out, lossD = model_bert_delta(model, OP, all_CO_neg, lossD, delta=0)
                #pass

        else:
            # Random Dropout
            if mode=="train":
                OP['ey'] = utils.drop_words(OP['ey'], OP['y_lens'], word_dropout)

                for CO_pos in all_CO_pos:
                    CO_pos['ey'] = utils.drop_words(CO_pos['ey'], CO_pos['y_lens'], word_dropout)

                for CO_neg in all_CO_neg:
                    CO_neg['ey'] = utils.drop_words(CO_neg['ey'], CO_neg['y_lens'], word_dropout)

                for CO_irr in all_CO_irr:
                    CO_irr['ey'] = utils.drop_words(CO_irr['ey'], CO_irr['y_lens'], word_dropout)


            #### TRANS_ENCODER VS RNN_ENCODER
            #### ENCODER
            prior_mu = torch.zeros(args.z_dim).cuda() #
            #print(mode, threadID[0])
            #print('1')

            OP_bce_loss, OP_kld_loss, OP_zs = model_encdec(model, OP, prior_mu, kl_w)
            lossD['bce'] += OP_bce_loss

            if args.framework == "vae":
                lossD['kld'] += OP_kld_loss

            if args.zsum != "ffn_pairwise":
                OP_z = model.summarise_z(OP_zs)
                if args.zsum == "rnnv":
                    # variational loss if summarizer is rnnv
                    zsum_qmu, zsum_qlogvar = model.summariser_z.get_params()
                    OP_zsum_kld = model.kld_loss(prior_mu, zsum_qmu, zsum_qlogvar)
                    lossD['kld'] += kl_w * OP_zsum_kld
            
            #print('2')

            if len(all_CO_pos)>0:
                OP_pos_zs, CO_pos_zs, lossD = model_summz(model, OP_zs, all_CO_pos, lossD, \
                                                            prior_mu, kl_w, args.zsum)
                
                pos_out, lossD = model_deltaz(model, OP_pos_zs, OP_z, CO_pos_zs, lossD, delta=1)

            if len(all_CO_neg)>0:
                OP_neg_zs, CO_neg_zs, lossD = model_summz(model, OP_zs, all_CO_neg, lossD, \
                                                            prior_mu, kl_w, args.zsum)

                neg_out, lossD = model_deltaz(model, OP_neg_zs, OP_z, CO_neg_zs, lossD, delta=0)

            if len(all_CO_irr)>0:
            #    #print("CO_irr0:", lossD)
                OP_irr_zs, CO_irr_zs, lossD = model_summz(model, OP_zs, all_CO_irr, lossD, \
                        prior_mu, kl_w, args.zsum, incl_loss=args.ss_recon_loss)
                #print("CO_irr1:", lossD)
                # no delta loss for all_CO_irr
            #### TRIPLET LOSS
            if args.triplet_thresh>0 and len(all_CO_neg)>0 and len(all_CO_pos)>0 and len(all_CO_irr)>0:
                lossD['md'] += model.triplet_loss(OP_z, CO_pos_zs, CO_neg_zs, CO_irr_zs, \
                        args.triplet_thresh, hyp=args.hyp)
                hidden_states[threadID[0]]['OP'] = OP_z.detach().cpu().numpy()
                hidden_states[threadID[0]]['CO_neg'] = CO_neg_zs.detach().cpu().numpy()
                hidden_states[threadID[0]]['CO_pos'] = CO_pos_zs.detach().cpu().numpy()
                hidden_states[threadID[0]]['CO_irr'] = CO_irr_zs.detach().cpu().numpy()
                

            if args.contrast_thresh>0 and len(all_CO_neg)>0 and len(all_CO_pos)>0:
                CO_pos_zsum = model.summarise_z(CO_pos_zs)
                CO_neg_zsum = model.summarise_z(CO_neg_zs)
                lossD['md'] += model.contrast_loss(CO_pos_zsum, CO_neg_zsum, args.contrast_thresh)
                    

        ### MARGIN LOSS
        if len(all_CO_pos)>0 and len(all_CO_neg)>0:
            y_pred = []
            y_true = []
            
            if args.framework == "bert":
                lossD['margin'] += model.margin_loss(pos_out, neg_out, margin=args.margin_thresh)

            else:
                lossD['margin'] += model.delta_nn.margin_loss(pos_out, neg_out, margin=args.margin_thresh)

            
            pos_cpu = pos_out.squeeze().detach().cpu().numpy()
            neg_cpu = neg_out.squeeze().detach().cpu().numpy()

            if np.mean(pos_cpu)>np.mean(neg_cpu):
                metricsD_epoch['acc'].append(1)
            else:
                metricsD_epoch['acc'].append(0)

            y_pred = np.hstack((pos_cpu, neg_cpu))
            y_true = np.hstack((np.ones(len(all_CO_pos)), np.zeros(len(all_CO_neg))))
            metricsD_epoch['roc'].append(roc_auc_score(y_true, y_pred))
            
            # Calculate accuracy with individual speakers
            y_pred_acc = np.zeros(len(all_CO_pos)+len(all_CO_neg))
            y_pred_acc[np.where(y_pred >= 0.5)]=1
            y_pred_acc[np.where(y_pred < 0.5)]=2
            
            y_true2 = np.hstack((np.ones(len(all_CO_pos)), np.ones(len(all_CO_neg))*2))
            metricsD_epoch['acc_indiv'].append(accuracy_score(y_true2, y_pred_acc))

        # average over Pos and Neg Deltas -- should we be averaging across all delta_losses?
        #delta_loss = delta_loss / (len(CO_pos_z)+len(CO_neg_z))

        if mode == "train":
            loss = loss_utils.total_loss(weightsD, lossD)
            if loss ==0 :
                pass
            else:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # release computation graph
        lossD_epoch = loss_utils.copy_loss(weightsD, lossD, lossD_epoch)
        lossD = loss_utils.reset_loss(lossD)

    print("\n--{}-- ".format(mode.upper()))
    loss_utils.print_loss(lossD_epoch, mode=mode, epoch=epoch)
    loss_utils.print_loss(metricsD_epoch, mode=mode, epoch=epoch)
    #print("OP mu:", OP_q_mu[0][0][0:5].data.cpu().numpy())
    #sys.stdout.flush()
    #sos_ix = 5
    #input0 = model.embedding(torch.LongTensor([sos_ix]).cuda() #)
    #z = torch.randn(5, args.z_dim)

    return metricsD_epoch, lossD_epoch, hidden_states


def model_summz(model, OP_zs, CO, lossD, prior_mu, kl_w, zsum, incl_loss=1):
    all_OP_z = []
    all_CO_z = [] 

    bce_loss = 0
    kld_loss = 0

    for i, C in enumerate(CO):
        # pass the average OP_z value immediately
        C_bce_loss, C_kld_loss, C_zs = model_encdec(model, C, prior_mu, kl_w)

        bce_loss += C_bce_loss
        kld_loss += C_kld_loss

        if zsum == "ffn_pairwise":
            OP_pos_z, C_z = model.ffn_pairwise_z(OP_zs, C_zs)
            all_OP_z.append(OP_pos_z)

        else:
            C_z = model.summarise_z(C_zs)
            #if zsum == "rnnv" and incl_loss==1:
            #    zsum_qmu, zsum_qlogvar = model.summariser_z.get_params()
           #     C_zsum_kld = model.kld_loss(prior_mu, zsum_qmu, zsum_qlogvar)

           #     lossD['kld']+= kl_w * C_zsum_kld

        all_CO_z.append(C_z)

    all_CO_z = torch.stack(all_CO_z)

    if incl_loss == 1:
        # incl loss for irr for semi-supervised
        lossD['bce'] += bce_loss/len(CO)
        lossD['kld'] += kld_loss/len(CO)
    
    return all_OP_z, all_CO_z, lossD

def model_bert_delta(model, OP, COs, lossD, delta=1):

    all_out = []
    for CO in COs:
        out = model(OP['bert_embed'].mean(1), CO['bert_embed'].mean(1))
        all_out.append(out)
        lossD['delta'] += model.delta_loss(out, delta=delta)

    all_out = torch.stack(all_out)
    lossD['delta'] = lossD['delta']/len(COs)
    return all_out, lossD

def model_deltaz(model, OP_zs, OP_z, CO_zs, lossD, delta=1):
    assert len(CO_zs)>0

    if len(OP_zs) == 0:
        OP_zs = OP_z.unsqueeze(0).repeat(len(CO_zs), 1)
    else:
        OP_zs = torch.stack(OP_zs)
    
    # Separate score and loss, because we reuse the score for margin loss calculation.
    delta_predict = model.delta_nn(OP_zs, CO_zs)
    
    # reduction="mean"
    lossD['delta'] += model.delta_nn.delta_loss(delta_predict, delta=delta)
    return delta_predict, lossD

def model_encdec(model, OP, prior_mu, kl_w, check=False):
    # check if normalized by number of sentences

    OP_xr, OP_q_mu, OP_q_logvar, OP_zs = model(OP)
    bce_loss = model.recon_loss(OP['ye'], OP['y_lens'], OP_xr, check)
    OP_kld = model.kld_loss(prior_mu, OP_q_mu, OP_q_logvar)
    
    kld_loss = kl_w * (OP_kld/OP['ye'].squeeze().size(0))

    return bce_loss, kld_loss, OP_zs


def get_test_id(args, device):
    _, test_dataloader = utils.prep_datasets(args, device, json_fn=args.test_fn, mode="test")
    # hack for getting test id
