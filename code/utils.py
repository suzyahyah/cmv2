from collections import Counter
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from nltk.tokenize import sent_tokenize
import itertools
import os
import process_debates
import torch
import pickle
import pdb
import json
import numpy as np
from gensim.models import KeyedVectors

import sys
import prep

import random

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer as tfvect

from sentence_transformers import SentenceTransformer
#from debugger import Debugger
#DB = Debugger()
#DB.debug_mode=True

#DEBUG_MODE=True
DEBUG_MODE=False

#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False


class JSONDataset(Dataset):

    def __init__(self, args, device=None):

        self.nwords = args['nwords']
        self.encoder = args['encoder']
        self.max_seq_len = args['max_seq_len']
        self.device = device
        self.w2ix = {'<pad>':0, '<unk>':1, 'N':2, '<eos>': 3, 'DISAGREE':4, '<sos>':5}
        self.ix2w = {v:k for k, v in self.w2ix.items()}
        self.universal_embed = args['universal_embed']
        self.word_dropout = args['word_dropout']

#        self.data, self.vocab_words =  self.read_json_cmu(json_fn)

        self.vocab_size = 0
        self.stopwords = args['stopwords']

        self.max_length = -1
        self.min_length = float('inf')

        print("using device:", device)
        self.encoder_model = self.get_encoder(encoder=self.encoder)
        self.sw = self.prep_stopwords()


    def __len__(self):
        return len(self.data)

    def get_encoder(self, encoder="", device=""):
        if encoder=="infersent":

            directory="/home/ssia/packages/InferSent"
            W2V_PATH ="/home/ssia/projects/cmv/data/embeddings/all/glove_w2vec.txt"
            sys.path.append(directory)
            
            from models import InferSent
            model_version = 1
            MODEL_PATH = os.path.join(directory, "encoder/infersent%s.pkl" % model_version)
            # cmu paper used 168 or 192
            params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                                    'pool_type': 'max', 'dpout_model': 0.0, 'version':
                                    model_version, 'device':device}
            model = InferSent(params_model)
            model.load_state_dict(torch.load(MODEL_PATH))
            model = model.cuda(device)
            model.set_w2v_path(W2V_PATH)
            #model.build_vocab_k_words(K=100000)
            model.build_vocab(self.vocab_words)
            return model.cuda()

        elif encoder=="bert":
            # make sure i'm using gpu here.
            return SentenceTransformer('bert-base-nli-mean-tokens').cuda() #, device=device)

    def get_sent_v(self, sent):
        sent = sent.split()[:self.max_seq_len]
        sent = [self.w2ix[w] if w in self.w2ix else self.w2ix['<unk>'] for w in sent]
        sent = torch.LongTensor(sent).cuda() # to(self.device)

        return sent

    def __getitem__(self, ix):
        #OP, CO, delta = self.data[ix]
        return self.data[ix]

    def prep_stopwords(self):
        with open('data/stopwords_long.txt', 'r') as f:
            sw = f.readlines()
        self.sw = [w.strip() for w in sw]
        suffix = ["n't", "'ve", "'d", "'re", "'ll", "'s"]
        sw = [word.strip() for word in sw]
        sw = [w.replace(x, '') for w in sw for x in suffix]
        sw.extend(suffix)

        return sw

   
    def replace_sw(self, sent):

        # keep stopwords, contentwords=<unk>
        if self.stopwords==1:
            #sent = [w if w in self.sw else '<unk>' for w in sent.split()]
            sent = [w for w in sent.split() if w in self.sw]

        # keep contentwords, stopwords=<unk> 
        elif self.stopwords==2:
            #sent = [w if w not in self.sw else '<unk>' for w in sent.split()]
            sent = [w for w in sent.split() if w not in self.sw]
        else:
            return sent
        return " ".join(sent)



    def proc_sent(self, OP):
        OPd = {}

        sos2 = torch.LongTensor([self.w2ix['<sos>']]).cuda() 
        eos3 = torch.LongTensor([self.w2ix['<eos>']]).cuda() 

        # stopword replacement here
        id, OP = OP
        OP_xx = [self.get_sent_v(self.replace_sw(s)) for s in OP]
        OP_xx = [x for x in OP_xx if len(x) >0]
        if len(OP_xx)==0:
            return None
        OP_ey = [torch.cat((sos2, s), dim=0) for s in OP_xx]
        OP_ye = [torch.cat((s, eos3), dim=0) for s in OP_xx]


        OPd['x_lens'] = [len(s) for s in OP_xx]
        OPd['y_lens'] = [len(s) for s in OP_ey]

        OPd['xx'] = pad_sequence(OP_xx, batch_first=True, padding_value=0)
        OPd['ye'] = pad_sequence(OP_ye, batch_first=True, padding_value=0)
        OPd['ey'] = pad_sequence(OP_ey, batch_first=True, padding_value=0)
        OPd['id'] = id

        if self.encoder!="glove":
            sents = []
            for i, op in enumerate(OP):
                sents.append(" ".join(self.replace_sw(op)))

            if self.encoder=="bert":
                OPd['bert_embed'] = torch.tensor(self.encoder_model.encode(sents)).cuda()# to(self.device)

            elif self.encoder=="infersent":
                OPd['infersent_embed'] = torch.tensor(self.encoder_model.encode(sents,
                    tokenize=False, verbose=False)).cuda()#, device=self.device)) #.to(self.device)

        return OPd

    def proc_tfidf(self):
        data_dict = []
        all_docs = []
        for ix in range(len(self.data)):
            thread_ID, OP, CO_pos, CO_neg, CO_irr = self.data[ix]
            all_docs.append(" ".join(OP))
            all_docs.append(" ".join(CO_pos))
            all_docs.append(" ".join(CO_neg))
            #all_docs.append(" ".join(CO_irr))
        
        tfidfvect = tfvect(stop_words='english')
        tfidfvect.fit(all_docs)

        xs = []
        ys = []

        for ix in range(len(self.data)):
            thread_ID, OP, CO_pos, CO_neg, _ = self.data[ix]
            CO_pos = " ".join(CO_pos)
            CO_neg = " ".join(CO_neg)
            OP = " ".join(OP)

            CO_pos_set = set(CO_pos.split())
            CO_neg_set = set(CO_neg.split())
            OP_set = set(OP.split())

            pos_jacc = len(CO_pos_set.intersection(OP_set))
            neg_jacc = len(CO_neg_set.intersection(OP_set))

            pos_jacc = pos_jacc/len(CO_pos_set.union(OP_set))
            neg_jacc = neg_jacc/len(CO_neg_set.union(OP_set))

            CO_pos_v = tfidfvect.transform([CO_pos])
            CO_neg_v = tfidfvect.transform([CO_neg])
            OP_v = tfidfvect.transform([OP])

            sim_pos = cosine_similarity(OP_v, CO_pos_v)[0][0]
            sim_neg = cosine_similarity(OP_v, CO_neg_v)[0][0]

           # xs.append((pos_jacc, sim_pos, len(CO_pos.split())))
           # xs.append((neg_jacc, sim_neg, len(CO_neg.split())))
           # xs.append((pos_jacc, sim_pos))
           # xs.append((neg_jacc, sim_neg))
            xs.append((pos_jacc, len(CO_pos.split())))
            xs.append((neg_jacc, len(CO_neg.split())))
 

            ys.append(1)
            ys.append(0)

        return xs, ys

    def proc_data(self):
        #json_fn = self.json_fn[self.json_fn.rfind('/')+1:]
        data_dict = []

        for ix in range(len(self.data)):

            thread_ID, OP, all_CO_pos, all_CO_neg, all_CO_irr = self.data[ix]
            
            OP =  self.proc_sent(OP)
            if OP is None:
                continue
            
            CO_pos = [self.proc_sent(c) for c in all_CO_pos]
            CO_neg = [self.proc_sent(c) for c in all_CO_neg]
            CO_irr = [self.proc_sent(c) for c in all_CO_irr]

            CO_pos = [c for c in CO_pos if c is not None]
            CO_neg = [c for c in CO_neg if c is not None]
            CO_irr = [c for c in CO_irr if c is not None]
            data_dict.append((thread_ID, OP, CO_pos, CO_neg, CO_irr))

        self.data = data_dict
        #with open(f"/export/b08/ssia/cmv/p/{train_fn}-{self.encoder}-data.p", 'wb') as f:
        #    pickle.dump(data_dict, f)
        #print("dumped data dict")


    def get_dataloader(self, batch_size=1):

        data_loader = DataLoader(dataset=self,
                                num_workers=0,
                                batch_size=batch_size,
                                shuffle=True)

        return data_loader

            # we need some word overlap but not completely or not too little...

            # dynamic triplet
            # build based on word overlap

#            for p1, p2, delta in pairs:
##                OH = threads[i]['ID_text'][p1]
#                CO = threads[i]['ID_text'][p2]


    def get_vocab(self, threads):
        raise NotImplementedError


    def make_ix_dicts(self, old_format_fn, new_emb_fn, vocab_fn):

        print("using stopwords:", self.stopwords)
        vocab_fn = vocab_fn+f".{self.nwords}"
        new_emb_fn = new_emb_fn+f".{self.nwords}"
        # construct ix from vocab

        if os.path.exists(vocab_fn):
            print("load from :", vocab_fn)
            with open(vocab_fn, 'r') as f:
                vocab = f.readlines()

            vocab = [l.strip() for l in vocab]

            i=0
            #for i, w in enumerate(lines):
            # the vocab has to correspond to the embedding later.
            # that's why we keep the index in the same order

            for w in vocab:
                if self.stopwords==1:
                    if w in self.sw:
                        self.w2ix[w] = i
                        self.ix2w[i] = w
                        i+=1
                
                elif self.stopwords==2:
                    if w not in self.sw:
                        self.w2ix[w] = i
                        self.ix2w[i] = w
                        i+=1

                else:
                    self.w2ix[w] = i
                    self.ix2w[i] = w
                    i+=1


        else:
            print(vocab_fn, " not found.. reconstructing vocab")
            # this is the full glove txt
            # we are going to make a smaller embedding and vocab

            embed = KeyedVectors.load_word2vec_format(old_format_fn)
            vocab_words = ['<pad>', '<unk>', 'N', '<eos>', 'DISAGREE', '<sos>']
            len_filler = len(vocab_words)

            new_embed = np.zeros((1, 300)) #zero pad
            filler = np.random.randn((len_filler-1),300) #rest of ix
            new_embed = np.vstack((new_embed, filler))

            i=0

            for w in self.vocab_words:
                if w not in embed.vocab:
                    continue

                vec = embed.word_vec(w)
                vocab_words.append(w)
                new_embed = np.vstack((new_embed, vec))
                self.w2ix[w] = i+len_filler
                self.ix2w[i+len_filler] = w
                i+=1

            vocab_txt = "\n".join(vocab_words)
            with open(vocab_fn, 'w') as f:
                f.write(vocab_txt)

            np.savetxt(new_emb_fn, new_embed, fmt="%.5f")
            print("saved new emb file:", new_emb_fn)
            print("saved new vocab file:", vocab_fn)
            sys.exit("rerun with new vocab")

        self.vocab_size = len(self.w2ix)
        print("vocab size1:", self.vocab_size)


class CMVDataset(JSONDataset):
    
    def __init__(self, args, device="cpu"):

        super().__init__(args, device=device)

        self.balanced = args['balanced']
        self.data, self.vocab_words = self.read_json_cmu(args['json_fn'])


    def read_json_cmu(self, fn):
        def _get_comment(thread_ID_text, thread_pairs):

            all_CO_ = []
            for pair in thread_pairs:
                id = pair[1]
                CO_ = thread_ID_text[id]
                CO_ = [s for s in CO_ if len(s.split())>5]

                if len(CO_)>0:
                    all_CO_.append((id, CO_))

            return all_CO_    


        with open(fn, encoding="utf-8") as f:
            threads = json.load(f)
       
        vocab_words = self.get_vocab(threads)

        skip = 0
        data = []
        pos = 0
        neg = 0
        irr = 0
        
        # COMMENT OUT
        if DEBUG_MODE:
            print("Debug mode....")
            threads = threads[:50]
            

        for i in range(len(threads)):
            pos_pairs = [(p1, p2) for (p1, p2, d) in threads[i]['ID_pairs'] if d==1]
            neg_pairs = [(p1, p2) for (p1, p2, d) in threads[i]['ID_pairs'] if d==0]
            #ulb_pairs = [(p1, p2) for (p1, p2, d) in threads[i]['ID_pairs'] if d==2]
            #if self.use_irr==1:
            irr_pairs = [(p1, p2) for (p1, p2, d) in threads[i]['ID_pairs'] if d==2]
            nchoice = min(len(pos_pairs)+len(neg_pairs), len(irr_pairs))
            irr_pairs = random.sample(irr_pairs, nchoice)

           # else:
           #     irr_pairs = [(p1, p2) for (p1, p2, d) in threads[i]['ID_pairs'] if (d==-1 or d==99)]
            if self.balanced==1:
                if len(pos_pairs)==0 or len(neg_pairs)==0:
                    continue


            
            while len(irr_pairs)<(len(pos_pairs)+len(neg_pairs)):
                p1 = threads[i]['thread_ID']

                # choose a random title from the threads
                irr_text = ''
                while len(irr_text.split())<6:
                    random_thread = random.choice(threads)
                    irr_text = random_thread['title']
                    p2 = 'irr_' + random_thread['thread_ID']

                irr_pairs.append((p1, p2))
                threads[i]['ID_text'][p2] = [irr_text]

            #if len(irr_pairs)==0:
            #    p1 = threads[i]['thread_ID']
            #    p2 = 'irr_random'
            #    irr_text = ""
            #    while len(irr_text.split())<6:
            #        irr_text = random.choice(threads)['title']

            #    irr_pairs = [(p1, p2)]
            #    threads[i]['ID_text'][p2] = [irr_text]


            pos += len(pos_pairs)
            neg += len(neg_pairs)
            irr += len(irr_pairs)

            thread_ID = threads[i]['thread_ID']
            OH = threads[i]['ID_text'][thread_ID]
            OH = [s for s in OH if len(s.split())>5]
            OH = (thread_ID, OH)
            thread_ID_text = threads[i]['ID_text']

            all_CO_pos = _get_comment(thread_ID_text, pos_pairs)
            all_CO_neg = _get_comment(thread_ID_text, neg_pairs)
            all_CO_irr = _get_comment(thread_ID_text, irr_pairs)
            #all_CO_ulb = _get_comment(thread_ID_text, ulb_pairs)

            data.append((thread_ID, OH, all_CO_pos, all_CO_neg, all_CO_irr))
        print(f"pos counts:{pos}, --neg counts:{neg} --irr counts:{irr}")
        return data, vocab_words


    def get_vocab(self, threads):
        # get all words in vocab that appear at least 3 times
        all_words = []
        for i in range(len(threads)):
            all_words.append(threads[i]['title'])
            texts = list(threads[i]['ID_text'].values())

            for t in texts:
                all_words.append("\n".join(t))

        all_words = "\n".join(all_words).split()
        c_all_words = Counter(all_words).most_common(self.nwords)
        vocab_words = [c[0] for c in c_all_words]
        
        #vocab_words = [w[0] for w in c_all_words.items() if w[1]>4]

        return vocab_words


class DebatesDataset(JSONDataset):
    
    def __init__(self, args, device="cpu"):

        super().__init__(args, device=device)
        self.sample = 10
        self.balanced = args['balanced']
        self.hyp = args['hyp']
        self.data, self.vocab_words = self.read_json_debates(args['json_fn'])

    def read_json_debates(self, fd):

        all_threads = []
        fns = os.listdir(fd)

        for fn in fns:
            fn = os.path.join(fd, fn)
            with open(fn, 'r') as f:
                thread = json.load(f)
            all_threads.append(thread)

        vocab_words = self.get_vocab(all_threads)

        data = []
        pos = 0
        neg = 0
        irr = 0

        if DEBUG_MODE:
            print("DEBUG MODE")
            all_threads = all_threads[:50]

        #delta_ix, nodelta_ix = process_debates.get_delta_ixs(all_threads, nstd=0.2)
        with open('data/IQ2_corpus/labels.txt', 'r') as f:
            labels = f.readlines()

        labels_D = {}
        for label in labels:
            label = label.strip().split('\t')
            labels_D[label[0]] = (label[1], label[2])

        #delta_fn = [l.split('\t')[0] for l in labels if int(l.split('\t')[1])==1]
        #nodelta_fn = [l.split('\t')[0] for l in labels if int(l.split('\t')[1])==0]

        pro_wins = 0
        con_wins = 0

        for i in range(len(all_threads)):
            fn = fns[i]

            thread = all_threads[i]
            thread_ID = thread['debateID']

            pros = process_debates.get_speaker(thread, 'pro')
            cons = process_debates.get_speaker(thread, 'con')
            OH = process_debates.get_speaker(thread, 'aud')

            OH = [sent_tokenize(s) for s in OH]
            OH = [s for ss in OH for s in ss]

            pros = [sent_tokenize(s) for s in pros]
            cons = [sent_tokenize(s) for s in cons]

            pros = [s for s in pros if len(s)>3]
            cons = [s for s in cons if len(s)>3]

            CO_irr = []
            CO_pos = []
            CO_neg = []

           #CO_irr.extend(process_debates.get_speaker(thread, 'mod'))


            if fn in labels_D:
                if labels_D[fn][0] == "1":
                    if labels_D[fn][1] == "pro":
                        CO_pos, CO_neg = pros, cons
                        pro_wins += 1
                    else:
                        CO_pos, CO_neg = cons, pros
                        con_wins += 1

                # if not balanced, use threads where there is no delta
                # only applies to training
                elif self.balanced == 0:
                    CO_neg = cons
                    CO_neg.extend(pros)
                else:
                    continue

            else:
                # or just treat it as no _delta.
                continue

            # get irrelevant comments (for hypothesis, and for ss_recon_loss)
            #if self.hyp!=0 or self.ss_recon_loss!=0:
            pre = process_debates.get_speaker(thread, 'pre')
            mod = process_debates.get_speaker(thread, 'mod')
            #pres = [sent_tokenize(s) for s in pre]
            mods = [sent_tokenize(s) for s in mod]
            #CO_irr.extend(pres)
            CO_irr = mods
            #CO_irr.extend(pres)
 

            OH = (thread_ID, OH)
            pos += len(CO_pos)
            neg += len(CO_neg)
            irr += len(CO_irr)

            all_CO_pos = [(thread_ID, p) for p in CO_pos]
            all_CO_neg = [(thread_ID, p) for p in CO_neg]
            all_CO_irr = [(thread_ID, p) for p in CO_irr]

            data.append((thread_ID, OH, all_CO_pos, all_CO_neg, all_CO_irr))

        print(f"pro wins:{pro_wins}, con wins:{con_wins}") 
        print(f"nthreads:{len(data)} pos counts:{pos}, --neg counts:{neg} --irr counts:{irr}")
        return data, vocab_words

    def get_vocab(self, threads):

        alltext = []
        for thread in threads:
            for content in thread['content']:
                for subcontent in content['subContent']:

                    if type(subcontent) is dict:
                        alltext.append(subcontent['sentenceContent'])

                    else:
                        for sent in subcontent:
                            alltext.append(sent['sentenceContent'])


        all_words = " ".join(alltext).split()
        c_all_words = Counter(all_words).most_common(self.nwords)
        vocab_words = [c[0] for c in c_all_words]

        return vocab_words


# deprecated
def anneal(steps, incr, anneal_steps):

    weight =  float(1/(1+np.exp(-0.0025*(steps-anneal_steps))))
    steps += incr

    if steps > anneal_steps:
        steps = steps - anneal_steps  # restart

    if anneal_steps==0:
        weight = 1
    
    return steps, weight


#def frange_cycle_sigmoid(start=0, stop=1, n_epoch=400, n_cycle=2, ratio=1):
def frange_cycle_sigmoid(params):
#https://github.com/haofuml/cyclical_annealing/blob/master/plot/plot_schedules.ipynb
    n_epoch = params['n_epoch']
    n_cycle = params['n_cycle']
    stop = params['stop']
    start = params['start']
    ratio = params['ratio']

    L = np.ones(n_epoch)
    period = n_epoch/n_cycle
    step = (stop-start)/(period*ratio) # step is in [0,1]
    
    for c in range(n_cycle):

        v , i = start , 0
        while v <= stop:
            L[int(i+c*period)] = 1.0/(1.0+ np.exp(- (v*12.-6.)))
            v += step
            i += 1
    return L    


def drop_words(ey, y_lens, word_dropout):

    unsqueeze=False
    if ey.shape[0]==1:
        ey = ey.squeeze(0)
        unsqueeze=True


    if word_dropout>0 and word_dropout<1:
        for i in range(ey.size(0)):
            drop_ = torch.rand(ey[i].size()).cuda()
            ey[i] *= (drop_ < word_dropout)

    if word_dropout==1:
        ey = torch.zeros_like(ey)

    if unsqueeze:
        ey = ey.unsqueeze(0)
    return ey

def tfidf_charts(fn):

    from matplotlib import pyplot as plt

    with open(fn, encoding='utf-8') as f:
        threads = json.load(f)


    all_pos_sims = []
    all_neg_sims = []

    for i in range(len(threads)):
        posIDs = [p2 for (p1, p2, d) in threads[i]['ID_pairs'] if d==True]
        negIDs = [p2 for (p1, p2, d) in threads[i]['ID_pairs'] if d==False]

        all_w = []
        IDs = []

        for k in threads[i]['ID_text'].keys():
            IDs.append(k)
            all_w.append((" ".join(threads[i]['ID_text'][k])))

        tfv = tfvect()
        X = tfv.fit_transform(all_w)
        X_sim = cosine_similarity(X)

        posIXs = [IDs.index(id) for id in posIDs]
        negIXs = [IDs.index(id) for id in negIDs]
        threadIX = IDs.index(threads[i]['thread_ID'])

        pos_sims = [X_sim[ix][threadIX] for ix in posIXs]
        neg_sims = [X_sim[ix][threadIX] for ix in negIXs]


        pos_sims = np.round(pos_sims, decimals=1)
        neg_sims = np.round(neg_sims, decimals=1)

        all_pos_sims.extend(pos_sims)
        all_neg_sims.extend(neg_sims)

    plt.hist(all_pos_sims)
    plt.show()

    # sentences
 

def prep_datasets(args, device, json_fn="", mode="train"):

    args = vars(args)
    args['json_fn'] = json_fn
    dataset = None
    
    print(f"mode:{mode}, fn:{json_fn}")

    if args['dataset'] == "IQ2":
        dataset = DebatesDataset(args, device=device)

    if args['dataset'] == "all":
        dataset = CMVDataset(args, device=device)

    dataset.make_ix_dicts(args['old_emb'], args['new_emb'], args['vocab_fn'])
    dataset.proc_data()
    dataloader = dataset.get_dataloader(args['batch_size'])
    print("--prep datasets done")

    return dataset, dataloader
