import pdb
import pandas as pd
import itertools
import json
import re
import nltk
from nltk.tokenize import sent_tokenize
import random
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
#import tensorflow as tf
#import tensorflow_hub as hub

import codecs
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from sentence_transformers import SentenceTransformer

#device = torch.device("cuda:{}".format(args.cuda) if int(args.cuda)>=0 else "cpu")


# After getting cmv we proceed to
# 1. Process json files
# 2. Preprocess text files to space out punctuation
# 3. Convert to paired (x,y) text files

def init():
    #process_cmu_ids(args.cmu_fn, args.json_fn, args.save_fn)
    import argparse

    argparser = argparse.ArgumentParser()
    ##argparser.add_argument('--cuda', type=str, default="-1")
    argparser.add_argument('--json_fn', type=str, default="")
    argparser.add_argument('--cmu_fn', type=str, default="")
    argparser.add_argument('--save_fn', type=str, default="")
    argparser.add_argument('--job_type', type=str, default="")

    args = argparser.parse_args()


    if args.job_type=="get_delta":
        get_delta_comments(args.cmu_fn, args.json_fn, args.save_fn)
    elif args.job_type=="process_all":
        process_cmu_ids2(args.cmu_fn, args.json_fn, args.save_fn)


def prep_glove(glove_fn):
    # after unzipping glove zip
    glove_file = datapath(glove_fn)
    tmp_file = get_tmpfile('/home/suzyahyah/projects/cmv/data/embeddings/glove_w2vec.txt')
    _ = glove2word2vec(glove_file, tmp_file)
    print("saved to :", tmp_file)


    # load all data, train and test
    # clean string and all that
    # get 20000 most common words
    # do make_ix_dict on dataset

def construct_vocab_list(embed_type="glove", emb_fn="", save_fn=""):

    if embed_type!="from_data":

        with open(emb_fn, 'r') as f:
            lines = f.readlines()

        lines = lines[1:]
        vocab = [l.split()[0] for l in lines]

        vocab = "\n".join(vocab)

        with open(save_fn, 'w') as f:
            f.write(vocab)

def get_reduced_embeddings(mode, dim):
    """ Get prior representation of the thread from its title
        1. Convert all titles to 768dim vectors using BERT Sentence Enc
        2. Fit PCA with d(arg) dimensions.
        3. Transform the titles to d dimensions
        4. Save to csv indexed by title ID
    """

    json_f = "data/all/{}_allthread_pairs.jsonlist".format(mode)
    with open(json_f, encoding="utf-8") as f:
        all_data = json.load(f)

    pca = PCA(n_components=dim)
    transf = SentenceTransformer('bert-base-nli-mean-tokens', device=device)

    all_zz = []
    all_sents = []
    sent_emb = []

    for i, data in enumerate(all_data):
    #    if i%10==0:
    #        print(mode, i)
    #    title = data['title'].replace('CMV', '')
    #    all_sents.extend([title])

        # Use all text instead of just titles
        for k in data['ID_text'].keys():
            author_sents = data['ID_text'][k]
            all_sents.extend(author_sents)

            #transf.encode([title])
            #all_sents.extend(author_sents)
    #        for sent in author_sents:
    #            sents = sent.split(',')
    #            sents = [s for s in sents if len(s.split())>5]
    #            all_sents.extend(sents)

    #    if len(all_sents)<=dim:
    #        rem = dim - len(all_sents) + 2 # artificially extend
    #        all_sents.extend([data['title']]*rem)

    print("transf encoding..", len(all_sents), " sents")
    sent_emb = transf.encode(all_sents)
    sent_emb = np.array(sent_emb)
    #zz = pca.fit_transform(sent_emb)
    print("pca fit...")
    pca.fit(sent_emb)

    for i, data in enumerate(all_data):
        if i%100==0:
            print(mode, i)
        title = data['title'].replace('CMV', '')
        #for k in data['ID_text'].keys():
        OP_text = tokenise_clean(data[i]['selftext'])
        #    author_sents = data['ID_text'][k]
        #pca.transform(transf.encode([title]))[0]
        zz = pca.transform(transf.encode([title]))[0]
        #zz = np.mean(zz, axis=0)
        assert(zz.shape[0] == dim), pdb.set_trace()
        all_zz.append([data['thread_ID'], zz])

    df = pd.DataFrame(all_zz, columns=['thread_ID', 'zz'])
    df.to_csv('data/pair_task/{}_threads_zz.csv'.format(mode))

def process_json_pair_task(fn=""):

    json_f = "data/pair_task/{}_pair_data.jsonlist".format(fn)
    data = []
    with open(json_f, encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))

    json_f = "data/all/{}_allthread_pairs.jsonlist".format(fn)
    with open(json_f, encoding="utf-8") as f:
        all_data = json.load(f)

    irrs = {}
    for ad in all_data:
        irr_ids = [ids for ids in ad['ID_pairs'] if ids[2]==-1]
 
        if len(irr_ids)==0:
            continue

        did = clean_id(ad['thread_ID'])
        irrs[did] = {}
        irr_id = irr_ids[0][1]
        irrs[did]['irr_text'] = ad['ID_text'][irr_id]
        irrs[did]['irr_id'] = irr_id


    all_threads = []

    print("total:", len(data))
    title_embed = {}

    #session, messages, output = prep_tf_model()

    for i in range(len(data)):

        thread = {}
        ID_text = {}
        ID_pairs = []

        pid = clean_id(data[i]['op_name'])
        ID_text[pid] = tokenise_clean(data[i]['op_text'])

        # doesnt take into account OH response. So we just collapse all the text..?
        pos_text = []

        pos_id = clean_id(data[i]['positive']['comments'][0]['id'])
        neg_id = clean_id(data[i]['negative']['comments'][0]['id'])

        for j in range(len(data[i]['positive']['comments'])):
            pos_text.append(data[i]['positive']['comments'][j]['body'])

        pos_text = "\n".join(pos_text)
        pos_text = tokenise_clean(pos_text)

        neg_text = []
        for j in range(len(data[i]['negative']['comments'])):
            neg_text.append(data[i]['negative']['comments'][j]['body'])
        neg_text = "\n".join(neg_text)
        neg_text = tokenise_clean(neg_text)
        
        if pid in irrs:
            irr_text = irrs[pid]['irr_text']
            irr_ir = irrs[pid]['irr_id']
        else:
            irr_text = [data[random.randint(0,len(data))]['op_title']] # take a rando irrelevant thing
            irr_id = "irr"

        ID_text[pos_id] = pos_text
        ID_text[neg_id] = neg_text
        ID_text[irr_id] = irr_text

        ID_pairs.append((pid, pos_id, 1))        
        ID_pairs.append((pid, neg_id, 0))
        ID_pairs.append((pid, irr_id, -1))

        thread['ID_pairs'] = ID_pairs
        thread['ID_text'] = ID_text
        thread['title'] = data[i]['op_title']
        thread['thread_ID'] = clean_id(data[i]['op_name'])

        if pid not in title_embed:
            title = data[i]['op_title']
            title = title.lower().replace("cmv:", "")
            #title_embed[pid] = session.run(output, feed_dict={messages: [title]}).tolist()

        all_threads.append(thread)

    # model split here
    if fn=="train":
        train_threads, valid_threads= train_test_split(all_threads,
                test_size=0.1, random_state=0, shuffle=False)

        json_f = "data/pair_task/{}_allthread_pairs.jsonlist".format(fn)
        with open(json_f, 'w', encoding='utf-8') as f:
            json.dump(train_threads, f)

        json_f = "data/pair_task/valid_allthread_pairs.jsonlist"
        with open(json_f, 'w', encoding='utf-8') as f:
            json.dump(valid_threads, f)

    else:
        json_f = "data/pair_task/{}_allthread_pairs.jsonlist".format(fn)
        with open(json_f, 'w', encoding='utf-8') as f:
            json.dump(all_threads, f)

    title_f = "data/pair_task/{}_title_embed.json".format(fn)
    with open(title_f, 'w') as f:
        json.dump(title_embed, f)




def tokenise_clean(text):
    # convert text body (string) to list of sentences (string)
    footnotestrip = '\n&gt; *Hello'
    if text.find(footnotestrip)==-1:
        pass
    else:
        text = text[:text.find(footnotestrip)]

    text = re.sub(r'http\S+', '', text)
    # paragraph vs sent level
    #text = [s for s in text.splitlines() if len(s)>3]
    text = [s for s in sent_tokenize(text) if len(s)>3]
    text = [s for s in text if not s.lstrip().startswith("&gt;") and not
            s.lstrip().startswith("____")] #and "edit" not in " ".join(s.lower().split()[:2])]
    text = [s.strip().replace("\n", '') for s in text]
    text = [s for s in text if len(s)>0]
    

    # stupid way of splitting
    if len(text)==1:
        text = text[0].split('. ')

    text = [clean_str(s) for s in text if len(s)>0]

    return text

def get_data_labels(cmu_fn="", json_fn="", mode="full"):

    data = []
    with open(json_fn, encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))

    print("len data:", len(data))
    added_delta = set()
    with open(cmu_fn, 'r') as f:
        cmu_ids = f.readlines()

    cmu_ids = [clean_id(d.strip()) for d in cmu_ids]

    if mode=="full":
        # get deltas by this pattern
        labels = []

        delta_u = b'\xe2\x88\x86'
        delta_ids = set()
    #    delta_us = "\u0394 &amp;#8710; !delta ∆".split()

        for i in range(len(data)):
            main_author = data[i]['author']
            for comment in data[i]['comments']:

                if 'body' not in comment.keys():
                    continue

                parent_id = clean_id(comment['parent_id'])
                if parent_id not in cmu_ids or comment['author'] != main_author:
                    continue

             #   for delta in delta_us:
             #       print(delta)
                if delta_u in comment['body'].encode('utf-8') and comment['author']==main_author:
               # if delta in comment['body']:
                    labels.append([parent_id, 1])
                    delta_ids.add(comment['id'])
                    break
        
        for id in cmu_ids:
            if id not in delta_ids:
                labels.append([id, 0])

        labels = pd.DataFrame(labels, columns=['id', 'delta'])

    elif mode=="cmu":
        labels = pd.read_csv('data/all/labels.csv', header=None, names=['id', 'delta'])
        labels['id'] = labels['id'].apply(lambda x: clean_id(x))

    print(labels['delta'].value_counts())
    return cmu_ids, data, labels


def get_delta_comments(cmu_fn="", json_fn="", save_fn=""):
    #cmu_ids = [clean_id(d.strip()) for d in cmu_ids]
    cmu_ids = [d.strip() for d in cmu_ids]
    delta_u = b'\xe2\x88\x86'
    delta_us = "\u0394 &amp;#8710; !delta ∆".split()
    delta_types = {}
    for delta in delta_us:
        delta_types[delta]=0

    delta_comments = []
    cmu_ids, data, labels = get_data_labels(cmu_fn, json_fn, mode="cmu")

    for i in range(len(data)):

        title = data[i]['title']
        main_author = data[i]['author']
        thread_id = data[i]['id']

        for comment in data[i]['comments']:
            comment_id = clean_id(comment['id'])
            parent_id = clean_id(comment['parent_id'])

            if parent_id not in cmu_ids:
                continue

            delta = int(labels[labels['id']==parent_id]['delta'].values[0])
            if 'body' not in comment.keys():
                continue

            #if delta_u in comment['body'].encode('utf-8') and comment['author']==main_author:
            for delta_u in delta_us:
                if delta_u in comment['body'] and comment['author']==main_author:
                    delta_types[delta_u]+=1
            
                    delta_comments.append([thread_id, parent_id, comment_id, title,
                        comment['body']])
                    break
        
    print(delta_types)
    data = pd.DataFrame(delta_comments, columns=['thread_id', 'parent_id',
    'comment_id', 'title', 'delta_comment'])

    data.to_csv(save_fn, sep=";")

def process_cmu_ids2(cmu_fn="", json_fn="", save_fn=""):

    # mode="cmu" relies on Yohan Jo's labels.csv file
    # mode="full" is our own attempt to get labels. deprecated.
    #
    #cmu_ids, data, labels = get_data_labels(cmu_fn, json_fn, mode="full")
    cmu_ids, data, labels = get_data_labels(cmu_fn, json_fn, mode="cmu")

    # following Yohan Jo's recommendation from email chain
    delta_us = "\u0394", "&amp;#8710;", "!delta", "∆" 

    ID_pairs = []
    delta_pos = 0
    delta_neg = 0
    delta_irr = 0
    all_threads = []
    delta_comments = []

    all_threads=[]
    used_ids = set()
    thread_ids = set()
    pos = 0
    unlabelled = 0
    neg = 0

    for i in range(len(data)):

        if i%100==0:
            print("done:", i/len(data)) 
        thread = {} 
        thread['title'] = data[i]['title'].lower().replace("cmv:", "")
        main_author = data[i]['author']

        thread_id = clean_id(data[i]['id'])
        thread['thread_ID'] = thread_id
        ID_pairs=[]
        ID_text={}

        main_author = data[i]['author']
        ID_pairs = []
        ID_text = {}
        # first store all the delta_IDs
        delta_IDs = []
        for comment in data[i]['comments']:
            parent_id = clean_id(comment['parent_id'])
            if parent_id not in cmu_ids:
                continue

            if 'body' not in comment.keys():
                print("parent_id in cmu id but no content")
                pdb.set_trace()
                continue

            #if delta_u in comment['body'].encode('utf-8') and comment['author']==main_author:
            for delta in delta_us:
                if delta in comment['body'] and comment['author']==main_author:
                    ID_pairs.append((thread_id, parent_id, 1))
                    delta_IDs.append(parent_id)
                    used_ids.add(parent_id)
                    delta_comments.append(comment['body'])
                

        for comment in data[i]['comments']:
            cid = clean_id(comment['id'])
            if 'body' not in comment.keys() or comment['author']=="DeltaBot" or comment['author']==main_author:
                #if cid in labels['id'].values:
                #    pdb.set_trace()
                continue

            if cid not in cmu_ids:
                if thread_id in thread_ids:
                    delta=2
                else:
                    continue

            else:
                if cid in labels['id'].values:
                    delta = int(labels[labels['id']==cid]['delta'].values[0])
                    thread_ids.add(thread_id)
                else:
                    if thread_id in thread_ids:
                        delta=2
                    else:
                        continue
                    #delta = 2 #unlabelled
                
            if delta==1:
                pos+=1
            if delta==2:
                unlabelled+=1
            if delta==0:
                neg +=1

            ID_pairs.append((thread_id, cid, delta))
            used_ids.add(cid)

            text = comment['body'].strip()
            text = strip_quote_reply(text)
            text = tokenise_clean(text) # convert to phrases/split by \n\n
            text = [x for x in text if len(x.split())>5]

            ID_text[cid] = text

        
        if len(ID_pairs)==0:
            continue

        thread['ID_pairs'] = ID_pairs
        ID_text[thread_id] = tokenise_clean(data[i]['selftext'])
        thread['ID_text'] = ID_text
        all_threads.append(thread)

    # SAVE
    print(cmu_fn, '\n', json_fn, '\n',  save_fn)
    print("missing ids:", len(set(cmu_ids).difference(used_ids)))
    print("total:", len(used_ids), "pos delta:", pos, 'unlabeled:', unlabelled, 'neg:', neg)
    print("save to:", save_fn)
    with open(save_fn, 'w', encoding='utf-8') as f:
        json.dump(all_threads, f)


def strip_quote_reply(text):

    if "&gt;" in text:
        text_ = text.split("&gt;")
        text_ = [t for t in text_ if len(t.strip())>0]
        text_ = [t.split("\n\n", 1) for t in text_]
        replies = []

        for t in text_:
            try:
                #quote_reply.append({'quote':t[0].strip(), 'reply':t[1].strip()})
                replies.append(t[1])
            except:
                pass

        #comment['quote_reply'] = quote_reply
        if len(replies)>0:
            text = " ".join(replies) #make sure we dont count quotes
    return text



def process_json_all(fn="", filter_deltas=True):

    def drop_comment(comment, author_daward):
            # Ignore conditions
            # no author, no text
            # ignore delta bot
 
        if "author" not in comment.keys():
            return True
        if comment['author']=="DeltaBot":
            return True

        if "body" not in comment.keys():
            return True

        if len(comment['body'])==0:
            return True

        # remove replies to author after already awarding delta
        par_id = clean_id(comment['parent_id'])
        if par_id in author_daward:
            return True

        return False


    json_f = "data/all/{}_period_data.jsonlist".format(fn)

    data = []
    with open(json_f, encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))


    for i in range(len(data)):

        data[i]['selftext'] = tokenise_clean(data[i]['selftext'])
        main_author = data[i]['author']

        main_author_ids = []
        latest_post = 0

        delta_awarded = []
        author_daward = []
        for comment in data[i]['comments']:
            comment['keep'] = True

            if drop_comment(comment, author_daward):
                comment['keep']=False
                continue

            else:
                if comment['author'] == main_author:
                    main_author_ids.append(clean_id(comment['id']))
                    if comment['created'] > latest_post:
                        latest_post = comment['created']


            # if they don't quote do we assume they are replying to everything 
            # if delta in main body, the parent has been awarded a delta.
            
            if b'\xe2\x88\x86' in comment['body'].encode('utf-8') and comment['author']==main_author:
                # delta awarded
                delta_awarded.append(clean_id(comment['parent_id']))
                comment['keep']=False
                author_daward.append(clean_id(comment['id']))
                continue

            else:
                text = comment['body'].strip()
                text = strip_quote_reply(text)
                text = tokenise_clean(text)

                comment['text'] = [x for x in text if len(x.split())>0]

            if len(comment['text'])==0:
                comment['keep']=False

        data[i]['comments'] = [f for f in data[i]['comments'] if f['keep']]

        # get irrelevant
        irrelevant = []
        for comment in data[i]['comments']:
            # Consider all comments before latest author post and by COs
            # irrelevant - nobody replied
            # irrelevant - someone replied but main author never replied
            if comment['created']<latest_post and comment['author']!=main_author:

                if comment['replies'] is None and len(comment['body'])>0:
                    irrelevant.append(comment['id'])

                elif len(comment['replies'])==0 and "body" in comment.keys():
                    irrelevant.append(comment['id'])
                else:
                    child_ids = comment['replies']['data']['children']
                    child_ids = [clean_id(id) for id in child_ids]
                    author_replied = set(child_ids).intersection(set(main_author_ids))
                    if len(author_replied)==0:
                        irrelevant.append(clean_id(comment['id']))
        #########################
 

        # Step through all the comments, award delta to the right person
        for comment in data[i]['comments']:
            if clean_id(comment['name']) in delta_awarded:
                comment['delta'] = 1
            else:
                if clean_id(comment['id']) in irrelevant:
                    comment['delta'] = -1
                else:
                    comment['delta'] = 0


    data = remove_keys(data)
    data = prep_for_vae(data, fn)

    return data

def prep_for_vae(data, fn):

    all_threads = []
    #session, messages, output = prep_tf_model()
    print("total:", len(data))
    title_embed = {}

    for dp in data:
        thread = {}
        ID_pairs = []
        ID_text = {}

        main_author = dp['author']
        ID_text[clean_id(dp['id'])] = dp['selftext']

        for comment in dp['comments']:
            ID_text[clean_id(comment['id'])] = comment['text']

            if comment['author'] == main_author:
                continue

            #ID_pairs.append((pid, comment['id'], comment['delta']))
            # simplify and just say that the first element is the OP main post
            ID_pairs.append((clean_id(dp['id']), clean_id(comment['id']), comment['delta']))

        thread['ID_pairs'] = ID_pairs
        thread['ID_text'] = ID_text
        thread['title'] = dp['title']
        thread['thread_ID'] = clean_id(dp['id']) #dp['name'][dp['name'].find('_')+1:]

        #if pid not in title_embed:
        if clean_id(dp['id']) not in title_embed:
            title = dp['title']
            title = title.lower().replace("cmv:", "")
    #        title_embed[pid] = session.run(output, feed_dict={messages: [title]}).tolist()

        all_threads.append(thread)
    
    if fn=="train":
        train_threads, valid_threads= train_test_split(all_threads,
                test_size=0.1, random_state=0, shuffle=False)

        json_f = "data/all/{}_allthread_pairs.jsonlist".format(fn)
        with open(json_f, 'w', encoding='utf-8') as f:
            json.dump(train_threads, f)

        json_f = "data/all/valid_allthread_pairs.jsonlist"
        with open(json_f, 'w', encoding='utf-8') as f:
            json.dump(valid_threads, f)

    else:
        json_f = "data/all/{}_allthread_pairs.jsonlist".format(fn)
        with open(json_f, 'w', encoding='utf-8') as f:
            json.dump(all_threads, f)


    title_f = "data/all/{}_title_embed.json".format(fn)
    with open(title_f, 'w') as f:
        json.dump(title_embed, f)

    return all_threads


def prep_universal_sent_encoder():

    directory="../../packages/InferSent"
    from models import InferSent
    model_version = 1
    MODEL_PATH = "encoder/infersent%s.pkl" % model_version
    params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                            'pool_type': 'max', 'dpout_model': 0.0, 'version': model_version}
    model = InferSent(params_model)
    model.load_state_dict(torch.load(MODEL_PATH))
    model = model.cuda() if use_cuda else model
    W2V_PATH = 'GloVe/glove.840B.300d.txt'
    model.set_w2v_path(W2V_PATH)
    model.build_vocab_k_words(K=100000)

    # sentences
    embeddings = model.encode(sentences, bsize=128, tokenize=False, verbose=True)
    



    


def prep_tf_model():

    module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
    model = hub.Module(module_url)

    tf.logging.set_verbosity(tf.logging.ERROR)

   # session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
   #     log_device_placement=True))
    session = tf.Session()
    session.run([tf.global_variables_initializer(), tf.tables_initializer()])
    messages = tf.placeholder(dtype=tf.string, shape=[None])
    output = model(messages)

    return session, messages, output

def remove_keys(data):
    remove_comments_k = ["subreddit_id", "banned_by", "removal_reason", "link_id", "likes",
            "replies", "user_reports", "saved", "gilded", "archived", "report_reasons",
            "score", "approved_by", "controversiality", "edited", "author_flair_css_class",
            "downs", "body_html", "subreddit", "score_hidden", "author_flair_text",
            "created_utc", "distinguished", "mod_reports", "num_reports","ups", "body"]

    remove_main_k = ["domain", "banned_by", "media_embed", "subreddit", "selftext_html",
    "likes", "suggested_sort", "user_reports", "secure_media", "link_flair_text",
    "from_kind", "gilded", "archived", "clicked", "report_reasons", "media", "score",
    "approved_by", "over_18", "hidden", "num_comments", "thumbnail", "subreddit_id",
    "hide_score", "edited", "link_flair_css_class", "author_flair_css_class", "downs",
    "secure_media_embed", "saved", "removal_reason", "stickied", "from", "is_self", "from_id",
    "permalink", "url", "author_flair_text", "quarantine", "created", "distinguished",
    "mod_reports", "visited", "num_reports", "ups"]

    for i in range(len(data)):
        for mk in remove_main_k:
            data[i].pop(mk, None)

        for j in range(len(data[i]['comments'])):
            for ck in remove_comments_k:
                data[i]['comments'][j].pop(ck, None)

    return data


# adapted from yoonkim
def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r'\d+', 'N', string) # replace digit with N
    return string.strip() if TREC else string.strip().lower()

def clean_id(pid):
    if "_" in pid:
        pid = pid[pid.find('_')+1:]
    return pid

if __name__=="__main__":
    init()
