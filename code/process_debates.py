#!/usr/bin/python3
# Author: Suzanna Sia

### Standard imports
#import random
import numpy as np
#import pdb
#import math
import os
import shutil
import sys
import pdb
import json
#import argparse

### Third Party imports
from collections import Counter
from scipy.spatial import distance
#from nltk.tokenize import sent_tokenize

### Local/Custom imports
#from debugger import Debugger
#DB = Debugger()
#DB.debug_mode=True

#from distutils.util import str2bool
#argparser = argparser.ArgumentParser()
#argparser.add_argument('--x', type=float, default=0)

def get_win_team(thread):

    prev = list(map(lambda x: int(x), thread['preVote'].split(',')))
    postv = list(map(lambda x: int(x), thread['postVote'].split(',')))

    pro_gains = postv[0] - prev[0]
    con_gains = postv[1] - prev[1]

    if pro_gains > con_gains:
        return "pro"
    else:
        return "con"

def get_delta_ixs(all_data, nstd=1):

    print("number of debates:", len(all_data))

    jsds = [calc_JSD_change(data) for data in all_data]
    mean_jsd = np.mean(jsds)
    std_jsd = np.std(jsds)
    print("mean:", mean_jsd)

    deltas = np.where(jsds>(mean_jsd + nstd*std_jsd))[0]
    no_deltas = np.where(jsds<(mean_jsd + nstd*std_jsd))[0]
    
    
    return (deltas, no_deltas)

def calc_JSD_change(data):
    prevv = np.array(list(map(lambda x: int(x), data['preVote'].split(','))))
    postv = np.array(list(map(lambda x: int(x), data['postVote'].split(','))))

    prevv = prevv/sum(prevv)
    postv = postv/sum(postv)

    jsd = distance.jensenshannon(prevv, postv, 2)

    DB.dp()

    return jsd

def change_view(data):
    prev = get_vote(data, time="pre")
    postv = get_vote(data, time="post")
    delta = ""
    if prev == postv:
        delta = 0
    else:
        delta = 1

    DB.dp({'prev':prev, 'postv':postv, 'delta':delta})

    return delta



def get_vote(data, time="pre"):
    if time == "pre":
        vote = "preVote"
    else:
        vote = "postVote"

    pos = np.argmax(np.array(list(map(lambda x: int(x), data[vote].split(',')))))
    belief = ""

    if pos == 0:
        belief = "pro"
    if pos == 1:
        belief = "con"
    if pos == 2:
        belief = "neutral"

    return belief

def get_summary_turns(data):
    next_turns = []
    for content in data['content']:
        if content['role'] == "mod":
            turnID = content['turnID']
            if len(content['subContent'])>0:
                for subcontent in content['subContent']:
                    sc = subcontent['sentenceContent']
                    # some text processing goingon here
                    if "motion" in sc and ("summ" in sc or "closing" in sc):
                        next_turns.append(turnID + 1)
    DB.dp({'next_turns':next_turns})
    return next_turns

def get_speaker(data, role):
    all_content = []
    for content in data['content']:
        if content['role'] == role:
            text = []
            for subcontent in content['subContent']:
                sc = subcontent['sentenceContent']
                # add substantial content
                if len(sc.split())>10:
                    text.append(sc)
            if len(text)>0:
                all_content.append(" ".join(text))
    return all_content

def summarise(data):
    debate = {}
    debate['pro'] = []
    debate['con'] = []

    next_turns = get_summary_turns(data)

    for content in data['content']:
        turnID = content['turnID']
        role = content['role']
        
        if turnID in next_turns:

            if role not in ['pro', 'con']:
                next_turns.append(turnID +1)
                continue
            
            statements = []
            for subcontent in content['subContent']:
                statements.append(subcontent['sentenceContent'])

            #statements = " ".join(statements)
            debate[role].append(statements)
    DB.dp()
    return debate

if __name__ == "__main__":

    fd = 'data/IQ2_corpus/json'
    fns = os.listdir(fd)

    all_threads = []
    for fn in fns:
        fn = os.path.join(fd, fn)
        with open(fn, 'r') as f:
            thread = json.load(f)
        all_threads.append(thread)

    delta_ix, nodelta_ix = get_delta_ixs(all_threads, nstd=0.25)
    
    fdnew = fd+"_use"
    fdrest = fd+"_rest"
    shutil.rmtree(fdnew)
    shutil.rmtree(fdrest)
    os.mkdir(fdnew)
    os.mkdir(fdrest)

    labels = []

    for i, fn in enumerate(fns):
        if (i in delta_ix):
            with open(os.path.join(fdnew, fn), 'w') as f:
                json.dump(all_threads[i], f)

        elif (i in nodelta_ix):
            with open(os.path.join(fdrest, fn), 'w') as f:
                json.dump(all_threads[i], f)

        if (i in delta_ix):
            win_team = get_win_team(all_threads[i])
            labels.append(f"{fn}\t1\t{win_team}")

        elif (i in nodelta_ix):
            labels.append(f"{fn}\t0\tna")
    
    assert (len(delta_ix)+len(nodelta_ix)) == (len(labels))

    labels = "\n".join(labels)

    print("deltas:", len(delta_ix), "no deltas:", len(nodelta_ix))
    print("writing labels to file:", 'data/IQ2_corpus/labels.txt')

    with open('data/IQ2_corpus/labels.txt', 'w') as f:
        f.write(labels+"\n")


