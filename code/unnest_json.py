#!/usr/bin/python3
# Author: Suzanna Sia

### Standard imports
#import random
#import numpy as np
#import pdb
#import math
import os
#import sys
#import argparse

### Third Party imports
import pandas as pd
import json
from pandas.io.json import json_normalize

### Local/Custom imports

#from distutils.util import str2bool
#argparser = argparser.ArgumentParser()
#argparser.add_argument('--x', type=float, default=0)

def debates_to_csv():
    json_dir = "data/IQ2_corpus/json"
    csv_dir = "data/IQ2_corpus/csv"
    stats_dir = "data/IQ2_corpus/stats"
    
    if not os.path.exists(csv_dir):
        os.mkdir(csv_dir)
        os.mkdir(stats_dir)

    for fn in os.listdir(json_dir):
        json_fn = os.path.join(json_dir, fn)
        with open(json_fn, 'r') as f:
            data = json.load(f)

    #    meta_df = json_normalize(data).drop(columns='content')
    #    stats_fn = fn[:fn.rfind('.')]+".meta"
    #    meta_df.to_csv(os.path.join(stats_dir, stats_fn))

        turn_df = json_normalize(data['content'])
        subcontents = turn_df['subContent']
        turn_IDs = turn_df['turnID'].values

        all_sentences = []
        for i in range(len(subcontents)):
            turnID = turn_IDs[i]
            sentences_df = json_normalize(subcontents[i])
            sentences_df['turnID'] = turnID
            all_sentences.append(sentences_df)

        all_sentences_df = pd.concat(all_sentences, axis=0)
        merge_df = turn_df.set_index('turnID').join(all_sentences_df.set_index('turnID'))
        merge_df = merge_df.drop(columns='subContent')

        csv_fn = fn[:fn.rfind('.')]+".csv"
        print("writing file:", os.path.join(csv_dir, csv_fn))
        merge_df.to_csv(os.path.join(csv_dir, csv_fn))



def cmv_to_csv(fn):
    # {test_cd, test_id, train, valid}-allthread_pairs.jsonlist
    with open(fn, 'r') as f:
        data = json.load(f)

    lines = []
    for d in data:
        title = d['title']
        tid = d['thread_ID']
        tpair = d['ID_pairs']
      
        for k in d['ID_text'].keys():
            text = d['ID_text'][k]
            delta=0
            for tp in tpair:
                if tp[1]==k:
                    delta = tp[2]

            lines.append((title, tid, k, delta, ". ".join(text)))

    csv_fn = fn[:fn.rfind('.')]+".csv"

    df = pd.DataFrame(lines, columns='title mainpost_ID comment_ID delta text'.split())
    df.to_csv(csv_fn)

if __name__ == "__main__":
    debates_to_csv()
