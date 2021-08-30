#!/usr/bin/env bash
# Author: Suzanna Sia

source ~/bin/utils.sh

i=$1 # random seed

FILD=data/IQ2_corpus
DELTAFD=$FILD/json_use
NDELTAFD=$FILD/json_rest

rm_mk $DELTAFD.tmp
rm_mk $FILD/train
rm_mk $FILD/test
rm_mk $FILD/valid

cp -r $DELTAFD/* $DELTAFD.tmp

ls $DELTAFD.tmp/* | shuf -n 5 --random-source=<(get_seeded_random $i) | xargs -n1 -I % mv % $FILD/test
ls $DELTAFD.tmp/* | shuf -n 10 --random-source=<(get_seeded_random $i) | xargs -n1 -I % mv % $FILD/valid

# uncomment this if we need some semi-supervised approach
# cp -r $NDELTAFD/* $FILD/train
cp -r $DELTAFD.tmp/* $FILD/train

ntrain=`ls $FILD/train/* | wc -l`
nvalid=`ls $FILD/valid/* | wc -l`
ntest=`ls $FILD/test/* | wc -l`

echo "seed: $i ntrain:$ntrain, nvalid:$nvalid, ntest:$ntest"
