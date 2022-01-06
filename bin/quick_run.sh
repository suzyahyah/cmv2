#!/usr/bin/env bash
# Author: Suzanna Sia

DS=all
enc=glove
fw=rnnv
ss=0

################
hyp=4 #{change this to 1,2,3,4,5}
seed=1
###############
sanity_check=1
weighted_triploss=1
trip_thresh=0.1

bash ./bin/exp_settings.sh $DS $enc $fw $hyp $ss $seed $sanity_check $weighted_triploss $trip_thresh
