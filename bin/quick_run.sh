#!/usr/bin/env bash
# Author: Suzanna Sia

DS=all
enc=glove
fw=rnnv
ss=0

################
hyp=0 #{change this to 1,2,3,4,5}
seed=0
###############
sanity_check=1

bash ./bin/exp_settings.sh $DS $enc $fw $hyp $ss $seed $sanity_check
