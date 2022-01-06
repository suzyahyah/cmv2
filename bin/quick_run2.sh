#!/usr/bin/env bash
# Author: Suzanna Sia

SEED=0

for HYP in 0 1 2 3 4 5; do
  qsub -N c.$HYP -o logs/qsub -e logs/qsub_e ./bin/exp_settings.sh $HYP $SEED
done

