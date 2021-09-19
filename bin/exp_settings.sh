#!/usr/bin/env bash
#$ -l 'hostname=b1[134567]|c*,mem_free=40G,ram_free=40G,gpu=1'
#$ -q g.q
#$ -cwd
#$ -m ea
#$ -M fengsf85@gmail.com

source activate rnn-vae
echo $HOSTNAME
echo $*

DATASET=$1
SAVEDIR=/export/c10/ssia/cmv

if [ "$HOSTNAME" == "DESKTOP-KFANQCM" ]; then
  CUDA=-1
else
  CUDA=`free-gpu`
  echo "free gpu : $CUDA"
  export CUDA_VISIBLE_DEVICE=$CUDA
fi


if [ "$DATASET" == "IQ2" ]; then
  bash ./bin/debates-data-prep2.sh 4
  dataset=${DATASET}_corpus
  glovematrix=data/embeddings/all/glove_matrix.debates
  vocabfn=data/embeddings/all/glove.vocab.debates
  trainfn=data/$dataset/train
  validfn=data/$dataset/valid
  testfn=data/$dataset/test
  testfn2=data/$dataset # not int use
  metric=acc
  hidden_dim=256
  latent_dim=128
  contrast_thresh=0.1
  margin_thresh=0 #balanced dataset so no need for margin_thresh
  
elif [ "$DATASET" == "all" ]; then
  glovematrix=data/embeddings/all/glove_matrix
  vocabfn=data/embeddings/all/glove.vocab
  trainfn=data/$DATASET/train-allthread_pairs.jsonlist #.filter
  validfn=data/$DATASET/valid-allthread_pairs.jsonlist
  testfn=data/$DATASET/test_id-allthread_pairs.jsonlist
  testfn2=data/$DATASET/test_cd-allthread_pairs.jsonlist
  metric=roc
  hidden_dim=256
  latent_dim=128
  contrast_thresh=0
  margin_thresh=0.5
fi

# triple can only take pair_task
declare -A F=(
["DATASET"]=$DATASET
["OLD_EMB"]=data/embeddings/all/glove_w2vec.txt
["NEW_EMB"]=$glovematrix
["VOCAB_FN"]=$vocabfn
["TRAIN_FN"]=$trainfn
["VALID_FN"]=$validfn
["TEST_FN"]=$testfn
["TEST_FN2"]=$testfn2
["SAVEDIR"]=$SAVEDIR
["TITLE_EMBED_FN"]=data/pair_task/train_title_embed.json
)

declare -A M=(
["ENCODER"]=$2 #, universal_se, glove
["FRAMEWORK"]=${3:-rnnv} #rnn # rnn #bert
["CONFIGN"]=0
["SS_RECON_LOSS"]=$5 # semi-supervised reconstruction loss
["SEED"]=0
["CUDA"]=$CUDA
["NUM_EPOCHS"]=300
["NWORDS"]=40000
["L_EPOCH"]=0
["RNNGATE"]=lstm
["HIDDEN_DIM"]=$hidden_dim
["LATENT_DIM"]=$latent_dim
["BATCH_SIZE"]=1
["N_LAYERS"]=2
["MAX_SEQ_LEN"]=100
["STOPWORDS"]=0 # 1=content<unk>, 2=stopwords<unk> # if true use stopwords only
["EVAL_METRIC"]=$metric
["BALANCED"]=1
)

declare -A Z=(
["ZSUM"]=rnn #cnn, fnn, #weighted_avg, simple_average, similarity
["TRIPLET_THRESH"]=${7:-0.01}
["CONTRAST_THRESH"]=$contrast_thresh
["MARGIN_THRESH"]=$margin_thresh
["USE_PRIOR_MU"]=False
["HYP"]=$4
["UNIVERSAL_EMBED"]=False # change this to encoder
["Z_COMBINE"]=concat
["SCALE_PZVAR"]=1
["WORD_DROPOUT"]=0.4
["UPDATE_ITR"]=10
)

PY="python code/main.py"

echo "===Model Params==="
for var in "${!M[@]}"
do
  PY+=" --${var,,} ${M[$var]}"
  echo "| $var:${M[$var]}"
done

echo "===File Names==="
for var in "${!F[@]}"
do
  PY+=" --${var,,} ${F[$var]}"
  echo "| $var:${F[$var]}"
done

echo "===ELBO tweaks==="
for var in "${!Z[@]}"
do
  PY+=" --${var,,} ${Z[$var]}"
  echo "| $var:${Z[$var]}"
done

#echo $PWD
echo $PY

eval $PY 

subject="${@}"
echo $subject

#cat logs/temp | mutt -s "$subject" fengsf85@gmail.com
