### getting IQ2 corpus
```bash
wget http://www.ccis.neu.edu/home/kechenqin/paper/IQ2_corpus.zip -O data/IQ2_corpus.zip
unzip data/IQ2_corpus.zip -d data
mkdir -p data/IQ2_corpus/json_use
mkdir -p data/IQ2_corpus/json_rest
```

### Run the following to process debates 
`>>> python code/process_debates.py`

### You should get the following
json - raw data json format, 119 debates \
json_use - 46 debates that we are using \
json_rest - 73 debates that we are not using \
labels.txt - 119 lines, file_name delta_present winning_team.

This file has `filename delta winning_team` on each line.

1 indicates there is a delta present and we use this debate to predict the winning team, 0 indicates we are not using this debate cos there is insufficient opinion change. 

### Other stuff 
#### helper script for unnesting from json to csv  (modify input file/dir)
`python code/unnest_json.py`

#### generating different train test splits

`bash bin/debates-data-prep2.sh $seed`


---

### running scripts for reddit-cmv

- Change `SAVEDIR` in ./bin/exp_settings.sh
- Change `CUDA=${device}` in ./bin/exp_settings.sh
- In main directory, run `./bin/quick_run.sh`, change hypothesis or seed flag where appropriate.

hidden states will be saved to `SAVEDIR/hidden_states/val_{hyp}_{seed}.p`
