#!/bin/bash
if [ $1 == "ngram" ] ;
then
python ../scripts/ngram_train_lm.py -i $2 -pkl 1 -o $3 -g 0.5 -d 0.75 -t 4 -n 3

elif [ $1 == "lstm" ];
then
python ../scripts/lstm_train.py -hl 1 -n 500 -dr 0.1 -e 100 -lr 0.001 -b 128 -i $2 -o $3 -w "no"
fi;

