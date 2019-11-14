#!/bin/bash
if [ $1 == "ngram" ] ;
then
python ../scripts/ngram_predict_lm.py -f $2 -l $3  -m $4 -ksn 0 -i "${@:5}"

elif [ $1 == "lstm" ];
then
python ../scripts/lstm_test.py -f $2 -l $3 -m $4 -i "${@:5}"
fi;
