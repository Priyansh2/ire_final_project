#!/bin/bash
if [ $1 == "ngram" ] ;
then
python ngram.py -t twitter --train data/raw_data

elif [ $1 == "lstm" ];
then
python lstm.py -t twitter -h 1 -n 500 -d 0.1 -e 100 --train data/raw_data
fi;
