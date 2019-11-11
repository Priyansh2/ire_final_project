#!/bin/bash
if [ $1 == "ngram" ] ;
then
python ngram.py -g 0.5 -d 0.75 $2 $3

elif [ $1 == "lstm" ];
then
python lstm.py -h 1 -n 500 -d 0.1 -e 100 -l 0.001 -b 128 $2 $3
fi;

