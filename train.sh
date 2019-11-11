#!/bin/bash
if [ $1 == "ngram" ] ;
then
python ngram.py  $2 $3

elif [ $1 == "lstm" ];
then
python lstm.py -h 1 -n 500 -d 0.1 -e 100 $2 $3
fi;

