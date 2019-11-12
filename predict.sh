#!/bin/bash
if [ $1 == "ngram" ] ;
then
python ngram.py -f $2 -l $3  -m $4 -i "${@:5}"

elif [ $1 == "lstm" ];
then
python lstm.py -f $2 -l $3 -m $4 -i "${@:5}"
fi;
