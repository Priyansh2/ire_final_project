#!/bin/bash
if [ $1 == "ngram" ] ;
then
python ngram.py -g 0.5 -d 0.75 $2 $3

elif [ $1 == "lstm" ];
then
python lstm.py -hl 1 -n 500 -dr 0.1 -e 100 -lr 0.001 -b 128 -i $2 -o $3 -w "no"
fi;

