#!/bin/bash

CORPUS=$1
SMOOTHING=$2

echo "Corpus: $CORPUS"
echo "Smoothing = $SMOOTHING"
VOCABFILE="data/working/vocab-$CORPUS-V100.txt"
COOCFILE="data/working/cooc-$CORPUS-V100-W10-D0.npz"

echo "Female-male bias"
A="FEMALE"
B="MALE"
OUTFILE="results/bias_pmi-$CORPUS-$A-$B.csv"
python3 -u scripts/sparse2biasdf.py \
  --vocab $VOCABFILE --cooc $COOCFILE --a $A --b $B --out $OUTFILE --smoothing $SMOOTHING
