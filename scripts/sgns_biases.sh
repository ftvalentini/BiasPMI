#!/bin/bash

CORPUS=$1

echo "Corpus: $CORPUS"
VOCABFILE="data/working/vocab-$CORPUS-V100.txt" &&
EMBEDFILE="data/working/w2v-$CORPUS-V100-W10-D300-SG1-S1.npy"

echo "Female-male bias"
A="FEMALE" &&
B="MALE" &&
OUTFILE="results/bias_sgns-$CORPUS-$A-$B.csv" &&
python3 -u scripts/vectors2biasdf.py \
  --vocab $VOCABFILE --matrix $EMBEDFILE --a $A --b $B --out $OUTFILE
