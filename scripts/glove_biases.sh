#!/bin/bash

CORPUS=$1

echo "Corpus: $CORPUS"
VOCABFILE="data/working/vocab-$CORPUS-V100.txt" &&
EMBEDFILE="data/working/glove-$CORPUS-V100-W10-D1-D300-R0.05-E100-M2-S1.npy"

echo "Female-male bias"
A="FEMALE" &&
B="MALE" &&
OUTFILE="results/bias_glovewc-$CORPUS-$A-$B.csv" &&
python3 -u scripts/vectors2biasdf.py \
  --vocab $VOCABFILE --matrix $EMBEDFILE --a $A --b $B --out $OUTFILE
