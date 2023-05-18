#!/bin/bash

CORPUS=$1

# out directory
OUT_DIR=${2:-"data"}

# vocab parameters
VOCAB_MIN_COUNT=${3:-"100"}

# cooccurrence parameters
WINDOW_SIZE=${4:-"10"}
DISTANCE_WEIGHTING=${5:-"0"}

# other parameters
BUILD_DIR=GloVe/build  # GloVe binaries are here
NUM_THREADS=8
MEMORY=4.0
VERBOSE=2

# Concat parameters
CORPUS_NAME=${CORPUS##*/}
CORPUS_NAME="${CORPUS_NAME%.*}"
VOCAB_PARAMS=V$VOCAB_MIN_COUNT
COOC_PARAMS=$VOCAB_PARAMS-W$WINDOW_SIZE-D$DISTANCE_WEIGHTING

# Files
VOCAB_FILE=$OUT_DIR/vocab-$CORPUS_NAME-$VOCAB_PARAMS.txt
OVERFLOW_FILE=$OUT_DIR/overflow-$CORPUS_NAME-$COOC_PARAMS
COOC_FILE=$OUT_DIR/cooc-$CORPUS_NAME-$COOC_PARAMS.bin

echo Using:
echo CORPUS = $CORPUS
echo VOCAB_MIN_COUNT = $VOCAB_MIN_COUNT
echo WINDOW_SIZE = $WINDOW_SIZE
echo DISTANCE_WEIGHTING = $DISTANCE_WEIGHTING
echo

# build vocab if not exists
if [[ ! -f $VOCAB_FILE ]]; then
  echo "Building $VOCAB_FILE"
  $BUILD_DIR/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $CORPUS > $VOCAB_FILE
else
  echo "Vocab file: $VOCAB_FILE exists. Skipping."
fi

# RUN
if [[ ! -f $COOC_FILE ]]; then
  echo "Building $COOC_FILE"
  $BUILD_DIR/cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE -overflow-file $OVERFLOW_FILE -distance-weighting $DISTANCE_WEIGHTING < $CORPUS > $COOC_FILE
else
  echo "Cooc file: $COOC_FILE exists. Skipping."
fi

echo
echo "DONE"
echo
