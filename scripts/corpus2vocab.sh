#!/bin/bash

CORPUS=$1
# out directory
OUT_DIR=${2:-"data"}
# vocab parameters
VOCAB_MIN_COUNT=${3:-"100"}

# other parameters
BUILD_DIR=GloVe/build  # GloVe binaries are here
NUM_THREADS=8
MEMORY=4.0
VERBOSE=2

# Concat parameters
CORPUS_NAME=${CORPUS##*/}
CORPUS_NAME="${CORPUS_NAME%.*}"
VOCAB_PARAMS=V$VOCAB_MIN_COUNT

# Files
VOCAB_FILE=$OUT_DIR/vocab-$CORPUS_NAME-$VOCAB_PARAMS.txt

echo
echo Using:
echo CORPUS = $CORPUS
echo VOCAB_MIN_COUNT = $VOCAB_MIN_COUNT
echo

# RUN
if [[ ! -f $VOCAB_FILE ]]; then
  echo "Building $VOCAB_FILE"
  $BUILD_DIR/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $CORPUS > $VOCAB_FILE
else
  echo "Vocab file: $VOCAB_FILE exists. Skipping."
fi

echo
echo "DONE"
echo
