#!/bin/bash
set -e

### PARAMETERS ##################################
CORPUS=$1
# out directory
OUT_DIR=${2:-"data"}
# vocab parameters
VOCAB_MIN_COUNT=${3:-"100"}
# cooccurrence parameters
WINDOW_SIZE=${4:-"10"}
DISTANCE_WEIGHTING=${5:-"1"}
# GloVe parameters
VECTOR_SIZE=${6:-"300"}
ETA=${7:-"0.05"}
MAX_ITER=${8:-"100"}
MODEL=${9:-"2"} # 1: W, 2: W+C
SEED=${10:-"1"}
# other parameters
BUILD_DIR=GloVe/build  # GloVe binaries are here
NUM_THREADS=8
MEMORY=6.0
VERBOSE=2
#################################################

# Concat parameters
CORPUS_NAME=${CORPUS##*/}
CORPUS_NAME="${CORPUS_NAME%.*}"
VOCAB_PARAMS=V$VOCAB_MIN_COUNT
COOC_PARAMS=$VOCAB_PARAMS-W$WINDOW_SIZE-D$DISTANCE_WEIGHTING
GLOVE_PARAMS=$COOC_PARAMS-D$VECTOR_SIZE-R$ETA-E$MAX_ITER-M$MODEL-S$SEED

# Files
VOCAB_FILE=$OUT_DIR/vocab-$CORPUS_NAME-$VOCAB_PARAMS.txt
OVERFLOW_FILE=$OUT_DIR/overflow-$CORPUS_NAME-$COOC_PARAMS
COOC_FILE=$OUT_DIR/cooc-$CORPUS_NAME-$COOC_PARAMS.bin
TEMP_FILE=$OUT_DIR/temp-$GLOVE_PARAMS
SHUF_FILE=$OUT_DIR/shuf-$GLOVE_PARAMS.bin
SAVE_FILE=$OUT_DIR/glove-$CORPUS_NAME-$GLOVE_PARAMS


echo Using:
echo CORPUS = $CORPUS
echo VOCAB_MIN_COUNT = $VOCAB_MIN_COUNT
echo WINDOW_SIZE = $WINDOW_SIZE
echo DISTANCE_WEIGHTING = $DISTANCE_WEIGHTING
echo VECTOR_SIZE = $VECTOR_SIZE
echo ETA = $ETA
echo MAX_ITER = $MAX_ITER
echo MODEL = $MODEL
echo SEED = $SEED
echo

# build vocab if not exists
if [[ ! -f $VOCAB_FILE ]]; then
  echo "Building $VOCAB_FILE"
  $BUILD_DIR/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $CORPUS > $VOCAB_FILE
else
  echo "Vocab file: $VOCAB_FILE exists. Skipping."
fi

# build cooc matrix if not exists
if [[ ! -f $COOC_FILE ]]; then
  echo "Building $COOC_FILE"
  $BUILD_DIR/cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE -overflow-file $OVERFLOW_FILE -distance-weighting $DISTANCE_WEIGHTING < $CORPUS > $COOC_FILE
else
  echo "Cooc file: $COOC_FILE exists. Skipping."
fi

# RUN
if [[ ! -f ${SAVE_FILE}.txt ]]; then
  echo "Building ${SAVE_FILE}.txt"
  # Shuffle
  $BUILD_DIR/shuffle -memory $MEMORY -verbose $VERBOSE -temp-file $TEMP_FILE -seed $SEED < $COOC_FILE > $SHUF_FILE
  # Train
  $BUILD_DIR/glove -save-file $SAVE_FILE -threads $NUM_THREADS -input-file $SHUF_FILE -eta $ETA -iter $MAX_ITER -checkpoint-every 0 -vector-size $VECTOR_SIZE -binary 0 -model $MODEL -vocab-file $VOCAB_FILE -verbose $VERBOSE -seed $SEED
  # Clean up
  rm $SHUF_FILE
else
  echo "Embedding: $SAVE_FILE exists. Skipping."
fi

# save as numpy array
if [[ ! -f ${SAVE_FILE}.npy ]]; then
  echo "Saving ${SAVE_FILE} as npy array"
  python3 scripts/glovetxt2array.py ${SAVE_FILE}.txt $VOCAB_FILE
else
  echo "$SAVE_FILE.npy exists. Skipping."
fi

echo
echo "DONE"
echo

# ### NOTE: glove flags:
# 
# -binary 2
# Save output in binary format (0: text, 1: binary, 2: both); default 0
# 
# -model 2
# Model for word vector output (for text output only); default 2
#    0: output all data, for both word and context word vectors, including bias terms
#    1: output word vectors, excluding bias terms
#    2: output word vectors + context word vectors, excluding bias terms
# ###
