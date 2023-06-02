
# A PMI-based approach to measure biases in texts

Code to replicate _On the Interpretability and Significance of Bias Metrics in Texts: a PMI-based Approach_ (Valentini et al., ACL 2023).

Cite as:

```bibtex
@inproceedings{valentini-etal-2023-pmi,
    title = "{O}n the {I}nterpretability and {S}ignificance of {B}ias {M}etrics in {T}exts: a {PMI}-based {A}pproach",
    author = "Valentini, Francisco  and
      Rosati, Germ{\'a}n  and
      Blasi, Dami{\'a}n and
      Fernandez Slezak, Diego  and
      Altszyler, Edgar",
    booktitle = "Proceedings of the 61th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)",
    month = july,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
}
```
<!-- TODO add in bibtex: -->
<!-- url = "...",
doi = "...",
pages = "...", -->

The following guide was run in Ubuntu 18.04.4 LTS with python=3.9.12 and R=4.0.0. You can set up a [conda environment](#conda-environment) but it is not compulsory. 

## Requirements

Install **Python requirements**:

```
python -m pip install -r py_requirements.txt
```

Install **R requirements**:

```
R -e 'install.packages("dplyr", repos="http://cran.us.r-project.org", dependencies=T)' &&
R -e 'install.packages("weights", repos="http://cran.us.r-project.org", dependencies=T)' &&
R -e 'install.packages("devtools", repos="http://cran.us.r-project.org", dependencies=T)' &&
R -e 'devtools::install_github("conjugateprior/cbn")'
```

To build GloVe:

* In Linux: `cd GloVe && make`

* In Windows: `make -C "GloVe"`


## Guide

### Corpora

#### Wikipedia

1. Download 2014 English Wikipedia dump (`enwiki-20141208-pages-articles.xml.bz2`) from https://archive.org/download/enwiki-20141208 into `corpora` dir:

```
URL=https://archive.org/download/enwiki-20141208 &&
FILE=enwiki-20141208-pages-articles.xml.bz2 &&
wget -c -b -P corpora/ $URL/$FILE
# flag "-c": continue getting a partially-downloaded file
# flag "-b": go to background after startup. Output is redirected to wget-log.
```

2. Extract dump into a raw .txt file:

```
chmod +x scripts/extract_wiki_dump.sh &&
scripts/extract_wiki_dump.sh corpora/enwiki-20141208-pages-articles.xml.bz2
```

3. Create text file with one line per sentence and removing articles of less than 50 words:

```
python3 -u scripts/tokenize_and_clean_corpus.py corpora/enwiki-20141208-pages-articles.txt
```

4. Remove non alpha-numeric symbols from sentences, clean whitespaces and convert caps to lower:

```
CORPUS_IN=corpora/enwiki-20141208-pages-articles_sentences.txt &&
CORPUS_OUT=corpora/wiki2014.txt &&
chmod +x scripts/clean_corpus.sh &&
scripts/clean_corpus.sh $CORPUS_IN > $CORPUS_OUT
```

#### OpenSubtitles

1. Download the raw corpus:

```
URL=http://opus.nlpl.eu/download.php?f=OpenSubtitles/v2018/raw &&
FILE=en.zip &&
wget -c -b -P corpora/ $URL/$FILE
```

```
# rename:
mv corpora/en.zip corpora/subs_en.zip
```

2. Strip xml and concatenate multiple files into one with one sentence per line:

```
cd corpora &&
python3 ../scripts/extract_subs.py en --strip --join
```

```
# rename:
mv en_stripped.zip subs_en_stripped.zip
mv en.txt subs_en_raw.txt
```

3. Remove non alpha-numeric symbols from sentences, clean whitespaces and convert caps to lower:

```
CORPUS_IN=corpora/subs_en_raw.txt &&
CORPUS_OUT=corpora/subs.txt &&
scripts/clean_corpus.sh $CORPUS_IN > $CORPUS_OUT
```

### Co-occurrence counts

1. Create vocabulary of corpus using GloVe module:

```
CORPUS=corpora/subs.txt &&
OUT_DIR=data/working &&
VOCAB_MIN_COUNT=100 &&
chmod +x scripts/corpus2vocab.sh &&
scripts/corpus2vocab.sh $CORPUS $OUT_DIR $VOCAB_MIN_COUNT
```

2. Create co-occurrence matrix as binary file using GloVe module:

```
CORPUS=corpora/subs.txt &&
OUT_DIR=data/working &&
VOCAB_MIN_COUNT=100 &&
WINDOW_SIZE=10 &&
DISTANCE_WEIGHTING=0 &&
chmod +x scripts/corpus2cooc.sh &&
scripts/corpus2cooc.sh $CORPUS $OUT_DIR $VOCAB_MIN_COUNT $WINDOW_SIZE $DISTANCE_WEIGHTING
```

3. Transform co-occurrence binary file to `scipy.sparse` format (`.npz` file):

```
VOCABFILE="data/working/vocab-subs-V100.txt" &&
COOCFILE="data/working/cooc-subs-V100-W10-D0.bin" &&
OUTFILE="data/working/cooc-subs-V100-W10-D0.npz" &&
python3 -u scripts/cooc2sparse.py -v $VOCABFILE -c $COOCFILE -o $OUTFILE
```

### PMI-based biases

Compute PMI biases of all words in vocabulary with respect to a given set of groups of words. The groups of words are specified in `words_lists/` as text files. Results are saved as a DataFrame in `.csv` format.

```
CORPUS=subs &&
SMOOTHING=0.5 &&
bash scripts/pmi_biases.sh $CORPUS $SMOOTHING
```

### Skip-gram

Train word embeddings with SGNS using python's `gensim` library. This saves a `.model` with trained model and `.npy` with the embeddings in array format. If the model is large, files with extension `.trainables.syn1neg.npy` and `.wv.vectors.npy` might be saved alongside `.model`.


Example:
```
CORPUS=subs &&
SG=1 && # 0:cbow, 1:sgns
SIZE=300 &&
WINDOW=10 &&
MINCOUNT=100 &&
SEED=1 &&
CORPUSFILE=corpora/$CORPUS.txt &&
VOCABFILE="data/working/vocab-$CORPUS-V100.txt" &&
python3 -u scripts/corpus2sgns.py --corpus $CORPUSFILE --vocab $VOCABFILE \
  --size $SIZE --window $WINDOW --count $MINCOUNT --sg $SG --seed $SEED
```

### GloVe

Train GloVe word embeddings and save them as `.npy` with the vectors in array format. Vectors are also saved in binary format.

Example:
```
CORPUS=corpora/subs.txt &&
OUT_DIR=data/working &&
VOCAB_MIN_COUNT=100  && # words with lower frequency are removed before windows
WINDOW_SIZE=10 &&
DISTANCE_WEIGHTING=1 && # normalized co-occurrence counts (vanilla GloVe)
VECTOR_SIZE=300 &&
ETA=0.05 &&
MAX_ITER=100 &&
MODEL=2 && # 1:W, 2:W+C
SEED=1 &&
scripts/corpus2glove.sh $CORPUS $OUT_DIR $VOCAB_MIN_COUNT \
  $WINDOW_SIZE $DISTANCE_WEIGHTING $VECTOR_SIZE $ETA $MAX_ITER $MODEL $SEED
```

### Embeddings-based biases

Using SGNS and GloVe embeddings, compute cosine-based biases of all words in vocabulary with respect to a given set of groups of words. The groups of words are specified in `words_lists/` as text files. Results are saved as a DataFrame in `.csv` format.

Example:

```
# SGNS:
CORPUS=subs &&
scripts/sgns_biases.sh $CORPUS
# GLOVE:
CORPUS=subs &&
scripts/glove_biases.sh $CORPUS
```

### Results

External data sources are needed to replicate experiments. 

* Download the _names_ dataset into `data/external/cbn_gender_name_stats.csv` using Will Lowe’s [`cbn`](
https://conjugateprior.github.io/cbn) R library with:

```
Rscript scripts/get_names_data.R
```

* Download the _occupations_ and Glasgow Norms datasets with:

```
wget -P data/external/ https://raw.githubusercontent.com/conjugateprior/cbn/master/inst/extdata/professionsBLS2015.tsv  &&
wget -O data/external/GlasgowNorms.csv https://static-content.springer.com/esm/art%3A10.3758%2Fs13428-018-1099-3/MediaObjects/13428_2018_1099_MOESM2_ESM.csv 
```

Run permutations tests with Python's `scipy` and prepare data for figures:

```
python3 scripts/prepare_figures_data.py
```

Add bootstrap results of R's `boot` library:

```
Rscript scripts/run_bootstrap.R
```

To replicate tables and figures execute the notebook `figures_bias.ipynb` with:

```
mkdir -p results/plots &&
conda deactivate &&
jupyter nbconvert \
  --ExecutePreprocessor.kernel_name=bias-pmi \
  --to notebook \
  --execute figures_bias.ipynb
```

Results are saved as `figures_bias.nbconvert.ipynb`.

Print the p-values of Table 1 with:

```
Rscript scripts/compute_correlation_pvalues.R
``` 

## conda environment

You can create a `bias-pmi` conda environment to install requirements and dependencies. This is not compulsory. 

To install miniconda if needed, run:

```
wget https://repo.anaconda.com/miniconda/Miniconda3-py37_4.10.3-Linux-x86_64.sh
sha256sum Miniconda3-py37_4.10.3-Linux-x86_64.sh
bash Miniconda3-py37_4.10.3-Linux-x86_64.sh
# and follow stdout instructions to run commands with `conda`
```

To create a conda env with Python and R:

```
conda config --add channels conda-forge
conda create -n "bias-pmi" --channel=defaults python=3.9.12
conda install --channel=conda-forge r-base=4.0.0
```

Activate the environment with `conda activate bias-pmi` and install pip with `conda install pip`.
