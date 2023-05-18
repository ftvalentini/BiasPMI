import numpy as np
import pandas as pd
import argparse
import datetime

from pathlib import Path
from gensim.models.word2vec import Word2Vec

from utils.vocab import load_vocab


class Corpus:
    """
    Helper iterator that yields documents/sentences (doc: list of str)
    Needed so that gensim can read docs from disk on the fly
    """
    def __init__(self, corpus_file):
        """
        corpus_file: a txt with one document per line and tokens separated
        by whitespace
        """
        self.corpus_file = corpus_file
    def __iter__(self):
        for line in open(self.corpus_file, encoding="utf-8"):
            # returns each doc as a list of tokens
            yield line.split()


def train_w2v(corpus_file, size, window, min_count, seed, sg):
    """
    Returns w2v gensim trained model
    Params:
        - min_count: min word frequency
        - sg: 1 if skipgram -- 0 if cbow
    """
    # create generator of lines
    sentences = Corpus(corpus_file)
    # train word2vec
    model = Word2Vec(
        sentences=sentences, vector_size=size, window=window, min_count=min_count
        , seed=seed, sg=sg)
    return model


def w2v_to_array(w2v_model, str2idx):
    """
    Convert w2v vectors to np.array DIMx(V+1), using column indices
    of str2idx of vocab_file produced by GloVe
    """
    vectors = w2v_model.wv
    D = w2v_model.wv.vector_size
    V = len(str2idx)
    M = np.full((D, V+1), np.nan)
    for w, i in str2idx.items():
        M[:,i] = vectors[w]
    return M


def main(corpus_file, vocab_file, outdir="", **kwargs_w2v):
    """
    Train w2v, save w2v model and save embeddings matrix
    """

    print("\nLoading vocab...")
    str2idx, idx2str, str2count = load_vocab(vocab_file)

    # model file name
    kw = kwargs_w2v
    corpus_name = Path(corpus_file).stem
    basename = \
        f"w2v-{corpus_name}-V{kw['min_count']}-W{kw['window']}-D{kw['size']}-SG{kw['sg']}-S{kw['seed']}"
    model_file = str(Path(outdir) / "data/working" / f"{basename}.model")

    if Path(model_file).is_file():
        print(f"Model file {model_file} exists. Skipping training.")
        model = Word2Vec.load(model_file)
    else:
        print("\nTraining vectors with params:", *kwargs_w2v.items(), sep="\n")
        model = train_w2v(corpus_file, **kwargs_w2v)
        print("\nSaving model...")
        model.save(model_file)

    print("\nTesting vocabulary...")
    str2count_w2v = {w: model.wv.get_vecattr(w, "count") for w in model.wv.key_to_index}
    assert str2count_w2v == str2count, \
        "gensim Word2Vec vocab is different from input vocab file"

    print("\nConverting vectors to array...")
    embed_matrix = w2v_to_array(model, str2idx)

    print("\nSaving array...")
    matrix_file = str(Path(outdir) / "data/working" / f"{basename}.npy")
    np.save(matrix_file, embed_matrix)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    required = parser.add_argument_group('required named arguments')
    required.add_argument('--corpus', type=str, required=True)
    required.add_argument('--vocab', type=str, required=True)
    optional = parser.add_argument_group('optional named arguments')
    optional.add_argument('--size', type=int, required=False, default=300)
    optional.add_argument('--window', type=int, required=False, default=10)
    optional.add_argument('--count', type=int, required=False, default=100)
    optional.add_argument('--sg', type=int, required=False, default=1)
    optional.add_argument('--seed', type=int, required=False, default=1)
    optional.add_argument(
        '--outdir', type=str, required=False, nargs='?', const='', default="")

    args = parser.parse_args()
    kwargs_w2v = {
        'size': args.size, 'window': args.window, 'min_count': args.count
        , 'sg': args.sg, 'seed': args.seed
        }

    print("\nSTART -- ", datetime.datetime.now())
    main(args.corpus, args.vocab, outdir=args.outdir, **kwargs_w2v)
    print("\nEND -- ", datetime.datetime.now())
