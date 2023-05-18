import numpy as np
import argparse

from pathlib import Path

from utils.vocab import load_vocab


def strings_to_floats(strings):
    """
    List of strings to np.array of floats
    """
    numeros = [float(i) for i in strings]
    return np.array(numeros)


def txt_to_array(glove_file, str2idx):
    """
    Convert data from GloVe txt to np.array of word vectors
    """
    # read
    with open(glove_file, 'r') as f:
        lineas = f.readlines()
    # word2vector dict
    words = [linea.split()[0] for linea in lineas]
    vectors = [strings_to_floats(linea.split()[1:]) for linea in lineas]
    word2vector = dict(zip(words, vectors))
    # save in array
    D = len(vectors[0])
    V = len(str2idx)
    M = np.full((D, V+1), np.nan)
    assert len(set(words) & set(str2idx)) == V, \
        "Vocab words are missing from GloVe txt"
    for w, i in str2idx.items():
        M[:,i] = word2vector[w]
    return M


def main(glove_file, vocab_file, outdir):

    # load vocab dicts
    str2idx, idx2str, str2count = load_vocab(vocab_file)

    # get matrix
    M = txt_to_array(glove_file, str2idx)

    # save matrix
    basename = Path(glove_file).stem
    matrix_file = str(Path(outdir) / "data/working" / f"{basename}.npy")
    np.save(matrix_file, M)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('glove_file', type=str)
    parser.add_argument('vocab_file', type=str)
    parser.add_argument('--outdir', type=str, required=False, default="")

    args = parser.parse_args()

    main(args.glove_file, args.vocab_file, args.outdir)
