import numpy as np
import datetime
import argparse

from utils.vocab import load_vocab
from utils.bias import bias_we_byword


def main(vocab_file, matrix_file, attr_a, attr_b, outfile):

    print("Loading input data...")
    str2idx, idx2str, str2count = load_vocab(vocab_file)
    embed_matrix = np.load(matrix_file)

    print("Getting words lists...")
    words_a = [
        line.strip().lower() for line in open(f'words_lists/{attr_a}.txt','r')]
    words_b = [
        line.strip().lower() for line in open(f'words_lists/{attr_b}.txt','r')]
    words_target = [
        w for w, freq in str2count.items() if w not in words_a + words_b]

    print("Computing WE bias wrt each context word...")
    df_bias = bias_we_byword(
        embed_matrix, words_target, words_a, words_b, str2idx, str2count)

    print("Saving results in csv...")
    df_bias.to_csv(outfile, index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    required = parser.add_argument_group('required named arguments')
    required.add_argument('--vocab', type=str, required=True)
    required.add_argument('--matrix', type=str, required=True)
    required.add_argument('--a', type=str, required=True)
    required.add_argument('--b', type=str, required=True)
    required.add_argument('--out', type=str, required=True)

    args = parser.parse_args()

    print("START -- ", datetime.datetime.now())
    main(args.vocab, args.matrix, args.a, args.b, args.out)
    print("END -- ", datetime.datetime.now())
