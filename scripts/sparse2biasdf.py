
import argparse
import logging

from scipy import sparse

from utils.vocab import load_vocab
from utils.bias import dpmi_byword


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")


def main():

    parser = argparse.ArgumentParser()
    required = parser.add_argument_group('required named arguments')
    required.add_argument('--vocab', type=str, required=True)
    required.add_argument('--cooc', type=str, required=True)
    required.add_argument('--a', type=str, required=True)
    required.add_argument('--b', type=str, required=True)
    required.add_argument('--out', type=str, required=True)
    required.add_argument('--smoothing', type=float, required=True)
    args = parser.parse_args()

    logging.info("Loading input data...")
    str2idx, idx2str, str2count = load_vocab(args.vocab)
    cooc_matrix = sparse.load_npz(args.cooc)
    del idx2str

    logging.info("Getting words lists...")
    words_a = [
        line.strip().lower() for line in open(f'words_lists/{args.a}.txt','r')]
    words_b = [
        line.strip().lower() for line in open(f'words_lists/{args.b}.txt','r')]
    words_target = [
            w for w, freq in str2count.items() if w not in words_a + words_b]
    del str2count

    logging.info("Computing DPPMI bias wrt each target word...")
    df_dppmi = dpmi_byword(cooc_matrix, words_target,
                           words_a, words_b, str2idx, smoothing=args.smoothing)
    del cooc_matrix

    logging.info("Saving DPPMI results in csv...")
    df_dppmi.to_csv(args.out, index=False)
    del df_dppmi

    logging.info("DONE!")


if __name__ == "__main__":
    main()
