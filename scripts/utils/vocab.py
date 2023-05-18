

def load_vocab(vocab_file):
    """
    Returns 3 dicts from vocab built with GloVe:
        - str2idx
        - idx2str
        - str2count
    str is word, idx is index given by glove, counts are total word count given
    by glove
    """
    # read vocab
    with open(vocab_file, 'r', encoding="utf-8") as f:
        vocab = []
        for line in f:
            vocab.append(line.strip())
    # create dicts
    str2idx = dict()
    idx2str = dict()
    str2count = dict()
    for i, word_and_count in enumerate(vocab):
        word, count = word_and_count.split()
        str2idx[word] = i+1
        idx2str[i+1] = word
        str2count[word] = int(count)
    return (str2idx, idx2str, str2count)
