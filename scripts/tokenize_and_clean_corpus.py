import sys
import unicodedata
from pathlib import Path

from blingfire import text_to_sentences


MIN_ARTICLE_WORD_COUNT = 50


def main():

    wiki_dump_file_in = Path(sys.argv[1])
    wiki_dump_file_out = wiki_dump_file_in.parent / \
        f'{wiki_dump_file_in.stem}_sentences{wiki_dump_file_in.suffix}'

    print(f'Tokenizing {wiki_dump_file_in} to {wiki_dump_file_out}...')
    
    with open(wiki_dump_file_out, 'w', encoding='utf-8') as out_f:
        with open(wiki_dump_file_in, 'r', encoding='utf-8') as in_f:
            for line in in_f:
                words = line.split()
                word_count = len(words)
                if word_count >= MIN_ARTICLE_WORD_COUNT:
                    # removes weird '\xa0' character
                    line = unicodedata.normalize("NFKD", line)
                    sentences = text_to_sentences(line)
                    out_f.write(sentences + '\n')
    
    print(f'Successfully tokenized {wiki_dump_file_in} to {wiki_dump_file_out}...')


if __name__ == '__main__':
    main()
