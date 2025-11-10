import numpy as np
import pandas as pd
import gensim
import os
from nltk import sent_tokenize
from gensim.utils import simple_preprocess
import nltk

nltk.download('punkt')
nltk.download('punkt_tab')


def main():
    story = []
    data_path = 'data'

    for filename in os.listdir(data_path):
        file_path = os.path.join(data_path, filename)
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                corpus = f.read()
                raw_sent = sent_tokenize(corpus)
                for sent in raw_sent:
                    tokens = simple_preprocess(sent)
                    if tokens:  # ✅ avoid empty sentences
                        story.append(tokens)

    if not story:
        print("No sentences found in data folder. Please check your files.")
        return

    model = gensim.models.Word2Vec(
        sentences=story,
        vector_size=100,
        window=4,
        min_count=1,  # lowered to 1 so small data still trains
        workers=4
    )
    model.save("word2vec.model")
    print("✅ Word2Vec model saved successfully as 'word2vec.model'")

if __name__ == "__main__":
    main()
