import math
from collections import Counter

import numpy as np
from nltk import word_tokenize

from PreProcessingFun import preprocess
from TF_IDF import doc_freq


def cosine_sim(a, b):
    # print("a",a," b",b)
    cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
    return cos_sim


def gen_vector(tokens,N,total_vocab,DF):
    Q = np.zeros((len(total_vocab)))

    counter = Counter(tokens)
    words_count = len(tokens)

    query_weights = {}

    for token in np.unique(tokens):

        tf = counter[token] / words_count
        df = doc_freq(token , DF)
        idf = math.log((N + 1) / (df + 1))

        try:
            ind = total_vocab.index(token)
            Q[ind] = tf * idf
        except:
            pass
    return Q


def cosine_similarity(k, query,D,N,total_vocab,DF):
    preprocessed_query = preprocess(query,1)
    tokens = word_tokenize(str(preprocessed_query))
    # print(tokens)
    # print("\nQuery:", query)

    d_cosines = []

    # print("N",N)
    # print("total", len(total_vocab))
    query_vector = gen_vector(tokens,N,total_vocab,DF)
    # print("query victor ", query_vector)


    for d in D:
        d_cosines.append(cosine_sim(query_vector, d))

    out = np.array(d_cosines).argsort()[-k:][::-1]

    # print("Most similar Documents-IDs : ")

    # print("is out ",out)

    return out

