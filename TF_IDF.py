from collections import Counter

import numpy as np
from nltk import word_tokenize


def DF(tokens_set):
    DF = {}

    for i in range(len(tokens_set)):
        tokens = tokens_set[str(i + 1)]
        for w in tokens:
            try:
                DF[w].add(i)
            except:
                DF[w] = {i}
    for i in DF:
        DF[i] = len(DF[i])
    return DF;

def doc_freq(word,DF):
    c = 0
    try:
        c = DF[word]
    except:
        pass
    return c

def TF_IDF(tokens_set,DF):
    doc = 0
    N = len(tokens_set)
    tf_idf = {}

    tokens = tokens_set[str(1)]
    for i in range( len(tokens_set)):
        if (i > 1):
            tokens = tokens_set[str(i)]

        counter = Counter(tokens)
        words_count = len(tokens)

        for token in np.unique(tokens):
            tf = counter[token] / words_count
            df = doc_freq(token, DF)
            idf = np.log((N + 1) / (df + 1))

            tf_idf[doc, token] = tf * idf
        doc += 1

    print("tf-idf done")
    return tf_idf



# def cosine_similarity(k, query):
#     preprocessed_query = preprocess(query)
#     tokens = word_tokenize(str(preprocessed_query))
#
#     # print("\nQuery:", query)
#
#     d_cosines = []
#
#     query_vector = gen_vector(tokens)
#
#     for d in D:
#         d_cosines.append(cosine_sim(query_vector, d))
#
#     out = np.array(d_cosines).argsort()[-k:][::-1]
#
#     # print("Most similar Dpocuments-IDs : ")
#
#     # print(out)
#
#     return out
#
#
# def gen_vector(tokens):
#     Q = np.zeros((len(total_vocab)))
#
#     counter = Counter(tokens)
#     words_count = len(tokens)
#
#     query_weights = {}
#
#     for token in np.unique(tokens):
#
#         tf = counter[token] / words_count
#         df = doc_freq(token)
#         idf = math.log((N + 1) / (df + 1))
#
#         try:
#             ind = total_vocab.index(token)
#             Q[ind] = tf * idf
#         except:
#             pass
#     return Q