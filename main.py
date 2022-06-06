import doctest
import os
import pickle

import numpy as np
from nltk.corpus import stopwords

import PreProcessingFun as pre
import CISIcleaning as CIclean
import cacmcleaning as CMcleaning
import TF_IDF as DD
import QueryAction as QQ
import Evaloation as EE
import Data1 as dddd1
import Data2 as dddd2


def usefile():
    a_file = open("fulldata.pkl", "rb")
    output = pickle.load(a_file)
    dddd1.liiiist = output;
    a_file.close()

    dddd1.doc_set = output[0]
    dddd1.processed_set = output[1]
    dddd1.N = output[2]
    dddd1.D = output[3]
    dddd1.total_vocab = output[4]
    dddd1.DF = output[5]

    print(dddd1.DF)

    return


def savefile():
    a_file = open("fulldata1.pkl", "wb")
    pickle.dump(dddd1.liiiist, a_file)
    a_file.close()

    a_file = open("fulldata2.pkl", "wb")
    pickle.dump(dddd2.liiiist, a_file)
    a_file.close()
    return


def startCISI():
    dddd1.doc_set,title = CIclean.cleanAll()
    dddd1.title=title

    for i in dddd1.doc_set:
        doc_token_id = i
        dddd1.processed_set[doc_token_id] = pre.preprocess(dddd1.doc_set[str(i)])
    tokens_set = pre.tokenizer(dddd1.processed_set)

    dddd1.DF = DD.DF(tokens_set)

    tf_idf = DD.TF_IDF(tokens_set, dddd1.DF)
    dddd1.total_vocab = [x for x in dddd1.DF]
    total_vocab_size = len(dddd1.total_vocab)  # number of term

    dddd1.N = len(tokens_set)  # number of Docs

    dddd1.D = np.zeros((dddd1.N, total_vocab_size))  # total_vocab_size is the length of dddd.DF
    for i in tf_idf:
        try:
            ind = dddd1.total_vocab.index(i[1])
            dddd1.D[i[0]][ind] = tf_idf[i]
        except:
            pass

    dddd1.liiiist = [dddd1.doc_set, dddd1.processed_set, dddd1.N, dddd1.D, dddd1.total_vocab, dddd1.DF]
    return


def startCACM():
    dddd2.doc_set = CMcleaning.cleanAll()
    for i in dddd2.doc_set:
        doc_token_id = i
        dddd2.processed_set[doc_token_id] = pre.preprocess(dddd2.doc_set[str(i)])
    tokens_set = pre.tokenizer(dddd2.processed_set)
    dddd2.DF = DD.DF(tokens_set)
    tf_idf = DD.TF_IDF(tokens_set, dddd2.DF)
    dddd2.total_vocab = [x for x in dddd2.DF]
    total_vocab_size = len(dddd2.total_vocab)  # number of term

    dddd2.N = len(tokens_set)  # number of Docs

    dddd2.D = np.zeros((dddd2.N, total_vocab_size))  # total_vocab_size is the length of dddd.DF
    for i in tf_idf:
        try:
            ind = dddd2.total_vocab.index(i[1])
            dddd2.D[i[0]][ind] = tf_idf[i]
        except:
            pass

    dddd2.liiiist = [dddd2.doc_set, dddd2.processed_set, dddd2.N, dddd2.D, dddd2.total_vocab, dddd2.DF]
    return
def Eval():
    qry_set = CIclean.cleanQRY()

    EE.eval(dddd1.doc_set, qry_set, dddd1.D, dddd1.N, dddd1.total_vocab, dddd1.DF)
    return

    # CIclean.cleanQRY()


def Query1(strQ):
    return QQ.cosine_similarity(10, strQ, dddd1.D, dddd1.N, dddd1.total_vocab, dddd1.DF)

def Query2(strQ):
    return QQ.cosine_similarity(10, strQ, dddd2.D, dddd2.N, dddd2.total_vocab, dddd2.DF)


def jojo():
    from sklearn.feature_extraction.text import TfidfVectorizer
    stop_words = stopwords.words('english')
    # tfidf vectorizer of scikit learn
    vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=10000, max_df=0.5, use_idf=True,
                                 ngram_range=(1, 3))
    X = vectorizer.fit_transform(dddd1.processed_set)
    print(X.shape)  # check shape of the document-term matrix
    terms = vectorizer.get_feature_names()
    from sklearn.cluster import KMeans
    num_clusters = 5
    km = KMeans(n_clusters=num_clusters)
    km.fit(X)
    clusters = km.labels_.tolist()

    tewmp=5
    print(tewmp)

    order_centroids = km.cluster_centers_.argsort()[:, ::-1]

    print(len(order_centroids))
    print(order_centroids[0])




    from sklearn.utils.extmath import randomized_svd
    U, Sigma, VT = randomized_svd(X, n_components=5, n_iter=100,
                                  random_state=122)
    # printing the concepts
    for i, comp in enumerate(VT):
        terms_comp = zip(dddd1.doc_set, comp)
        sorted_terms = sorted(terms_comp, key=lambda x: x[1], reverse=True)
        print("Concept " + str(i) + ": ")
        for t in sorted_terms:
            print(t[0])
        print(" ")


    return


if __name__ == '__main__':
    startCISI()
    jojo()
    # Eval()
    # startCISI()
    # print(dddd1.doc_set["99"])
    # print(dddd2.doc_set["99"])
    # Query2()



