import doctest
import os
import pickle
import sys

import numpy as np
from nltk import word_tokenize
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


def start():
    dddd1.doc_set, title = CIclean.cleanAll()
    dddd1.title = title
    dddd2.doc_set = CMcleaning.cleanAll()


def startCISI():
    for i in dddd1.doc_set:
        doc_token_id = i
        dddd1.processed_set[doc_token_id] = pre.preprocess(dddd1.doc_set[str(i)], 1)
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


def startCISI_cluser(clus_num):
    for i in dddd1.arr_dic[clus_num]:
        doc_token_id = i
        dddd1.arr_proc[clus_num][doc_token_id] = pre.preprocess(dddd1.arr_dic[clus_num][str(i)],1)

    tokens_set = pre.tokenizer(dddd1.arr_proc[clus_num])

    print("\n \n ")
    print("token setlen ", len(tokens_set))
    print("dddd1.arr_dic[clus_num].keys()", dddd1.arr_dic[clus_num].keys())
    dddd1.arr_DF[clus_num] = DD.DF2(tokens_set, dddd1.arr_dic[clus_num].keys())

    tf_idf = DD.TF_IDF2(tokens_set, dddd1.arr_DF[clus_num], dddd1.arr_dic[clus_num].keys())
    dddd1.arr_total[clus_num] = [x for x in dddd1.arr_DF[clus_num]]
    total_vocab_size = len(dddd1.arr_total[clus_num])  # number of term

    dddd1.arr_N[clus_num] = len(dddd1.arr_dic[clus_num])  # number of Docs

    dddd1.arr_D[clus_num] = np.zeros(
        (dddd1.arr_N[clus_num], total_vocab_size))  # total_vocab_size is the length of dddd.DF
    for i in tf_idf:
        try:
            ind = dddd1.arr_total[clus_num].index(i[1])
            dddd1.arr_D[clus_num][i[0]][ind] = tf_idf[i]
        except:
            pass

    # dddd1.liiiist = [dddd1.doc_set, dddd1.processed_set, dddd1.N, dddd1.D, dddd1.total_vocab, dddd1.DF]
    return


def startCACM():
    for i in dddd2.doc_set:
        doc_token_id = i
        dddd2.processed_set[doc_token_id] = pre.preprocess(dddd2.doc_set[str(i)],2)
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
    sum2=0
    # for i in range(1,11):
    #     print(i)
    #     hotfudge = EE.eval(dddd1.doc_set, qry_set, dddd1.D, dddd1.N, dddd1.total_vocab, dddd1.DF,i)
    #     sum+=hotfudge
    hotfudge = EE.eval(dddd1.doc_set, qry_set, dddd1.D, dddd1.N, dddd1.total_vocab, dddd1.DF, 10)
    # randlist=hotfudge[1]
    # sumRR=0.0
    # for r in randlist:
    #     sumRR+= 1/r
    # MRR= sumRR/len(hotfudge[0])
    # print("MRR",MRR)
    return hotfudge

    # CIclean.cleanQRY()


def EvalClusters():
    qry_set = CIclean.cleanQRY()

    hotfudge = EE.eval(dddd1.doc_set, qry_set, dddd1.D, dddd1.N, dddd1.total_vocab, dddd1.DF)
    return hotfudge


def Query1(strQ):
    return QQ.cosine_similarity(10, strQ, dddd1.D, dddd1.N, dddd1.total_vocab, dddd1.DF)


def Query2(strQ):
    return QQ.cosine_similarity(10, strQ, dddd2.D, dddd2.N, dddd2.total_vocab, dddd2.DF)


def Query1Cluster(strQ, clus_num):
    return QQ.cosine_similarity(10, strQ, dddd1.arr_D[clus_num], dddd1.arr_N[clus_num], dddd1.arr_total[clus_num],
                                dddd1.arr_DF[clus_num])


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

    # centers_5=km.cluster_centers_.argsort()[:, ::-1]
    #
    # allcluster_5=km.cluster_centers_.argsort()
    #
    # print("center=",centers_5)
    # print("allclusters=",allcluster_5)

    order_centroids = km.cluster_centers_.argsort()[:, ::-1]

    print(len(order_centroids))
    print(order_centroids[0])

    T = []
    centers = []

    from sklearn.utils.extmath import randomized_svd
    U, Sigma, VT = randomized_svd(X, n_components=5, n_iter=100,
                                  random_state=122)
    # printing the concepts
    for i, comp in enumerate(VT):

        terms_comp = zip(dddd1.doc_set, comp)
        sorted_terms = sorted(terms_comp, key=lambda x: x[1], reverse=True)[:int(len(dddd1.doc_set) / num_clusters)]
        print("Concept " + str(i) + ": ")
        for t in sorted_terms:
            T.append(t[0])
            dddd1.arr_dic[i][str(t[0])] = dddd1.doc_set[str(t[0])]

        print(" ")

    for i, comp in enumerate(VT):
        terms_comp = zip(dddd1.doc_set, comp)
        sorted_terms = sorted(terms_comp, key=lambda x: x[1], reverse=True)[:1]
        print("Concept " + str(i) + ": ")
        for t in sorted_terms:
            centers.append(t[0])
            print(t[0])
        print(" ")
    dddd1.centers = centers
    return


def checkCluster(query):
    preprocessed_query = pre.preprocess(query)
    tokens = word_tokenize(str(preprocessed_query))
    print(tokens)
    print("\nQuery:", query)

    d_cosines = []

    print("N", dddd1.N)
    print("total", len(dddd1.total_vocab))
    query_vector = QQ.gen_vector(tokens, dddd1.N, dddd1.total_vocab, dddd1.DF)

    print(query_vector)
    for c in dddd1.centers:
        print("c", c)
        print("d[c]", dddd1.D[int(c)])
        d_cosines.append(QQ.cosine_sim(query_vector, dddd1.D[int(c)]))
    print("d_cosines", d_cosines)
    out = np.array(d_cosines).argsort()[::-1]
    print("cluster ", out[0])
    res = out[0]

    if (d_cosines[res] == 0):
        return -1
    return out[0]


def check():
    print(dddd1.arr_D[4])
    print(dddd1.arr_N[4])
    print(dddd1.arr_total[3])

    return


if __name__ == '__main__':
    start()
    # startCACM()
    startCISI()
    print(Eval())
    # jojo()
    # startCISI_cluser(0)
    # startCISI_cluser(1)
    # startCISI_cluser(2)
    # startCISI_cluser(3)
    # startCISI_cluser(4)
    # check()

    # startCISI_cluser()
    # Eval()
    # startCISI()
    # print(dddd1.doc_set["99"])
    # print(dddd2.doc_set["99"])
    # Query2()
