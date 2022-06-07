import json
import os

import requests
from flask import Flask, jsonify
import numpy as np
from sqlalchemy import JSON

import PreProcessingFun as pre
import CISIcleaning as CIclean
import TF_IDF as DD
import QueryAction as QQ
import Evaloation as EE

import Data1 as DDDD1
import Data2 as DDDD2

import main as MM

app = Flask(__name__)


# def hello():
#     doc_set = CIclean.cleanAll()
#
#     processed_set = {}
#     proc_token_id = ""
#     proc_token_text = ""
#
#     for i in doc_set:
#         doc_token_id = i
#         processed_set[doc_token_id] = pre.preprocess(doc_set[str(i)])
#
#     tokens_set = pre.tokenizer(processed_set)
#
#     DF = DD.DF(tokens_set)
#
#     tf_idf = DD.TF_IDF(tokens_set, DF)
#
#     total_vocab = [x for x in DF]
#     total_vocab_size = len(total_vocab)  # number of term
#
#     N = len(tokens_set)  # number of Docs
#
#     D = np.zeros((N, total_vocab_size))  # total_vocab_size is the length of DF
#     for i in tf_idf:
#         try:
#             ind = total_vocab.index(i[1])
#             D[i[0]][ind] = tf_idf[i]
#         except:
#             pass
#
#     qry_set = CIclean.cleanQRY()
#     QQ.cosine_similarity(10, qry_set["1"], D, N, total_vocab, DF)
#
#     EE.eval(doc_set, qry_set, D, N, total_vocab, DF)

@app.route("/")
def home():
    # hello()
    MM.Eval()
    return "Hello, Flask!"


from flask import request


@app.route('/api/add_message', methods=['POST', 'GET'])
def add_message():
    return "username"


from textblob import TextBlob


@app.route('/check/<sentence>')
def correct_sentence_spelling(sentence):
    sentence = TextBlob(sentence)

    result = sentence.correct()

    print(result)
    return str(result)


@app.route('/Eval')
def Eval():
    # show the user profile for that user
    stuffs = MM.Eval()
    print("stuff", len(stuffs[0]))
    lsit = []
    for i in range(len(stuffs[0])):
        lsit.append({'doc': "pres=" + str(stuffs[0][i]) + " recall = " + str(stuffs[1][i]), 'id': str(i)})
    lsit.append({'doc': "averege pres =" + str(stuffs[3]) + "     average recall =" + str(
        stuffs[4]) + "    F_Measure = " + str(stuffs[5]), 'id': "100"})
    print(lsit)

    return jsonify(lsit)


@app.route('/Search1/<QueryText>')
def SearchCISI(QueryText):
    # show the user profile for that user
    Q = MM.Query1(QueryText)
    print("q", Q)
    lsit = []
    for i in range(len(Q)):
        lsit.append({'doc': DDDD1.doc_set[str(Q[i])], 'id': str(Q[i])})
    print(lsit)

    return jsonify(lsit)


@app.route('/Search3/<QueryText>')
def SearchCISIWithClisuter(QueryText):
    # show the user profile for that user
    MM.checkCluster(QueryText)
    qry = "The need to provide personnel for the information field."
    res = MM.checkCluster(QueryText)
    lsit = []
    if res != -1:
        Q = MM.Query1Cluster(qry, res)
        for i in Q:
            lsit.append({'doc': DDDD1.doc_set[str(i)], 'id': str(i)})

    else:
        Q = MM.Query1(QueryText)
        print("q", Q)

        for i in range(len(Q)):
            lsit.append({'doc': DDDD1.doc_set[str(Q[i])], 'id': str(Q[i])})
    print(lsit)

    return jsonify(lsit)


@app.route('/Search2/<QueryText>')
def SearchCACM(QueryText):
    # show the user profile for that user
    Q = MM.Query2(QueryText)
    print(Q)
    lsit = []
    for i in range(len(Q)):
        lsit.append({'doc': DDDD2.doc_set[str(Q[i])], 'id': str(Q[i])})
    print(lsit)

    return jsonify(lsit)


if __name__ == '__main__':
    # if os.path.exists("fulldata.pkl"):
    #     MM.usefile()
    #     print(True)
    # else:
    #     MM.startCISI()
    #     MM.savefile()
    MM.start()
    MM.startCACM()
    MM.startCISI()
    MM.jojo()
    MM.startCISI_cluser(0)
    MM.startCISI_cluser(1)
    MM.startCISI_cluser(2)
    MM.startCISI_cluser(3)
    MM.startCISI_cluser(4)
    app.run(debug=False, port=4000)
