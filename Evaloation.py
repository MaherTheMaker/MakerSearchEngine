import numpy as np

from QueryAction import cosine_similarity


def eval(doc_set, qry_set, D,N, total_vocab, DF,k):
    rel_set = {}
    with open('CISI/CISI.REL') as f:
        for l in f.readlines():
            qry_id = l.lstrip(" ").strip("\n").split("\t")[0].split(" ")[0]
            doc_id = int(l.lstrip(" ").strip("\n").split("\t")[0].split(" ")[-1])
            if qry_id in rel_set:
                rel_set[qry_id].append(doc_id)
            else:
                rel_set[qry_id] = []
                rel_set[qry_id].append(doc_id)

    # print(rel_set["3"])  # note that the dictionary indexes are strings, not numbers.



    precision_list = []
    recall_list = []
    accuracy_list = []
    RR_List=[]

    for i in range(0, len(qry_set)):
        try:
            result_from_cosine= cosine_similarity(k, qry_set[str(i)],D,N,total_vocab,DF)
            result_from_ground_truth = rel_set[str(i)]

            #### for MRR
            for j in range(1,len(result_from_cosine)):
                docnum=result_from_cosine[j]
                if(result_from_ground_truth.__contains__(docnum)):
                    RR_List.append(j)
                    break
            # if(i==0):
            #     print(result_from_cosine)
            #     print(result_from_ground_truth)

            true_Positive = len(set(result_from_cosine) & set(
                result_from_ground_truth))  # set(a) & set(b) gives us intersection between a and b
            false_Positive = len(np.setdiff1d(result_from_cosine, result_from_ground_truth))
            false_Negative = len(np.setdiff1d(result_from_ground_truth, result_from_cosine))
            true_negative = (len(doc_set) - (true_Positive + false_Negative + false_Positive))
            # print("true psotive", true_Positive)
            # print("false negative", false_Negative)

            try:
                precission = (true_Positive) / (true_Positive + false_Positive)
                recall = (true_Positive) / (true_Positive + false_Negative)

                accuracy = (true_negative + true_Positive) / (
                            true_negative + true_Positive + false_Negative + false_Positive)
                print(" precisssion = ",precission," recall = ",recall)
            except ZeroDivisionError:
                pass

            precision_list.append(precission)
            recall_list.append(recall)
            accuracy_list.append(accuracy)



        except KeyError:
            pass

    average_precision = sum(precision_list)/len(precision_list)

    average_recall = sum(recall_list)/len(precision_list)

    Accuracy = sum(accuracy_list)
    print(average_recall,average_precision)

    F_Measure = (2 * average_precision * average_recall) / (average_precision + average_recall)

    print("Average Precision is : ", average_precision)
    print("Average Recall is : ", average_recall)
    print("F-score is : ", F_Measure)
    print("Accuracy : ", Accuracy)

    return precision_list,recall_list,accuracy_list,average_precision,average_recall,F_Measure,Accuracy
    # return precision_list,RR_List
def eval2(doc_set, qry_set, D,N, total_vocab, DF):
    rel_set = {}
    with open('CISI/CISI.REL') as f:
        for l in f.readlines():
            qry_id = l.lstrip(" ").strip("\n").split("\t")[0].split(" ")[0]
            doc_id = int(l.lstrip(" ").strip("\n").split("\t")[0].split(" ")[-1])
            if qry_id in rel_set:
                rel_set[qry_id].append(doc_id)
            else:
                rel_set[qry_id] = []
                rel_set[qry_id].append(doc_id)


    for i in range(1, len(qry_set)):
        result_from_cosine = cosine_similarity(10, qry_set[str(i)], D, N, total_vocab, DF)
        result_from_ground_truth = rel_set[str(i)]
        print("result_from_cosine",result_from_cosine)
        print("result_from_ground_truth",result_from_ground_truth)



    precision_list = []
    recall_list = []
    accuracy_list = []


    return
