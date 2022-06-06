from datetime import datetime


def cleanAll():
    with open('cacm/cacm.all') as CISI_file:
        lines = ""
        for l in CISI_file.readlines():
            lines += "\n" + l.strip() if l.startswith(".") else " " + l.strip()
        lines = lines.lstrip("\n").split("\n")

    print("Done")
    date_set={}
    doc_set = {}
    doc_id = ""
    doc_text = ""
    cnt=0
    for l in lines:
        if l.startswith(".I"):
            doc_id = l.split(" ")[1].strip()
        elif l.startswith(".X"):
            doc_set[doc_id] = doc_text.lstrip(" ")
            doc_id = ""
            doc_text = ""
        elif l.startswith(".N"):
                m=l.split((" "))[3]
                d=l.split((" "))[4]
                y=l.split((" "))[5]
                date=m+" "+d+" "+y
                date1_obj = datetime.strptime(date ,'%B %d, %Y')
                date_set[doc_id]=date1_obj


        elif l.startswith(".B"):
            pass
        else:
            doc_text += l.strip()[3:] + " "  # The first 3 characters of a line can be ignored.
    print(cnt)
    # Print something to see the dictionary structure, etc.
    print(f"Number of documents = {len(doc_set)}" + ".\n")
    return doc_set;

def cleanQRY():
    with open('cacm/query.text') as f:
        lines = ""
        for l in f.readlines():
            lines += "\n" + l.strip() if l.startswith(".") else " " + l.strip()
        lines = lines.lstrip("\n").split("\n")

    qry_set = {}
    qry_id = ""
    for l in lines:
        if l.startswith(".I"):
            qry_id = l.split(" ")[1].strip()
        elif l.startswith(".W"):
            qry_set[qry_id] = l.strip()[3:]
            qry_id = ""

    # Print something to see the dictionary structure, etc.
    print(f"Number of queries = {len(qry_set)}" + ".\n")
    return qry_set


if __name__ == '__main__':
   w= cleanAll()
   print(w["2000"])
   q= cleanQRY()
   print(q["5"])