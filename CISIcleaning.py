

def cleanAll():
    with open('CISI/CISI.ALL') as CISI_file:
        lines = ""
        for l in CISI_file.readlines():
            lines += "\n" + l.strip() if l.startswith(".") else " " + l.strip()
        lines = lines.lstrip("\n").split("\n")

    print("Done Cleaning")

    doc_set = {}
    doc_id = ""
    doc_text = ""
    for l in lines:
        if l.startswith(".I"):
            doc_id = l.split(" ")[1].strip()
        elif l.startswith(".X"):
            doc_set[doc_id] = doc_text.lstrip(" ")
            doc_id = ""
            doc_text = ""
        else:
            doc_text += l.strip()[3:] + " "  # The first 3 characters of a line can be ignored.

    # Print something to see the dictionary structure, etc.
    print(f"Number of documents = {len(doc_set)}" + ".\n")
    return doc_set;

def cleanQRY():
    with open('CISI/CISI.QRY') as f:
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