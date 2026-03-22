import os

def load_documents(folder_path):
    documents = []

    for file in os.listdir(folder_path):
        path = os.path.join(folder_path, file)

        with open(path, "r") as f:
            text = f.read()

        documents.append(text)

    return documents