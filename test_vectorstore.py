import openai
import os

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings

import pickle
import httpx

httpx_client = httpx.Client(verify=False)


def create_docs_vectorstore(source_chunks, vectorstore_path,deployment):

    embeddings = OpenAIEmbeddings(deployment= deployment, http_client=httpx_client)


    text = []

    text_embeddings = []
    metadatas = []
    
    i=0
    for doc in source_chunks:
        i = i+1
        print(i / len(source_chunks))
        text.append(doc.page_content)

        text_embeddings.append(embeddings.embed_documents([doc.page_content])[0])

    
    text_embeddings_pairs = list(zip(text, text_embeddings))
    faiss_ = FAISS.from_embeddings(text_embeddings_pairs, embeddings, metadatas=metadatas)

    with open(vectorstore_path, "wb") as f:
        pickle.dump((text, text_embeddings, metadatas), f)

    return faiss_


def load_docs_vectorstore(vectorstore_path, deployment):

    embeddings = OpenAIEmbeddings(deployment=deployment, http_client=httpx_client)

    with open (vectorstore_path, "rb") as f:

        (text, text_embeddings, metadatas) = pickle.load(f)

    text_embeddings_pairs = list(zip(text, text_embeddings))
    return FAISS.from_embeddings(text_embeddings_pairs, embeddings, metadatas=metadatas)