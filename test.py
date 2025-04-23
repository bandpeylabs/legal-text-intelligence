import os
import streamlit as st
from dotenv import load_dotenv
from openai import AzureOpenAI
from PIL import Image
from langchain_community.docstore.document import Document
import pickle

from test_vectorstore import create_docs_vectorstore, load_docs_vectorstore

# Load environment variables
load_dotenv()

# Fetch environment variables
endpoint = os.getenv("ENDPOINT")
deployment = os.getenv("DEPLOYMENT")
subscription_key = os.getenv("SUBSCRIPTION_KEY")
api_version = os.getenv("API_VERSION")

# Check if the environment variables are present
if not endpoint or not subscription_key:
    st.error("Endpoint or Subscription Key is missing from the .env file.")
    st.stop()

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_key=subscription_key,
    api_version=api_version,
    azure_endpoint=endpoint
)

# Sidebar module selection
app_name = st.sidebar.radio("Choose your module:", ("Legal Information", "Odontology Information"))

# Load and display image
image = Image.open("image.png")
st.image(image, caption="Legal Assistant", use_column_width=True)

# Streamlit UI
st.title("💬 Azure OpenAI Chatbot")
st.write("Ask me anything!")


def read_pdf_pagewise(path_name, file_name, page_range=None, remove_newline=True, bbox=None):
    file_path = os.path.join(path_name, file_name)
    source_name = file_name.split('.pdf')[0]
    import fitz

    doc= fitz.open(file_path)

    if bbox is None:
        bbox = doc.load_page(0).rect

    if page_range is None:
        page_range = (0, doc.page_count)

    source_chunks = []

    for page in range (*page_range):
        if remove_newline:
            chunk = doc[page].get_textbox(bbox).replace('\n','')

        else:

            chunk = doc[page].get_textbox(bbox)

        source_chunks.append(Document(page_content=chunk, metadata={'source': source_name + ', page no ' + str(page)}
                                     
                                      )
                                                      
                            )
    
    return source_chunks


embed_path = 'EmbedFiles'
embedding_deployment = 'text-embedding-ada-002'

if app_name == "Legal Information":

    legal_path_raw = os.path.join('AppDocs', 'Legaldocs')
    list_legal_pdf = [x for x in os.listdir(legal_path_raw) if '.pdf' in x]
    list_legal_vectorstores = [x[:-2] + '.pkl' for x in list_legal_pdf]

    vectorstore_legal = []

    for k in range (len(list_legal_pdf)):

        try:
            vectorstore_path = os.path.join(embed_path, list_legal_vectorstores[k])
            vectorstore_legal.append(load_docs_vectorstore(vectorstore_path, embedding_deployment))

        except:
            path_name = legal_path_raw
            file_name = list_legal_pdf[k]
            source_chunks = read_pdf_pagewise(path_name, file_name)

            faiss_index = create_docs_vectorstore(source_chunks, vectorstore_path, embedding_deployment)
            vectorstore_legal.append(load_docs_vectorstore(vectorstore_path, embedding_deployment))

    
    vectorstore_legal_all = vectorstore_legal[0]
    for k in range(1, len(list_legal_vectorstores)):
        vectorstore_legal_all.merge_from(vectorstore_legal[k])




# Text input and model invocation
user_input = st.text_input("Your Question:", placeholder="E.g., What MLB team won the World Series during COVID-19?")

if st.button("Ask"):
    if user_input:
        try:
            with st.spinner("Thinking..."):
                response = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": user_input}
                    ],
                    model=deployment,
                    max_tokens=1000,
                    temperature=0.7
                )
                st.success("Response:")
                st.write(response.choices[0].message.content)
        except Exception as e:
            st.error(f"Something went wrong: {e}")
    else:
        st.warning("Please enter a question before clicking Ask.")
