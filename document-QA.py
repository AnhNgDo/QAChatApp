import streamlit as st
import time
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Global Variables
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
SEARCH_TYPE = 'similarity'
SEARCH_K = 3

# read file content, process text data, create Q&A chain
@st.cache_resource
def create_qa_chain(uploaded_file):
    '''
    Read pdf file content, split into text chunks
    then perform embedding and store in vector database

    Parameters:
    uploaded_file: file object (pdf)

    Returns:
    qa_chain: Langchain's QA chain
    text_chunks: list of split texts 
    document_text: string of all text in input file

    '''
    # Add progress bar for debug
    progress_text = "Running create_vector_stores function in progress. Please wait."
    my_bar = st.progress(0, text=progress_text)

    for percent_complete in range(100):
        time.sleep(0.1)
        my_bar.progress(percent_complete + 1, text=progress_text)
    # End progress bar    

    # read file content
    document_text = ""
    pdf_reader = PdfReader(uploaded_file)

    for page in pdf_reader.pages:
        document_text += page.extract_text()
    
    # split file content into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, length_function=len)
    text_chunks = text_splitter.split_text(document_text)

    # embedding 
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # vector store
    db = Chroma.from_texts(text_chunks, embedding=embeddings)

    # define retriever
    retriever = db.as_retriever(search_type=SEARCH_TYPE, search_kwargs={"k": SEARCH_K})

    # llm
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)

    # QA chain
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff" ,retriever=retriever)

    return qa_chain, text_chunks, document_text

##### __main__ #####
# Page title
st.title("🦜🔗 Q&A, Chat with your docs")

# side bar
with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", type="password")

# file upload widget
uploaded_file = st.file_uploader("Upload your PDF document", type=["pdf"])

if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.")
elif uploaded_file is not None:
    qa_chain, text_chunks, document_text = create_qa_chain(uploaded_file)

    st.write("Document Text")
    st.write(document_text)

    st.write(f"Number of Chunks: {len(text_chunks)}")
    st.write(text_chunks)

    question = st.text_input("Enter your question:")
    if question:
        response = qa_chain({"query": question})
        st.write(response['result'])

