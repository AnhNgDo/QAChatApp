import streamlit as st
import os, time, tempfile
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler

# Global Variables
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
SEARCH_TYPE = 'similarity'
SEARCH_K = 3

# callback handler for streaming output
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text) 

# read file content, process text data, create Q&A chain
@st.cache_resource
def create_qa_chain(uploaded_files):
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
    # Progress bar for debug
    # progress_text = "Running create_vector_stores function in progress. Please wait."
    # my_bar = st.progress(0, text=progress_text)

    # for percent_complete in range(100):
    #     time.sleep(0.1)
    #     my_bar.progress(percent_complete + 1, text=progress_text)
    # End progress bar    

     # Read documents
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    for file in uploaded_files:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        loader = PyPDFLoader(temp_filepath)
        docs.extend(loader.load())

     # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    doc_chunks = text_splitter.split_documents(docs)

    # embedding 
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # vector store
    vector_db = DocArrayInMemorySearch.from_documents(doc_chunks, embedding=embeddings)

    # define retriever
    retriever = vector_db.as_retriever(search_type=SEARCH_TYPE, search_kwargs={"k": SEARCH_K})
    
    # llm for condense chat history + follow up question
    # streaming = False to not show the condensed question in chat
    condense_llm = ChatOpenAI(model_name="gpt-3.5-turbo-0613",
                              temperature=0,
                              streaming=False,
                              openai_api_key=openai_api_key)
    
    # llm for user for question + contextual conversation
    # using 16k tokens version if need larger context window
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", 
                     temperature=0,
                     streaming=True,
                     openai_api_key=openai_api_key)

    # memory for contextual conversation
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # custom prompt for condense chat history + follow up question
    condense_template = """
    Given the following conversation and a follow up question, 
    rephrase the follow up question to be a standalone question in the same language.\n
    Chat History: {chat_history}\n
    Follow Up Input: {question}\n
    Standalone question:
    """
    condense_question_prompt = PromptTemplate(input_variables=["chat_history", "question"],template=condense_template)

    # custom prompt for user question + contextual conversation
    # PROMPT ENGINEER TECHNIQUE: 
    # - to enable the LLM to answer questions using its general knowledge:
    #       "If the context is not relevant, please answer the question by using your own knowledge about the topic."
    # - to ground the LLM to answer questions using ONLY the document information:
    #       "If you don't know the answer, just say that you don't know, don't try to make up an answer."" 
    question_template = """
    Use the following pieces of context and a follow up question to answer the question at the end.
    If the context is not relevant, please answer the question by using your own knowledge about the topic.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use dot points format as much as possible when answer the question.\n
    Context: {context}\n
    Question: {question}\n
    Helpful Answer:
    """
    question_prompt = PromptTemplate(input_variables=["context", "question"],template=question_template,)

    # QA coversation chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        condense_question_prompt=condense_question_prompt,
        condense_question_llm=condense_llm,
        chain_type="stuff", 
        memory=memory,
        combine_docs_chain_kwargs={"prompt": question_prompt}
    )

    return qa_chain, doc_chunks

##### __main__ #####
# Page title
st.set_page_config(page_title="Anh's Bot", page_icon="🦜")
st.title("🤖 Anh Q&A Chatbot 🦜🔗")

# side bar
with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", type="password")

if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.")
    st.stop()

# file upload widget
uploaded_files = st.file_uploader("Upload your PDF documents", type=["pdf"], accept_multiple_files=True)

if not uploaded_files:
    st.info("Please upload PDF documents to continue.")
    st.stop()

# create Q&A conversational chain
qa_chain, doc_chunks_chunks = create_qa_chain(uploaded_files)

### Debug printing ###
# st.write("Document Text")
# st.write(document_text[:300])

# st.write(f"Number of Chunks: {len(doc_chunks)}")
# st.write(doc_chunks)

# st.write("QA Chain:")
# st.write(qa_chain)
### End Debug printing ##

# Initialize chat history
if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if question := st.chat_input("Ask me any thing..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": question})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(question)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        stream_handler = StreamHandler(message_placeholder)
        answer = qa_chain.run(question, callbacks=[stream_handler])
        st.session_state.messages.append({"role": "assistant", "content": answer})