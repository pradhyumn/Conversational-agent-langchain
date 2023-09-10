import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from PyPDF2 import PdfReader
import langchain
langchain.verbose = False
def get_text_from_files(file_list):
    text = ""
    for file in file_list:
        ext_name = file.name.split(".")[-1].lower()
        if ext_name == "pdf":   
            pdf_reader = PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
        elif ext_name == "txt":
            text = file.getvalue().decode("utf-8")
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=800,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    return chunks

##get keys from .env file
import os
from dotenv import load_dotenv
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vector_store

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(openai_api_key=openai_api_key,
                     temperature=0.25)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
    )
    return conversation_chain

def handle_user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            user = st.chat_message(name="User", avatar="💃")
            user.write(message.content)
        else:
            assistant = st.chat_message(name="J.A.A.F.A.R.", avatar="🤖")
            assistant.write(message.content)

def main():
    st.set_page_config(page_title="Chat with multiple files", page_icon="🤓")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple files 📚")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_user_input(user_question)
    
    # print(st.session_state.chat_history)
    print(st.session_state.conversation)



    with st.sidebar:
        st.subheader("Your files")
        files_or_urls = st.file_uploader(
            "Upload your files and click `Process`",
            accept_multiple_files=True,
        )
        ## allow simple text input
        text_info = st.text_area("Or paste your text here:")
        if st.button("Process"):
            with st.spinner("Processing files"):
                if text_info:   
                    raw_text = str(text_info)
                else:
                    raw_text = ""
    
                ## allow if only text is input
                if files_or_urls:
                    raw_text = raw_text+ "\n" + str(get_text_from_files(files_or_urls))
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)




if __name__ == "__main__":
    main()