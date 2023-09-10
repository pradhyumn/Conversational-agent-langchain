import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

def main ():
    load_dotenv()
    st.set_page_config(page_title="ESG")
    st.header("ESG")

    pdf=st.file_uploader("Upload your PDF",type="pdf")

    if pdf is not None:
        pdf_reader=PdfReader(pdf)
        text=""
        for page in pdf_reader.pages:
            text +=page.extract_text()
        
        text_splitter=CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks=text_splitter.split_text(text)

        embeddings=OpenAIEmbeddings()
        knowledge_base=FAISS.from_texts(chunks,embeddings)

        user_input=st.text_input("Enter your prompt query")
        if user_input:
            docs=knowledge_base.similarity_search(user_input)
            #st.write(docs)

            llm=OpenAI()
            chain= load_qa_chain(llm,chain_type="stuff")
            with get_openai_callback() as cb:
                response= chain.run(input_documents=docs,question=user_input)
                print(cb)

            st.write(response)
        #st.write(chunks)
if __name__=='__main__':
    main()
