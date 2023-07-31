import streamlit as st
from langchain.llms import HuggingFaceHub
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
import os 



def main():

    load_dotenv()
    
    st.set_page_config(page_title="Ask your pdf")
    st.header("Ask your pdf")

    user_pdf = st.file_uploader("Upload your pdf", type="pdf")
    if user_pdf is not None:
        pdf_reader = PdfReader(user_pdf)

       
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
            length_function=len
            )
        chunks = text_splitter.split_text(text=text)

        embeddings = HuggingFaceInstructEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_texts(chunks, embedding=embeddings)

        query = st.text_input("Ask a question about your PDF: ")
        if query:
            docs = vectorstore.similarity_search(query=query)
            llm = HuggingFaceHub(
            repo_id="tiiuae/falcon-7b",
            model_kwargs={"temperature":.9, "max_length":64, "max_new_tokens":100},
            )
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=query)
            st.write(response)
       


        

if __name__ == "__main__":
    main()