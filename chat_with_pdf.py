import streamlit as st
from openai import OpenAI
from openai import AzureOpenAI
from os import environ
import dotenv
from dotenv import load_dotenv
import base64
from base64 import b64encode
import PyPDF2
import langchain_openai
from langchain_openai import AzureChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import Document 


load_dotenv() 
st.title("📝 File Q&A with OpenAI")
uploaded_file = st.file_uploader(
    "Upload an article", 
    type=("txt", "pdf"),
    accept_multiple_files= True
)

question = st.chat_input(
    "Ask something about the article",
    disabled=not uploaded_file,
)

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Ask something about the article"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if question and uploaded_file:
    combined_file_content = ""
    for file in uploaded_file:
        if file.type == "application/pdf":
        # Extract text from PDF
            pdf_reader = PyPDF2.PdfReader(file)
            file_content = ""
            for page in pdf_reader.pages:
                file_content += page.extract_text()
        elif file.type == "text/plain":
        # For txt files, just read the content
            file_content = file.read().decode("utf-8")

        combined_file_content += f"\n\n--- Content from {file.name} ---\n\n"
        combined_file_content += file_content

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(combined_file_content)
    documents = [Document(page_content=chunk) for chunk in chunks]
    vectorstore = Chroma.from_documents(documents=documents, embedding=AzureOpenAIEmbeddings(model="text-embedding-3-large"))
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 1})

    # def format_docs(docs):
    #     return "\n\n".join(doc.page_content for doc in docs)

    client = AzureOpenAI(
    api_key= environ['AZURE_OPENAI_API_KEY'],
    api_version= "2023-03-15-preview",
    azure_endpoint=environ['AZURE_OPENAI_ENDPOINT'],
    azure_deployment=environ['AZURE_OPENAI_MODEL_DEPLOYMENT'],
    )
    relevant_chunks = chunks[:5] 


    llm = AzureChatOpenAI(
    azure_deployment="gpt-4o",
    temperature=0.2,
    api_version="2023-06-01-preview",
    max_tokens=None,
    timeout=None,
    max_retries=2,
)
    # Append the user's question to the messages
    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user").write(question)

    retrieved_docs = retriever.invoke(question)
    print(len(retrieved_docs))
    print(type(retrieved_docs))
    print(retrieved_docs[0].page_content)





    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    format_docs(retrieved_docs)
    formatted_docs = format_docs(retrieved_docs)
    print (retrieved_docs)
    print('TESTING')


     # combined_chunk_text = "\n".join(relevant_chunks)
   
    template = """
        You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
        
        Question: {question} 
        
        Context: {context} 
        
        Answer:
    """
    prompt = PromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever| format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    # rag_chain.invoke(question)
    ress = rag_chain.invoke(question)

    print(rag_chain.invoke(question))

    
    # combined_chunk_text = "\n".join(relevant_chunks)
   
    # template = """
    #     You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
    #     If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
        
    #     Question: {question} 
        
    #     Context: {context} 
        
    #     Answer:
    # """
    # prompt = PromptTemplate.from_template(template)

    # rag_chain = (
    # {"context": retriever, "question": RunnablePassthrough()}
    # | prompt
    # | llm
    # | StrOutputParser()
    # )

    # with st.chat_message("assistant"):
    #     stream = client.chat.completions.create(
    #         model="gpt-4o",  # Change this to a valid model name
    #         messages=[
    #             {"role": "system", "content": f"Here's the content of the file:\n\n{combined_file_content}"},
    #             *st.session_state.messages
    #         ],
    #         stream=True
    #     )
    #     # res = rag_chain.invoke(stream)
    #     # ress = rag_chain.invoke(context=combined_chunk_text, question=question)
    #     # response = st.write_stream(stream)
    with st.chat_message("assistant"):
        st.write(ress)
    # # Append the assistant's response to the messages
    st.session_state.messages.append({"role": "assistant", "content": ress})


