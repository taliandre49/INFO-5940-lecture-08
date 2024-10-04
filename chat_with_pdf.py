import streamlit as st
from openai import OpenAI
from openai import AzureOpenAI
from os import environ
import dotenv
from dotenv import load_dotenv
load_dotenv() 
import base64
from base64 import b64encode
from base64 import b64decode
import PyPDF2
import langchain_openai
from langchain_openai import AzureChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter


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

    client = AzureOpenAI(
    api_key= environ['AZURE_OPENAI_API_KEY'],
    api_version= "2023-03-15-preview",
    azure_endpoint=environ['AZURE_OPENAI_ENDPOINT'],
    azure_deployment=environ['AZURE_OPENAI_MODEL_DEPLOYMENT'],
    )
    relevant_chunks = chunks[:5] 

    # Append the user's question to the messages
    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user").write(question)
    combined_chunk_text = "\n".join(relevant_chunks)


    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model="gpt-4o",  # Change this to a valid model name
            messages=[
                {"role": "system", "content": f"Here's the content of the file:\n\n{combined_file_content}"},
                *st.session_state.messages
            ],
            stream=True
        )
        response = st.write_stream(stream)

    # Append the assistant's response to the messages
    st.session_state.messages.append({"role": "assistant", "content": response})


