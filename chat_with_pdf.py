import streamlit as st
from openai import OpenAI
from openai import AzureOpenAI
from os import environ
import dotenv
from dotenv import load_dotenv
load_dotenv()  

st.title("📝 File Q&A with OpenAI")
uploaded_file = st.file_uploader("Upload an article", type=("txt", "md", "pdf"))

question = st.chat_input(
    "Ask something about the article",
    disabled=not uploaded_file,
)

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Ask something about the article"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if question and uploaded_file:
    # Read the content of the uploaded file
    file_content = uploaded_file.read().decode("utf-8")
    print(file_content)
    
    # client = OpenAI(api_key=environ['OPENAI_API_KEY'])
    
    client = AzureOpenAI(
    api_key= environ['AZURE_OPENAI_API_KEY'],
    api_version= "2023-03-15-preview",
    azure_endpoint=environ['AZURE_OPENAI_ENDPOINT'],
    azure_deployment=environ['AZURE_OPENAI_MODEL_DEPLOYMENT'],
)
    # Append the user's question to the messages
    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user").write(question)

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model="gpt-4o",  # Change this to a valid model name
            messages=[
                {"role": "system", "content": f"Here's the content of the file:\n\n{file_content}"},
                *st.session_state.messages
            ],
            stream=True
        )
        response = st.write_stream(stream)

    # Append the assistant's response to the messages
    st.session_state.messages.append({"role": "assistant", "content": response})


