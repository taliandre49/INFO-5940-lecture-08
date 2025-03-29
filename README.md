# AI-Powered Document Chatbot with RAG & Vector Search#

I built an intelligent AI chatbot that seamlessly connects to OpenAI, allowing users to interact with multiple PDFs and text files simultaneously. Whether you need a quick summary, precise answers to specific questions, or deep insights from your documents, this chatbot harnesses the power of Retrieval-Augmented Generation (RAG) to deliver the most relevant information in real-time.

To enhance retrieval accuracy, I implemented a context-aware text splitter and chunking strategy, ensuring documents are processed intelligently before being indexed in ChromaDB. By leveraging vector similarity search, the system efficiently identifies and retrieves the most relevant sections based on user queries. Plus, with support for both PDF and text file updates, users can continuously refine their knowledge base with new information.

Simply upload multiple files, and the chatbot will seamlessly manage and analyze them, transforming the way we interact with information. This project isn't just about building an AI assistant—it’s about making research, document analysis, and knowledge extraction more effortless, powerful, and intuitive than ever! 🚀


### Steps to run:
1. Feel free to download the repository
2. Please create a .env and copy and past the following into the file, replacing the place holders with the actual values for your AZURE connection:

    AWS_PROFILE: aaii
    AZURE_OPENAI_API_KEY: <REPLACE WITH API_KEY>
    AZURE_OPENAI_ENDPOINT: <REPLACE WITH ENDPOINT>
    AZURE_OPENAI_MODEL_DEPLOYMENT: gpt-4
    OPENAI_API_KEY: <REPLACE WITH API_KEY>

3. run `pip install requirements` to install and requirements for this project
4. run `Streamlit run chat_with_pdf.py`




