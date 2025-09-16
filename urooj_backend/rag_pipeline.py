# rag_pipeline.py

import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams


# 1. Load PDFs from data folder
def load_and_split_documents(pdf_folder="./data"):
    loader = DirectoryLoader(pdf_folder, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    print(f"Loaded {len(documents)} document pages.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    return chunks

# 2. Create or Load Vectorstore
def setup_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True}
    )

    qdrant_client = QdrantClient(host="localhost", port=6333)

    collection_name = "rag_pipeline_memory"

    if collection_name not in [c.name for c in qdrant_client.get_collections().collections]:
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )

        vectorstore = Qdrant(
            client=qdrant_client,
            collection_name=collection_name,
            embeddings=embeddings,
        )

        vectorstore.add_documents(chunks)

    else:
    # Load existing collection without re-adding
        vectorstore = Qdrant(
            client=qdrant_client,
            collection_name=collection_name,
            embeddings=embeddings,
        )


    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# 3. Load Gemini Model
def load_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key="AIzaSyCcfO5KP82yGv8WmUeG6vFMKPY3y0PWLnI",  
        temperature=0.3
    )

# 4. Create Conversational RAG Chain
def setup_qa_chain():
    chunks = load_and_split_documents()
    retriever = setup_vectorstore(chunks)
    llm = load_llm()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    prompt = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template="""
You are an intelligent and friendly AI assistant having a conversation with a human. Use the chat history and the provided context to answer the question in a helpful, natural, and conversational way.

If the answer is not available in the provided context, use your own knowledge to answer.

- Always speak directly to the user using "you".
- Use a friendly and empathetic tone.
- Avoid restating facts like "the human user said..."

Context:
{context}

Chat History:
{chat_history}

Current Question:
{question}

Your Response:"""
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt}
    )

    return qa_chain
