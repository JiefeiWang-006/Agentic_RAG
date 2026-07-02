
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
from langchain_community.vectorstores import Chroma
response_model = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key="sk-4c891358a2674840b2de0929394dd01d",
    openai_api_base="https://api.deepseek.com",
    temperature=0
)

def build_vectorstore_from_docs(chunks:list) -> Chroma:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    vectorstore = Chroma.from_documents(chunks, embeddings)
    return vectorstore