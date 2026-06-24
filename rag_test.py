import query_transformation
from query_transformation import rewrite_query, build_query_rewriting_chain,llm
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader, TextLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.retrievers import EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# import the necessary packages
import easyocr
from easyocr import Reader
import argparse
import cv2
from pdf2image import convert_from_path
import numpy as np
from langchain.schema import Document

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder


from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
import os

load_dotenv()

# 1. 加载文档
#loader = TextLoader("test.txt", encoding="utf-8")
loader =  {"test.pdf": PyPDFLoader("test.pdf"), "test.txt": TextLoader("test.txt")}
docs = []
for loader in loader.values(): 
 docs.extend(loader.load())



pages = convert_from_path("test2.pdf")
reader = easyocr.Reader(['ch_sim', 'en'])
for page in pages:
    page_array = np.array(page)
    result = reader.readtext(page_array)
   # print(result)
    docs.append(Document(page_content=" ".join([line[1] for line in result]), metadata={"source": "test2.pdf"}))

# 2. 分块
#chunk_overlap 以少量冗余信息换取连贯语义
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# 3. Embedding 用 huggingface
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)




# 4. 存入 Chroma
#vectorstore 绑定了embedding模型

#chroma 语义 检索，bm25 关键词检索

#存入chroma
vectorstore = Chroma.from_documents(chunks, embeddings)

#去重（保留原文档的metadata），整合潜在逻辑，并将结果存入vectorstore
#去重逻辑：如果两个chunk的文本内容相似度超过0.9，则认为它们是重复的，保留其中一个，并将它们的metadata合并
"""from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
def deduplicate_chunks(chunks,threshold= 0.9):
    unique_chunks = [] #保存去重后的chunk
    for chunk in chunks:
        is_duplicate = False
        for unique_chunk in unique_chunks:
            similarity= cosine_similarity([chunk.page_content], [unique_chunk.page_content])[0][0]
            if similarity > threshold:
                is_duplicate = True
                unique_chunk.metadata.update(chunk.metadata)
                break
        if not is_duplicate:
            unique_chunks.append(chunk)
    return unique_chunks

vectorstore = Chroma.from_documents(deduplicate_chunks(chunks), embeddings)

#小chunk的文本内容相似度超过0.9,则认为其所在的长chunk之间有潜在逻辑关联。"""




#构建语义检索器
semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
 
#存入bm25
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 3

#构建混合检索器（各占50%权重）
ensemble_retriever = EnsembleRetriever(
    retrievers=[semantic_retriever, bm25_retriever],
    weights=[0.5, 0.5]
)


reranker_model = HuggingFaceCrossEncoder(model_name = "BAAI/bge-reranker-base")
compressor = CrossEncoderReranker(model = reranker_model,top_n = 3)
#构建上下文压缩检索器
reranking_retriever = ContextualCompressionRetriever(base_compressor = compressor, base_retriever = ensemble_retriever)



# 5. LLM 用 DeepSeek
llm = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    openai_api_base="https://api.deepseek.com"
)




#回溯聊天历史逻辑：
prompt1 = ChatPromptTemplate.from_messages([("system", "根据聊天历史，将用户的问题改写成独立完整的查询。"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")])
history_aware_retriever = create_history_aware_retriever(llm,
    reranking_retriever,
    prompt1
)

prompt2 = ChatPromptTemplate.from_messages([("system", "你是一个问答助手，根据以下文档回答问题：\n\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")])
combine_docs_chain = create_stuff_documents_chain(llm, prompt2)                
rag_chain = create_retrieval_chain(history_aware_retriever, combine_docs_chain)

store = {}
#提高调用效率，避免每次调用都创建新的 ChatMessageHistory 实例
def get_session_history(session_id: str) :
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

#每次调用RunnableWithMessageHistory都会更新store[session_id]
conversational_rag_chain = RunnableWithMessageHistory(rag_chain, get_session_history, input_messages_key="input", history_messages_key="chat_history", output_messages_key="answer")







# 提问
result = conversational_rag_chain.invoke({"input": rewrite_query("文档的主要内容是什么？", build_query_rewriting_chain(llm))}, config = {"configurable": {"session_id": "test_session"}})
print(result)






















# 6. 构建 RAG 链
#qa = RetrievalQA.from_chain_type(
   # llm=llm,
    #vectorstore.as_retriever()创建了检索器，其内部已经绑定了创建 vectorstore 时用的 embedding 模型
   # retriever=reranking_retriever
#)









#loader = PyPDFLoader("test.pdf")

#retriever = vectorstore.as_retriever(search_kwargs={"k": 3})