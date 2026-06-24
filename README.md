# Project Name
Agentic Rag + RAG pipeline

## Introduction
Agentic Rag: Combined agentic RAG with knowledge graph, guaranteed a flexible decision process, while also maintained solid answers.  
Applied OCR to ingest PDF file.

RAG pipeline: Fundamental RAG pipeline, applied Chromadb as vector database. 

## Architecture
(项目结构图或流程图)

Agentic RAG
https://lucid.app/lucidchart/12d40e8b-1275-4c9f-a114-7084d46e322b/edit?viewport_loc=-113%2C-976%2C2888%2C1454%2C0_0&invitationId=inv_3f4c0a5a-665a-43f3-afaa-53f3b991972f

RAG pipeline
https://lucid.app/lucidchart/4389089a-5c61-447c-8962-579e6d1d4fe1/edit?viewport_loc=-1573%2C767%2C1836%2C1058%2C0_0&invitationId=inv_314adb4f-b763-4ca0-a9c3-74a4be885d5a


## How to Start
neo4j start
python driver.py

## Project File Composition
my_langchain_rag_project/
├── agentic_rag.py           #main agentic process
├── build_graph.py           #building knowledge graph
├── build_vectorstore.py     #embedding process
├── my_state.py              #self-built state 
├── traces.jsonl             #traces
├── driver.py               
├── rag_test.py              #RAG pipeline
├── query_transformation.py  #rewrite input
├── fudan_case.pdf
├── agentic_rag_demo.py
├── .env
├── _pycache_/
└── venv/


## Technique Stack 
- LangChain
- LangGraph
- Neo4j
- DeepSeek
- HuggingFace
- Chroma
- BM25
- OCR


## Example:
query1: 凶手是谁
answer: 根据文档信息，凶手是**林森浩**。他是故意杀人案罪犯，涉及投毒行为，使用二甲基亚硝胺，导致被害人黄某死亡，并被判决故意杀人罪。

query2: 凶手使用了什么杀人方式  
answer: 根据知识图谱中的信息，林森浩使用的杀人方式是**投毒**。具体来说，他使用了**二甲基亚硝胺**（一种化学毒物）进行投毒，导致被害人黄某中毒死亡。







完整思路：
use llm to extract key words/key logics from existed text

-- search on web，store webpages(url) in a list

-- grade and get top k webpage

-- transform into chunks

-- extend the knowledge graph (添加filter功能)









