import agentic_rag_driver
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()
from langchain_neo4j import Neo4jGraph

llm = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key="DEEPSEEK_API_KEY",
    openai_api_base="https://api.deepseek.com",
    temperature=0
)

#判断病症环节：
#1.调用retreval_agent, 返回所有可能deeper problems, 
description = ""
prompt = "搜索出所有可能导致以下症状的原因,这是description:{description}"
deeper_problems = agentic_rag_driver.one_time_answer(prompt)


#2.调用reasoning_agent,filter and analyze出最可能的deeper problems


#做出决策环节
