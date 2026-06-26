from langchain_neo4j import Neo4jGraph
#from langchain_community.vectorstores import Neo4jVector
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import json

load_dotenv()
neo4j_graph = Neo4jGraph(
    url="bolt://localhost:7687",
    username="neo4j",
    password="jir000444333"
)
response_model = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key="sk-4c891358a2674840b2de0929394dd01d",
    openai_api_base="https://api.deepseek.com",
    temperature=0
)
def build_knowledge_graph(chunks:list) -> Neo4jGraph:
    for chunk in chunks:
        EXTRACT_PROMPT = """从以下刑侦案件文本中抽取实体和关系。

                       实体类型：人物、地点、事件、证据、法条、机构
                       关系类型：涉及、发生于、使用、导致、判决、属于

                       文本：{text}

                        只返回JSON,不要其他内容,
                       {{
                     "entities": [
                      {{"name": "实体名", "type": "实体类型", "description": "简短描述"}}
                        ],
                       "relationships": [
                      {{"source": "实体1", "relation": "关系", "target": "实体2"}}
                       ]
                       }}"""
        text = chunk.page_content
        response = response_model.invoke(EXTRACT_PROMPT.format(text=text))
        data = json.loads(response.content)
        for entity in data["entities"]:
           # name = entity["name"]
            neo4j_graph.query( """MERGE (e:Entity{name:$name})
                        SET e.type = $type, 
                        e.description = $description """,params = entity)
        for relationship in data["relationships"]:
            neo4j_graph.query("""MATCH (a:Entity {name:$source})
                           MATCH (b:Entity {name:$target})
                           MERGE (a)-[:RELATION{type: $relation}]->(b)""",params = relationship)
            
    return neo4j_graph





































