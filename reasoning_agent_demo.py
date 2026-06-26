import langgraph
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()
from langchain_neo4j import Neo4jGraph
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser
from typing import List
from pydantic import BaseModel, Field
from typing import List
from pydantic import BaseModel
neo4j_graph = Neo4jGraph(
    url="bolt://localhost:7687",
    username="neo4j",
    password="jir000444333"
)   

llm = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key="DEEPSEEK_API_KEY",
    openai_api_base="https://api.deepseek.com",
    temperature=0                   
)




class Summary(BaseModel):
    chief_complaint: Optional[str] = Field(None, description="主诉") 
    present_illness: Optional[str] = Field(None, description="现病史")
    past_history: Optional[str] = Field(None, description="既往史")
    medications: Optional[str] = Field(None, description="用药史")
    labs: Optional[str] = Field(None, description="检验结果")
    vitals: Optional[str] = Field(None, description="生命体征")

class Factor(BaseModel):
    symptoms: List[str] = Field(..., description="主要症状")
    symptom_features: List[str] = Field(..., description="症状特征")
    risk_factors: List[str] = Field(..., description="危险因素")
    systems: List[str] = Field(..., description="可能涉及系统")
    candidate_diseases: List[str] = Field(..., description="疾病方向")

class DiseaseHypothesis(BaseModel):
    disease_name: str = Field(..., description="疾病名称")
    probability: float = Field(..., description="初步概率0-1")
    supporting_features: List[str] = Field(
        default_factory=list,
        description="支持该疾病的临床特征"
    )


class Hypothesis(BaseModel):
    hypotheses: List[DiseaseHypothesis]




class EvidenceItem(BaseModel):
    disease_name: str
    supporting_evidence: List[str]
    contradicting_evidence: List[str]
    missing_evidence: List[str]
    evidence_strength: float

class Evidence(BaseModel):
    evidence_results: List[EvidenceItem]


class RankedDiagnosis(BaseModel):

    disease_name: str

    confidence: float

    rationale: str

    recommended_tests: List[str]

    urgency: Literal[
        "low",
        "medium",
        "high"
    ]
class Diagnosis(BaseModel):

    ranked_diagnoses: List[RankedDiagnosis]







from typing import TypedDict

class CDSSState(TypedDict):
    description: str

    summary: Summary
    factor: Factor
    hypothesis: Hypothesis
    evidence: Evidence
    diagnosis: Diagnosis
    rag_context:str






def diagnosis(state:CDSSState):
    summary_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        你是一名病历结构化助手。

        从病历中提取：

        chief_complaint 
        present_illness
        past_history
        medications
        labs
        vitals

        只返回JSON。
        """
    ),
    ("human", "{description}")
    ])
    #result = llm.invoke(summary_prompt.format_messages(description = description)).content
    factor_llm = llm.with_structured_output(Summary)
    chain = summary_prompt | factor_llm

    return chain.invoke({
        "description": description.model_dump()  
    })


def factor(state:CDSSState)-> Factor:
    
    factor_prompt = ChatPromptTemplate.from_messages([
(
"system",
"""
你是一名资深内科医生。

分析病历信息。

返回：

{
  "symptoms":[],
  "symptom_features":[],
  "risk_factors":[],
  "systems":[],
  "candidate_diseases":[]
}

不要给最终诊断。
只返回JSON。
"""
),
("human","{summary}")
])
    factor_llm = llm.with_structured_output(Factor)
    chain = factor_prompt | factor_llm

    return chain.invoke({
        "summary": state["summary"].model_dump()   
    })



    

def hypothesis(state:CDSSState) -> Hypothesis:
    hypothesis_prompt = ChatPromptTemplate.from_messages([
(
"system",
"""
你是一名经验丰富的内科医生。

任务:

根据患者症状、症状特征、危险因素和受累系统，
生成鉴别诊断列表（Differential Diagnosis）。

要求：

1. 给出5个以内最可能的疾病方向
2. 不要给最终诊断
3. 每个疾病说明为什么考虑
4. 概率为相对概率(0~1)
5. 只依据提供的信息推理
6. 不允许编造不存在的症状

输出JSON。
"""
),
(
"human",
"{factor}"
)
])
    hypothesis_llm = llm.with_structured_output(Hypothesis)
    
    chain = hypothesis_prompt | hypothesis_llm

    return chain.invoke({
        "factor": state["factor"].model_dump()   
    })


def rag_verify(state:CDSSState) -> Evidence:
    evidence_prompt = ChatPromptTemplate.from_messages([
(
"system",
"""
你是临床决策支持系统(CDSS)中的证据评估模块。

任务：

针对每个候选疾病：

1. 找出支持证据，保证逻辑链清晰
2. 找出反对证据
3. 找出缺失证据
4. 只能依据提供的患者信息和参考文献
5. 不允许编造事实
6. 不允许给最终诊断

返回JSON。
"""
),
(
"human",
"""
患者分析：

{factor}

候选疾病：

{hypothesis}

医学参考资料：

{rag_context}
"""
)
])
    evidence_llm = llm.with_structured_output(Evidence)

    chain = evidence_prompt | evidence_llm

    return chain.invoke ({
        "factor": state["factor"].model_dump(),
        "hypothesis" : state["hypothesis"].model_dump(),
        "rag_context" : state["rag_context"]
    })



def rank(state:CDSSState) -> Diagnosis:
    diagnosis_prompt = ChatPromptTemplate.from_messages(
        [   
            (
                "system",
                """
你是一名临床决策支持系统(CDSS)中的诊断排序模块。

任务：

根据提供的证据分析结果，对候选疾病进行排序。

要求：

1. 综合支持证据、反对证据和缺失证据
2. 为每个疾病生成最终置信度(0~1)
3. 按置信度从高到低排序
4. 给出主要诊断依据(rationale)
5. 给出建议检查(recommended_tests)
6. 给出紧急程度(urgency)

紧急程度只能取：

- low
- medium
- high

规则：

- 支持证据越多，置信度越高
- 反对证据越多，置信度越低
- 缺失关键证据时降低置信度
- 不允许编造不存在的症状
- 不允许编造检查结果
- 不允许给出治疗方案

返回Diagnosis结构。
                """
            ),
            (
                "human",
                """
证据分析结果：

{evidence}
                """
            )
        ]
    )

    diagnosis_llm = llm.with_structured_output(Diagnosis)
    chain = diagnosis_prompt | diagnosis_llm

    return chain.invoke({
        "evidence": state["evidence"].model_dump()
    })


from langgraph.graph import END, START, StateGraph
workflow = StateGraph(CDSSState)

workflow.add_node("diagonosis",diagnosis)
workflow.add_node("factor",factor)
workflow.add_node("hypothesis",hypothesis)
workflow.add_node("rag_verify",rag_verify)
workflow.add_node("rank",rank)

workflow.set_entry_point("diagonosis")

workflow.add_edge("diagonosis","factor")
workflow.add_edge("factor","hypothesis")
workflow.add_edge("hypothesis","rag_verify")
workflow.add_edge("rag_verify","rank")
workflow.add_edge("rank",END)

graph = workflow.compile()

