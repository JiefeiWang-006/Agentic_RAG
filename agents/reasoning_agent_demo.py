import langgraph
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_neo4j import Neo4jGraph
from typing import List, Optional, Dict, Any
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser
from typing import List
from typing import Literal
neo4j_graph = Neo4jGraph(
    url="bolt://localhost:7687",
    username="neo4j",
    password="jir000444333"
)   

llm = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
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



class Reflection(BaseModel):
    consistency: bool
    missing_diseases: List[str]
    reasoning_issues: List[str]
    additional_tests: List[str]
    confidence_adjustment: Optional[str]
    



from typing import TypedDict



class CDSSState(TypedDict):
    description: str
    summary: Optional[Summary]
    factor: Optional[Factor]
    hypothesis: Optional[Hypothesis]
    evidence: Optional[Evidence]
    reflection: Optional[Reflection]
    diagnosis: Optional[Diagnosis]
    rag_context: str
    report: str







def diagnosis(state:CDSSState):
    summary_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        你是一名病历结构化助手。必须以 JSON 格式输出结果。

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
    factor_llm = llm.with_structured_output(Summary,method="json_mode")
    chain = summary_prompt | factor_llm

    summary = chain.invoke({
        "description": state["description"]
    })
    return {
        "summary": summary
    }


def factor(state:CDSSState)-> Factor:
    
    factor_prompt = ChatPromptTemplate.from_messages([
(
"system",
"""
你是一名资深内科医生。必须以 JSON 格式输出结果。

分析病历信息。

返回：

{{
  "symptoms":[],
  "symptom_features":[],
  "risk_factors":[],
  "systems":[],
  "candidate_diseases":[]
}}

不要给最终诊断。
只返回JSON。
"""
),
("human","{summary}")
])
    
    factor_llm = llm.with_structured_output(Factor, method="json_mode")
    chain = factor_prompt | factor_llm

    factor = chain.invoke({
        "summary": state["summary"].model_dump()   
    })
    return {
        "factor":factor
    }


    

def hypothesis(state:CDSSState) -> Hypothesis:
    hypothesis_prompt = ChatPromptTemplate.from_messages([
(
"system",
"""
你是一名经验丰富的内科医生。必须以 JSON 格式输出结果。

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

输出格式：
{{
  "hypotheses": [
    {{
      "disease_name": "疾病名称",
      "probability": 0.0,
      "supporting_features": ["支持特征1", "支持特征2"]
    }}
  ]
}}

"""
),
(
"human",
"{factor}"
)
])
    hypothesis_llm = llm.with_structured_output(Hypothesis, method="json_mode")
    
    chain = hypothesis_prompt | hypothesis_llm

    hypothesis = chain.invoke({
        "factor": state["factor"].model_dump()   
    })
    return {
        "hypothesis":hypothesis
    }
def rag_retrieve(state:CDSSState) -> str:
    hypotheses = state["hypothesis"].hypotheses
    disease_list = "、".join([h.disease_name for h in hypotheses])
    rag_context_prompt = f"""
请针对以下候选疾病，分别检索并提供：
1. 诊断标准
2. 典型临床表现
3. 鉴别诊断要点

候选疾病：{disease_list}

患者主要症状：{state["description"]}
"""
    import retrieval_agent
    config = {"configurable": {"thread_id": "user_A:chat_2"}}
    result = retrieval_agent.graph.invoke({ "messages": [HumanMessage(content=rag_context_prompt)],
                        "session_id": "test_session",
                        "docs": [],
                        "retry_count": 0
        },
                         config = config)
    return {
        "rag_context":result["messages"][-1].content
    }

def rag_verify(state:CDSSState) -> Evidence:
    evidence_prompt = ChatPromptTemplate.from_messages([
(
"system",
"""
你是临床决策支持系统(CDSS)中的证据评估模块。必须以 JSON 格式输出结果。

任务：

针对每个候选疾病：

1. 找出支持证据，保证逻辑链清晰
2. 找出反对证据
3. 找出缺失证据
4. 只能依据提供的患者信息和参考文献
5. 不允许编造事实
6. 不允许给最终诊断

返回格式：

{{
  "evidence_results": [
    {{
      "disease_name": "疾病名称",
      "supporting_evidence": ["支持证据1", "支持证据2"],
      "contradicting_evidence": ["反对证据1", "反对证据2"],
      "missing_evidence": ["缺失证据1", "缺失证据2"],
      "evidence_strength": 0.0
    }}
  ]
}}

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
    evidence_llm = llm.with_structured_output(Evidence, method="json_mode")

    chain = evidence_prompt | evidence_llm

    evidence = chain.invoke ({
        "factor": state["factor"].model_dump(),
        "hypothesis" : state["hypothesis"].model_dump(),
        "rag_context" : state["rag_context"]
    })
    return {
        "evidence":evidence
    }


def reflection(state:CDSSState) -> Reflection:
    reflection_prompt = ChatPromptTemplate.from_messages([
(
    "system",
    """
你是一名临床决策支持系统（CDSS）中的推理审查模块（Reflection Agent）。必须以 JSON 格式输出结果。

你的任务不是重新诊断患者，而是审查已有的推理过程是否存在问题。

请依据：

1. 患者病历摘要（Summary）
2. 临床特征（Factor）
3. 鉴别诊断（Hypothesis）
4. 证据分析（Evidence）

对整个推理链进行质量审查。

请重点检查：

1. 是否存在推理漏洞或逻辑矛盾。
2. 是否遗漏了重要或高风险疾病。
3. 当前证据是否足以支持各候选疾病。
4. 是否存在关键缺失证据影响诊断可信度。
5. 是否建议补充检查以提高诊断确定性。
6. 不允许编造患者不存在的症状、体征或检查结果。
7. 不允许给出最终诊断或治疗建议。

返回形式：
{{
  "consistency": true,
  "missing_diseases": ["疾病1", "疾病2"],
  "reasoning_issues": ["问题1", "问题2"],
  "additional_tests": ["检查1", "检查2"],
  "confidence_adjustment": "调整说明或null"
}}

"""
),
(
    "human",
    """
患者病历摘要：

{summary}

临床特征：

{factor}

鉴别诊断：

{hypothesis}

证据分析：

{evidence}
"""
)
])

    reflection_llm = llm.with_structured_output(Reflection, method="json_mode")
    chain = reflection_prompt | reflection_llm

    reflection = chain.invoke({
        "evidence": state["evidence"].model_dump(),
        "summary":state["summary"].model_dump(),
        "factor":state["factor"].model_dump(),
        "hypothesis":state["hypothesis"].model_dump()
    })
    return {
        "reflection" : reflection
    }



def rank(state:CDSSState) -> Diagnosis:
    diagnosis_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
你是一名临床决策支持系统（CDSS）中的诊断排序模块。必须以 JSON 格式输出结果。

任务：

根据证据分析结果（Evidence）和推理审查结果（Reflection），
对候选疾病进行最终排序。

要求：

1. 综合支持证据、反对证据和缺失证据。
2. 结合Reflection中的推理审查意见，对各候选疾病的置信度进行合理调整。
3. 为每个疾病生成最终置信度（0~1）。
4. 按置信度从高到低排序。
5. 给出主要诊断依据（rationale）。
6. 给出建议检查（recommended_tests）。
7. 给出紧急程度（urgency）。

紧急程度只能取：

- low
- medium
- high

规则：

- 支持证据越多，置信度越高。
- 反对证据越多，置信度越低。
- 缺失关键证据时降低置信度。
- 如果Reflection指出推理存在逻辑漏洞、关键证据缺失或诊断依据不足，应适当降低相应疾病的置信度。
- 如果Reflection提示存在需要优先排除的高风险疾病，应在排序中予以充分考虑，并在诊断依据中说明原因。
- Reflection仅用于评估推理质量，不允许据此编造新的患者症状、体征、检查结果或医学事实。
- 不允许编造不存在的症状。
- 不允许编造检查结果。
- 不允许给出治疗方案。

返回Diagnosis结构。
返回格式：

{{
    "ranked_diagnoses": [
    {{
        "disease_name": [],
        "confidence": [],
        "rationale": [],
        "recommended_tests": [],
        "urgency": "low 或 medium 或 high"
    }}
    ]
}} 
"""
            ),
            (
                "human",
                """
                    证据分析结果：

{evidence}

推理审查结果：

{reflection}
                """
            )
        ]
    )

    diagnosis_llm = llm.with_structured_output(Diagnosis, method="json_mode")
    chain = diagnosis_prompt | diagnosis_llm

    diagnosis = chain.invoke({
        "evidence": state["evidence"].model_dump(),
        "reflection":state["reflection"].model_dump()
    })
    return {
        "diagnosis" : diagnosis
    }


def format_report(state: CDSSState) -> dict:
    s = state
    lines = []

    # ── 基本信息 ──────────────────────────────────────────
    lines.append("=" * 60)
    lines.append("         临床决策支持报告（CDSS）")
    lines.append("=" * 60)
    lines.append(f"主诉：{s['description']}")
    lines.append("")

    # ── 病历摘要 ──────────────────────────────────────────
    summary = s["summary"]
    lines.append("【病历摘要】")
    lines.append(f"  主诉：        {summary.chief_complaint or '未提供'}")
    lines.append(f"  现病史：      {summary.present_illness or '未提供'}")
    lines.append(f"  既往史：      {summary.past_history or '未提供'}")
    lines.append(f"  用药史：      {summary.medications or '未提供'}")
    lines.append(f"  检验结果：    {summary.labs or '未提供'}")
    lines.append(f"  生命体征：    {summary.vitals or '未提供'}")
    lines.append("")

    # ── 症状分析 ──────────────────────────────────────────
    factor = s["factor"]
    lines.append("【症状分析】")
    lines.append(f"  症状：        {', '.join(factor.symptoms) or '未提供'}")
    lines.append(f"  症状特征：    {', '.join(factor.symptom_features) or '未提供'}")
    lines.append(f"  危险因素：    {', '.join(factor.risk_factors) or '未提供'}")
    lines.append(f"  受累系统：    {', '.join(factor.systems) or '未提供'}")
    lines.append("")

    # ── 鉴别诊断 ──────────────────────────────────────────
    lines.append("【鉴别诊断】")
    for h in s["hypothesis"].hypotheses:
        lines.append(f"  • {h.disease_name}（概率 {h.probability:.0%}）")
        lines.append(f"    支持特征：{', '.join(h.supporting_features)}")
    lines.append("")

    # ── 证据评估 ──────────────────────────────────────────
    lines.append("【证据评估】")
    for e in s["evidence"].evidence_results:
        lines.append(f"  ▌ {e.disease_name}（证据强度 {e.evidence_strength:.0%}）")
        lines.append(f"    支持：  {', '.join(e.supporting_evidence)}")
        lines.append(f"    反对：  {', '.join(e.contradicting_evidence)}")
        lines.append(f"    缺失：  {', '.join(e.missing_evidence)}")
    lines.append("")

    # ── 推理反思 ──────────────────────────────────────────
    ref = s["reflection"]
    lines.append("【推理反思】")
    lines.append(f"  推理一致性：  {'✓ 一致' if ref.consistency else '✗ 存在问题'}")
    if ref.missing_diseases:
        lines.append(f"  遗漏疾病：    {', '.join(ref.missing_diseases)}")
    if ref.reasoning_issues:
        lines.append("  推理问题：")
        for issue in ref.reasoning_issues:
            lines.append(f"    - {issue}")
    if ref.additional_tests:
        lines.append(f"  建议检查：    {', '.join(ref.additional_tests)}")
    if ref.confidence_adjustment:
        lines.append(f"  置信度说明：  {ref.confidence_adjustment}")
    lines.append("")

    # ── 最终诊断排名 ──────────────────────────────────────
    lines.append("【最终诊断建议】")
    urgency_map = {"high": "🔴 紧急", "medium": "🟡 中等", "low": "🟢 常规"}
    for i, d in enumerate(s["diagnosis"].ranked_diagnoses, 1):
        urgency = urgency_map.get(d.urgency, d.urgency)
        lines.append(f"  {i}. {d.disease_name}（置信度 {d.confidence:.0%}）{urgency}")
        lines.append(f"     {d.rationale}")
        lines.append(f"     建议检查：{', '.join(d.recommended_tests)}")
    lines.append("")
    lines.append("=" * 60)
    lines.append("⚠️  本报告仅供临床参考，最终诊断以医生判断为准。")
    lines.append("=" * 60)

    report = "\n".join(lines)
    print(report)
    return {"report": report}



from langgraph.graph import END, START, StateGraph
workflow = StateGraph(CDSSState)

workflow.add_node("diagnosis",diagnosis)
workflow.add_node("factor",factor)
workflow.add_node("hypothesis",hypothesis)
workflow.add_node("rag_context",rag_retrieve)
workflow.add_node("rag_verify",rag_verify)
workflow.add_node("reflection",reflection)
workflow.add_node("rank",rank)
workflow.add_node("format_report", format_report)

workflow.set_entry_point("diagnosis")

workflow.add_edge("diagnosis","factor")
workflow.add_edge("factor","hypothesis")
workflow.add_edge("hypothesis","rag_context")
workflow.add_edge("rag_context","rag_verify")
workflow.add_edge("rag_verify","reflection")
workflow.add_edge("reflection","rank")
workflow.add_edge("rank", "format_report")
workflow.add_edge("format_report", END)

graph = workflow.compile()


result = graph.invoke( {"description": "胸痛","rag_context":" "})
                         

print(result["report"])


















