import agents.retrieval.retrieval_agent as retrieval_agent
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage



def one_time_answer(query:str) :
    result = retrieval_agent.graph.invoke({ "messages": [HumanMessage(content=query)],
                        "session_id": "test_session",
                        "docs": [],
                       "retry_count": 0
        },
                         config = config)
    return result["messages"][-1].content