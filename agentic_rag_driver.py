import retrieval_agent
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage



config = {"configurable": {"thread_id": "user_A:chat_2"}}

while True:
    user_input = input("你：")
    if user_input == "exit":
        break  
    result = retrieval_agent.graph.invoke({ "messages": [HumanMessage(content=user_input)],
                        "session_id": "test_session",
                        "docs": [],
                       "retry_count": 0
        },
                         config = config)
                        
    print(result["messages"][-1].content)

def one_time_answer(query:str) :
    result = retrieval_agent.graph.invoke({ "messages": [HumanMessage(content=query)],
                        "session_id": "test_session",
                        "docs": [],
                       "retry_count": 0
        },
                         config = config)
    return result