import agentic_rag
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage

user_id = "user_A"
session_id = "chat_1"

config = {"configurable": {"thread_id": "{user_id}:{session_id}"}}

while True:
    user_input = input("你：")
    if user_input == "exit":
        break
    result = agentic_rag.graph.invoke({ "messages": [HumanMessage(content=user_input)],
                        "session_id": "test_session",
                        "docs": [],
                       "retry_count": 0
       },
                         config = config)
    
    print(result["messages"][-1].content)
