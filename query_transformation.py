
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import PromptTemplate

import os
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Set the OpenAI API key environment variable
#os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain


# LLM configuration
#def get_rewriting_llm(model_name: str = "deepseek-chat", temperature: float = 0, max_tokens: int = 4000) -> ChatOpenAI:
   # return ChatOpenAI(temperature=temperature, model_name=model_name, max_tokens=max_tokens)
llm = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    openai_api_base="https://api.deepseek.com"
)

# Prompt Template for Query Rewriting
def create_query_rewrite_prompt() -> PromptTemplate:
    prompt = ChatPromptTemplate.from_messages([("You are an AI assistant tasked with reformulating "
    "user queries to improve retrieval in a RAG system. Given the original query, rewrite it to be more specific, detailed, "
    "and likely to retrieve relevant information.Original query: {original_query} Rewritten query:")
    ])
    return prompt

# Build Query Rewriting Chain
def build_query_rewriting_chain(llm: ChatOpenAI) -> LLMChain:
    prompt_template = create_query_rewrite_prompt()
    return prompt_template | llm

# Function to Rewrite the Query
def rewrite_query(original_query: str, query_rewriter_chain: LLMChain) -> str:
    response = query_rewriter_chain.invoke(original_query)
    return response.content.strip()
"""if __name__ == "__main__":  
   # llm = get_rewriting_llm()
    query_rewriter_chain = build_query_rewriting_chain(llm)
    original_query = "What are the impacts of climate change on the environment?"
    rewritten_query = rewrite_query(original_query, query_rewriter_chain)
    print("Original query:", original_query)
    print("\nRewritten query:", rewritten_query)"""

""" OUTPUT

Original query: What are the impacts of climate change on the environment?

Rewritten query: How does climate change affect various aspects of the environment, such as ecosystems, biodiversity, weather patterns, and sea levels?

"""