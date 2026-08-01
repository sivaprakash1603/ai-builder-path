from langchain_core.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from tools import it_tools, finance_tools
from pydantic import BaseModel
import os
from typing import Literal
from langgraph.prebuilt import create_react_agent

# Initialize the Anthropic LLM
llm = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0, anthropic_api_url=os.getenv("ANTHROPIC_BASE_URL"), api_key=os.getenv("ANTHROPIC_API_KEY"))

# ==========================================
# 1. IT Agent Setup
# ==========================================
it_system_prompt = (
    "You are a helpful IT support agent. You handle all IT-related queries such as software, VPN, hardware, and access requests.\n"
    "You have access to internal IT documents ('data/it_docs.txt') via the read_file_tool and the web via web_search.\n"
    "Always try to use internal documents first. If the answer is not in internal docs, you may search the web."
)
it_agent = create_react_agent(llm, tools=it_tools, prompt=it_system_prompt)

# ==========================================
# 2. Finance Agent Setup
# ==========================================
finance_system_prompt = (
    "You are a helpful Finance support agent. You handle all finance-related queries such as reimbursements, budgets, and payroll.\n"
    "You have access to internal finance documents ('data/finance_docs.txt') via the read_file_tool and the web via web_search.\n"
    "Always try to use internal documents first. If the answer is not in internal docs, you may search the web."
)
finance_agent = create_react_agent(llm, tools=finance_tools, prompt=finance_system_prompt)

# ==========================================
# 3. Supervisor Agent Setup
# ==========================================
members = ["IT", "Finance"]
options = ["FINISH"] + members

class RouteResponse(BaseModel):
    next: Literal["IT", "Finance", "FINISH"]

supervisor_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a supervisor tasked with managing a conversation between the following workers: {members}.\n"
               "Given the following user request and conversation history, decide who should act next.\n"
               "If it's an IT question (VPN, laptop, software), route to IT.\n"
               "If it's a Finance question (payroll, reimbursement, budget), route to Finance.\n"
               "If the user's question has been fully answered by a worker, respond with FINISH."),
    ("placeholder", "{messages}")
]).partial(members=", ".join(members))

supervisor_agent = supervisor_prompt | llm.with_structured_output(RouteResponse)
