import operator
from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

from agents import it_agent, finance_agent, supervisor_agent

class AgentState(TypedDict):
    # The list of messages in the conversation
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # The 'next' field is populated by the supervisor
    next: str

# Node function for IT agent
def it_node(state: AgentState):
    # The prebuilt react agent expects {"messages": ...} and returns the updated state.
    # We only want to append the final message to the global state.
    result = it_agent.invoke({"messages": state["messages"]})
    # The last message is the final response from the agent
    final_message = result["messages"][-1]
    # Wrap it in an AIMessage with the name set to "IT" so the supervisor knows who spoke
    return {"messages": [AIMessage(content=final_message.content, name="IT")]}

# Node function for Finance agent
def finance_node(state: AgentState):
    result = finance_agent.invoke({"messages": state["messages"]})
    final_message = result["messages"][-1]
    return {"messages": [AIMessage(content=final_message.content, name="Finance")]}

# Node function for Supervisor
def supervisor_node(state: AgentState):
    result = supervisor_agent.invoke({"messages": state["messages"]})
    return {"next": result.next}

# Build the graph
workflow = StateGraph(AgentState)
workflow.add_node("IT", it_node)
workflow.add_node("Finance", finance_node)
workflow.add_node("Supervisor", supervisor_node)

# Workers always report back to the supervisor
workflow.add_edge("IT", "Supervisor")
workflow.add_edge("Finance", "Supervisor")

# Conditional edges from the supervisor
conditional_map = {
    "IT": "IT",
    "Finance": "Finance",
    "FINISH": END
}

workflow.add_conditional_edges("Supervisor", lambda x: x["next"], conditional_map)
workflow.set_entry_point("Supervisor")

# Compile the graph
graph = workflow.compile()
