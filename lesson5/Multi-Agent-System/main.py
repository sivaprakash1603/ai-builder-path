import sys
import os
from dotenv import load_dotenv

# Load environment variables, including ANTHROPIC_API_KEY
load_dotenv()

from langchain_core.messages import HumanMessage
from graph import graph

def run_query(query: str):
    print(f"\n{'='*50}\nQuery: {query}\n{'='*50}\n")
    
    # We pass the initial state with the user's query
    initial_state = {
        "messages": [HumanMessage(content=query)]
    }
    
    # Run the graph and stream the output to see the steps
    try:
        for event in graph.stream(initial_state, {"recursion_limit": 10}):
            for node_name, state_update in event.items():
                print(f"\n--- Node Executed: {node_name} ---")
                if "messages" in state_update:
                    for message in state_update["messages"]:
                        sender = getattr(message, 'name', None) or message.type
                        print(f"[{sender}]: {message.content}")
                if "next" in state_update:
                    print(f"[Supervisor Route]: {state_update['next']}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Ensure ANTHROPIC_API_KEY is present
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable not found. Please set it or add it to a .env file.")
        sys.exit(1)

    if len(sys.argv) > 1:
        query = sys.argv[1]
        run_query(query)
    else:
        print("Running default test queries...\n")
        run_query("How to set up VPN?")
        run_query("When is payroll processed?")
