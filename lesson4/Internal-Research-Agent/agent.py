import os
from dotenv import load_dotenv

# Load environment variables (like ANTHROPIC_API_KEY)
load_dotenv()

from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent

# Import our custom tools
from tools.rag_tool import rag_tool
from tools.search_tool import web_search_tool
from tools.mcp_tool_wrapper import mcp_google_docs_tool

SYSTEM_PROMPT = (
    "You are a highly capable Internal Research Agent for Presidio. "
    "Your job is to answer queries by using your available tools.\n\n"
    "You have access to:\n"
    "1. Internal_Knowledge_Search: Use this for internal HR policies, compliance rules, marketing feedback, and company metrics.\n"
    "2. Web_Search: Use this for external industry benchmarks, trends, and news.\n"
    "3. Google_Docs_Reader: Use this to read specific Google Docs for insurance-related queries.\n\n"
    "Always cite which tool you used to get the information."
)

def create_agent():
    """Initializes and returns the Internal Research Agent."""
    # Check for either the standard key or the Presidio auth token
    api_key = os.getenv("ANTHROPIC_AUTH_TOKEN") or os.getenv("ANTHROPIC_API_KEY")
    if not api_key or api_key == "your_anthropic_api_key_here":
        raise ValueError("Please set your ANTHROPIC_AUTH_TOKEN or ANTHROPIC_API_KEY.")
        
    llm = ChatAnthropic(
        model="claude-sonnet-4-5-20250929", 
        temperature=0,
        anthropic_api_url=os.getenv("ANTHROPIC_BASE_URL"),
        api_key=api_key
    )
    
    tools = [rag_tool, web_search_tool, mcp_google_docs_tool]
    
    agent_executor = create_react_agent(llm, tools)
    
    return agent_executor

if __name__ == "__main__":
    print("🤖 Initializing Presidio Internal Research Agent...")
    try:
        agent = create_agent()
        print("\nAgent is ready! (Note: Testing Google Docs requires a valid credentials.json and Document ID)")
        print("-" * 50)
        
        queries = [
            "Summarize all customer feedback related to our Q1 marketing campaigns.",
            "Compare our current hiring trend with industry benchmarks.",
            "Find relevant compliance policies related to AI data handling."
        ]
        
        for q in queries:
            print(f"\n🗣️ User Query: {q}")
            response = agent.invoke({"messages": [("system", SYSTEM_PROMPT), ("user", q)]})
            final_message = response["messages"][-1].content
            print(f"🤖 Agent Response:\n{final_message}\n")
            print("-" * 50)
            
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
