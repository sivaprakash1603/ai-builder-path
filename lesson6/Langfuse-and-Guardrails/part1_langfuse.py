import os
from dotenv import load_dotenv
load_dotenv()

from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langfuse.langchain import CallbackHandler
from langgraph.prebuilt import create_react_agent

# Initialize Langfuse Callback Handler
langfuse_handler = CallbackHandler()

# Initialize LLM
llm = ChatAnthropic(
    model="claude-sonnet-4-5-20250929",
    temperature=0,
    anthropic_api_url=os.getenv("ANTHROPIC_BASE_URL"),
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

@tool
def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    return f"The weather in {location} is 72°F and sunny."

tools = [get_weather]
system_prompt = "You are a helpful assistant. Use the provided tools to answer user questions."
agent = create_react_agent(llm, tools=tools, prompt=system_prompt)

def run_agent():
    print("Running Agent with Langfuse tracing...")
    query = "What is the weather like in San Francisco today?"
    
    # We pass the langfuse_handler into the config's callbacks list
    config = {"callbacks": [langfuse_handler]}
    
    result = agent.invoke({"messages": [("user", query)]}, config=config)
    
    print("\n" + "="*50)
    print(f"Final Answer: {result['messages'][-1].content}")
    print("="*50)
    print("\n✅ Execution complete. Check your Langfuse dashboard to see:")
    print("  - Token usage (input & output)")
    print("  - Prompts")
    print("  - Tool usage (get_weather)")

if __name__ == "__main__":
    if not os.getenv("LANGFUSE_SECRET_KEY") or os.getenv("LANGFUSE_SECRET_KEY").startswith("sk-lf-..."):
        print("Please configure your LANGFUSE_SECRET_KEY and other Langfuse keys in the .env file to run this script.")
    else:
        run_agent()
