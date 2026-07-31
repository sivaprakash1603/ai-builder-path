from langchain_core.tools import Tool
from langchain_community.tools import DuckDuckGoSearchRun

# Initialize the DuckDuckGo search tool
ddg_search = DuckDuckGoSearchRun()

def perform_search(query: str) -> str:
    """Use DuckDuckGo to search the web."""
    try:
        return ddg_search.run(query)
    except Exception as e:
        return f"Error performing web search: {str(e)}"

# Create the LangChain tool wrapper
web_search_tool = Tool(
    name="Web_Search",
    description="Use this tool to search the internet for external information such as industry benchmarks, market trends, and regulatory updates.",
    func=perform_search
)
