import os
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

@tool
def read_file_tool(file_path: str) -> str:
    """Read the contents of a local file. Use this to read internal documentation.
    Available files: 
    - 'data/it_docs.txt' (for IT queries)
    - 'data/finance_docs.txt' (for Finance queries)
    Provide the exact relative file path."""
    try:
        if not os.path.exists(file_path):
            return f"Error: File '{file_path}' does not exist."
        with open(file_path, "r") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

# Instantiate the web search tool
web_search_tool = DuckDuckGoSearchRun(
    name="web_search",
    description="A wrapper around DuckDuckGo Search. Useful for searching the web for public information, external sources, or when internal docs don't have the answer."
)

# Lists of tools for the specialist agents
it_tools = [read_file_tool, web_search_tool]
finance_tools = [read_file_tool, web_search_tool]
