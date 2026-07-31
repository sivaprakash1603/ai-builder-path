import asyncio
import os
from langchain_core.tools import Tool
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Define the server parameters to run mcp_server.py
server_params = StdioServerParameters(
    command="python3",
    args=[os.path.join(os.path.dirname(__file__), "..", "mcp_server.py")]
)

async def _read_doc_async(document_id: str) -> str:
    """Connect to the MCP server and call the read_google_doc tool."""
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize the connection
                await session.initialize()
                
                # Call the tool exposed by the MCP server
                result = await session.call_tool("read_google_doc", arguments={"document_id": document_id})
                
                # Process the result content
                if result and result.content:
                    return "\n".join([item.text for item in result.content if item.type == "text"])
                return "No content returned."
    except Exception as e:
        return f"MCP Client Error: {str(e)}"

def read_doc_sync(document_id: str) -> str:
    """Synchronous wrapper to call the MCP tool."""
    return asyncio.run(_read_doc_async(document_id))

# Create the LangChain tool wrapper
mcp_google_docs_tool = Tool(
    name="Google_Docs_Reader",
    description="Use this tool to read Presidio insurance documents from Google Docs. You must provide the Google Document ID (from the URL).",
    func=read_doc_sync
)
