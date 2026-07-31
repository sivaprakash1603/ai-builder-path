import os
import sys
import json
from fastmcp import FastMCP
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Initialize FastMCP Server
mcp = FastMCP("GoogleDocsServer")

def get_docs_service():
    """Authenticate and return the Google Docs service."""
    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "credentials.json")
    if not os.path.exists(creds_path):
        raise ValueError(
            f"Google API credentials not found at {creds_path}. "
            "Please create a service account in Google Cloud Console, download the JSON key, and save it as credentials.json."
        )
    
    # We need readonly scope for docs
    SCOPES = ['https://www.googleapis.com/auth/documents.readonly']
    creds = service_account.Credentials.from_service_account_file(creds_path, scopes=SCOPES)
    return build('docs', 'v1', credentials=creds)

@mcp.tool()
def read_google_doc(document_id: str) -> str:
    """Read and extract text from a Google Document. Use this to read Presidio insurance documents.
    
    Args:
        document_id: The ID of the Google Doc (found in the URL: https://docs.google.com/document/d/<document_id>/edit)
    """
    try:
        service = get_docs_service()
        doc = service.documents().get(documentId=document_id).execute()
        
        # Extract text from the document
        content = doc.get('body').get('content')
        text = ""
        for element in content:
            if 'paragraph' in element:
                elements = element.get('paragraph').get('elements')
                for elem in elements:
                    if 'textRun' in elem:
                        text += elem.get('textRun').get('content')
        return text
    except HttpError as err:
        return f"Google API Error: {err}"
    except Exception as e:
        return f"Error reading document: {str(e)}"

if __name__ == "__main__":
    # Start the MCP server using stdio transport
    mcp.run(transport='stdio')
