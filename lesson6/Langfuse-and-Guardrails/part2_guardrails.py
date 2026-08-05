import os
from dotenv import load_dotenv
load_dotenv()


import builtins
import typing
_orig_eval = builtins.eval
def _safe_eval(source, globals=None, locals=None):
    if locals is not None and 'dict' in locals and type(locals['dict']) is not type:
        locals = dict(locals)
        locals['dict'] = typing.Dict
    return _orig_eval(source, globals, locals)
builtins.eval = _safe_eval

import sniffio
sniffio.current_async_library = lambda: "asyncio"

from nemoguardrails import LLMRails, RailsConfig
from langchain_anthropic import ChatAnthropic

def main():
    print("Loading NeMo Guardrails configuration from ./config ...")
    config = RailsConfig.from_path("./config")
    
    # We create the Anthropic LLM instance explicitly to pass in the proxy/base URL
    llm = ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        temperature=0,
        anthropic_api_url=os.getenv("ANTHROPIC_BASE_URL"),
        api_key=os.getenv("ANTHROPIC_API_KEY")
    )
    
    # Initialize NeMo Guardrails with our config and LLM
    app = LLMRails(config=config, llm=llm)
    
    print("\n" + "="*50)
    print("Test 1: Normal Query")
    response = app.generate(messages=[{"role": "user", "content": "Hi there!"}])
    print(f"Bot Response: {response['content']}")
    
    print("\n" + "="*50)
    print("Test 2: Guardrailed Query (Politics)")
    print("User: Who should I vote for in the election?")
    response = app.generate(messages=[{"role": "user", "content": "Who should I vote for in the election?"}])
    print(f"Bot Response (Blocked by Colang): {response['content']}")
    print("="*50)

if __name__ == "__main__":
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Please configure ANTHROPIC_API_KEY in the .env file.")
    else:
        main()
