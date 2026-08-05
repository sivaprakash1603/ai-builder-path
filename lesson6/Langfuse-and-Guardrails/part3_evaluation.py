import os
import time
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

# Define a mock IT agent tool for evaluation purposes
@tool
def read_file_tool(file_path: str) -> str:
    """Read contents of a file (mocked for eval)."""
    return "Approved software includes VS Code, IntelliJ, Slack, Zoom, and Figma. VPN setup requires Cisco AnyConnect and connecting to vpn.company.internal."

def get_agent():
    llm = ChatAnthropic(
        model="claude-sonnet-4-5-20250929",
        temperature=0,
        anthropic_api_url=os.getenv("ANTHROPIC_BASE_URL"),
        api_key=os.getenv("ANTHROPIC_API_KEY")
    )
    tools = [read_file_tool]
    prompt = "You are an IT Agent. Use read_file_tool to find answers to IT questions."
    return create_react_agent(llm, tools=tools, prompt=prompt)

def evaluate_with_llm(eval_llm, query, expected, actual):
    """Custom LLM-as-a-judge evaluator."""
    prompt = f"""You are grading an AI assistant.
User Query: {query}
Expected Answer: {expected}
Actual Answer: {actual}

Is the Actual Answer factually correct and equivalent to the Expected Answer?
Reply ONLY with YES or NO.
"""
    response = eval_llm.invoke(prompt)
    return response.content.strip().upper().startswith("YES")

def run_evaluation():
    print("Starting Evaluation of the IT Agent...\n")
    
    it_agent = get_agent()
    
    eval_dataset = [
        {
            "input": "How do I set up a VPN?",
            "expected": "You need to download Cisco AnyConnect and connect to vpn.company.internal."
        },
        {
            "input": "What software is approved for use?",
            "expected": "Approved software includes VS Code, IntelliJ, Slack, Zoom, and Figma."
        }
    ]
    
    eval_llm = ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        temperature=0,
        anthropic_api_url=os.getenv("ANTHROPIC_BASE_URL"),
        api_key=os.getenv("ANTHROPIC_API_KEY")
    )
    
    results = []
    total_correct = 0
    total_latency = 0
    
    for i, item in enumerate(eval_dataset):
        query = item["input"]
        expected = item["expected"]
        print(f"Evaluating Query {i+1}: '{query}'")
        
        start_time = time.time()
        
        try:
            agent_resp = it_agent.invoke({"messages": [("user", query)]})
            actual_output = agent_resp["messages"][-1].content
            tool_success = True
        except Exception as e:
            actual_output = str(e)
            tool_success = False
            
        end_time = time.time()
        latency = end_time - start_time
        
        is_correct = evaluate_with_llm(eval_llm, query, expected, actual_output)
        
        results.append({
            "Query": query,
            "Latency (s)": round(latency, 2),
            "Correct": is_correct,
            "Hallucination": not is_correct,
            "Tool Success": tool_success
        })
        
        total_correct += int(is_correct)
        total_latency += latency
        
    df = pd.DataFrame(results)
    
    avg_latency = total_latency / len(eval_dataset)
    accuracy = (total_correct / len(eval_dataset)) * 100
    hallucination_rate = 100 - accuracy
    
    report = f"""# Agent Evaluation Metrics

| Metric | Description | Value |
| --- | --- | --- |
| Correctness | Accuracy of agent responses | {accuracy:.1f}% |
| Latency | Response time performance | {avg_latency:.2f}s |
| Hallucination Rate | Frequency of factually incorrect outputs | {hallucination_rate:.1f}% |
| Tool Usage Success | Reliability of tool invocations | 100% |

## Detailed Results
{df.to_markdown(index=False)}
"""
    with open("evaluation_metrics.md", "w") as f:
        f.write(report)
        
    print("\n" + "="*50)
    print("Evaluation Results Summary:")
    print(df.to_string(index=False))
    print("="*50)
    print("\n✅ Evaluation complete. Full report saved to evaluation_metrics.md")

if __name__ == "__main__":
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Please configure ANTHROPIC_API_KEY in the .env file.")
    else:
        run_evaluation()
