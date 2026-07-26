# Switching Endpoints in Azure OpenAI Service

## Executive Summary
When deploying enterprise-grade generative AI applications with Azure OpenAI, organizations often need to dynamically switch endpoints at runtime. This requirement arises from multi-region failover strategies, load balancing across model deployments, and environment promotions (Development -> Staging -> Production).

## Architecture & Patterns
In a traditional setup, client applications hardcode a single Azure OpenAI endpoint URL and deployment name. However, high-availability RAG systems implement an **Endpoint Switching Pattern**:

1. **Abstract Client Interface**: Wrap the underlying `AzureOpenAI` SDK client inside a domain provider class (e.g., `AzureOpenAILLMProvider`).
2. **Dynamic Credential Rotation**: Support passing new `azure_endpoint`, `api_key`, `api_version`, and `deployment_name` parameters on the fly without terminating active user chat sessions.
3. **Regional Redundancy**: If the primary East US endpoint experiences throttling (HTTP 429) or latency spikes, the RAG orchestrator automatically redirects requests to a secondary West Europe or South Central US resource.

## Key Benefits for RAG Applications
- **Zero Downtime**: Users continuing multi-turn Q&A conversations experience no interruption during model upgrades or regional failovers.
- **Model AB Testing**: Easily compare responses between `gpt-4o`, `gpt-4-turbo`, and smaller models like `gpt-35-turbo` by toggling deployment names in real time.
- **Rate Limit Mitigation**: Distribute embedding token generation (`text-embedding-ada-002` or `text-embedding-3-small`) across multiple Azure subscriptions.

## Reference Implementation
To switch endpoints in Python using the official `openai` SDK:
```python
client = AzureOpenAI(
    azure_endpoint="https://primary-resource.openai.azure.com/",
    api_key="primary-key",
    api_version="2024-08-01-preview"
)

# Switch to secondary endpoint dynamically
client.azure_endpoint = "https://secondary-resource.openai.azure.com/"
client.api_key = "secondary-key"
```
In our Antigravity RAG Studio, this is handled effortlessly via the `/api/switch-provider` REST endpoint or interactive sidebar controls!
