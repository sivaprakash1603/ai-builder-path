# Agent Evaluation Metrics

| Metric | Description | Value |
| --- | --- | --- |
| Correctness | Accuracy of agent responses | 50.0% |
| Latency | Response time performance | 4.16s |
| Hallucination Rate | Frequency of factually incorrect outputs | 50.0% |
| Tool Usage Success | Reliability of tool invocations | 100% |

## Detailed Results
| Query                              |   Latency (s) | Correct   | Hallucination   | Tool Success   |
|:-----------------------------------|--------------:|:----------|:----------------|:---------------|
| How do I set up a VPN?             |          5.12 | True      | False           | True           |
| What software is approved for use? |          3.2  | False     | True            | True           |
