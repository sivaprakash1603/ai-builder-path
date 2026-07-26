                    ┌──────────────────────┐
                    │      User Query      │
                    │ "Show students who   │
                    │  haven't paid fees"  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  Query Processin g   |
                    │ • Clean Query       │
                    │ • Generate Embeddin │
                    └──────────┬───────────┘
                               │
                               ▼
                  ┌──────────────────────────┐
                  │ Vector Database         │
                  │ (ChromaDB / FAISS)      │
                  │                         │
                  │ Stored Documents:       │
                  │ • Table Schemas         │
                  │ • Column Metadata       │
                  │ • Business Rules        │
                  │ • Sample Queries        │
                  └──────────┬───────────────┘
                             │
                    Retrieve Top-K Documents
                             │
                             ▼
                 ┌──────────────────────────┐
                 │ Context Builder          │
                 │                          │
                 │ Combine:                 │
                 │ • User Question          │
                 │ • Retrieved Schemas      │
                 │ • Business Rules         │
                 └──────────┬───────────────┘
                            │
                            ▼
               ┌────────────────────────────┐
               │ Large Language Model      │
               │ GPT-4o / Claude / Gemini  │
               │ or DeepSeek-R1 (Ollama)   │
               └──────────┬─────────────────┘
                          │
                          ▼
             ┌─────────────────────────────┐
             │ Generated Output           
             │ •SQL Query                |
             │ • Explanation              |
             │ • Summary                  │
             └──────────┬──────────────────┘
                        │
                        ▼
              ┌────────────────────────────┐
              │ PostgreSQL Database       │
              └──────────┬─────────────────┘
                         │
                         ▼
                ┌─────────────────────┐
                │ Final Result       |
                │ Table + Explanatio |
                └─────────────────────┘
