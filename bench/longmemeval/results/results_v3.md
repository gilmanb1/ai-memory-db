# LongMemEval Benchmark Results — ai-memory-db

**Overall Accuracy**: 75.0% (21/28)

## Results by Question Type

| Question Type | Total | Valid | Correct | Invalid | Accuracy |
|---|---|---|---|---|---|
| single-session-user | 5 | 5 | 5 | 0 | 100.0% |
| single-session-assistant | 5 | 5 | 3 | 0 | 60.0% |
| multi-session | 5 | 3 | 2 | 2 | 66.7% |
| temporal-reasoning | 5 | 5 | 5 | 0 | 100.0% |
| knowledge-update | 5 | 5 | 3 | 0 | 60.0% |
| single-session-preference | 5 | 5 | 3 | 0 | 60.0% |
| **OVERALL** | **30** | **28** | **21** | **2** | **75.0%** |

## Comparison with Published Baselines

| System | Accuracy |
|---|---|
| Hindsight | 91.4% |
| Supermemory | 85.2% |
| Zep | 71.2% |
| GPT-4o | 60.2% |
| **ai-memory-db** | **75.0%** |

## Model Configuration
- **Extraction**: see detailed_results
- **Answer Generation**: see detailed_results
- **Judge**: see detailed_results
- **Embeddings**: nomic-embed-text