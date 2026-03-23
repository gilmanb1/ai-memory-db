# LongMemEval Benchmark Results — ai-memory-db

**Overall Accuracy**: 70.0% (21/30)

## Results by Question Type

| Question Type | Total | Valid | Correct | Invalid | Accuracy |
|---|---|---|---|---|---|
| single-session-user | 5 | 5 | 5 | 0 | 100.0% |
| single-session-assistant | 5 | 5 | 4 | 0 | 80.0% |
| multi-session | 5 | 5 | 2 | 0 | 40.0% |
| temporal-reasoning | 5 | 5 | 4 | 0 | 80.0% |
| knowledge-update | 5 | 5 | 4 | 0 | 80.0% |
| single-session-preference | 5 | 5 | 2 | 0 | 40.0% |
| **OVERALL** | **30** | **30** | **21** | **0** | **70.0%** |

## Comparison with Published Baselines

| System | Accuracy |
|---|---|
| Hindsight | 91.4% |
| Supermemory | 85.2% |
| Zep | 71.2% |
| GPT-4o | 60.2% |
| **ai-memory-db** | **70.0%** |

## Model Configuration
- **Extraction**: see detailed_results
- **Answer Generation**: see detailed_results
- **Judge**: see detailed_results
- **Embeddings**: nomic-embed-text