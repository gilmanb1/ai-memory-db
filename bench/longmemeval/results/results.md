# LongMemEval Benchmark Results — ai-memory-db

**Overall Accuracy**: 74.1% (20/27)

## Results by Question Type

| Question Type | Total | Valid | Correct | Invalid | Accuracy |
|---|---|---|---|---|---|
| single-session-user | 5 | 5 | 4 | 0 | 80.0% |
| single-session-assistant | 5 | 4 | 2 | 1 | 50.0% |
| multi-session | 5 | 5 | 3 | 0 | 60.0% |
| temporal-reasoning | 5 | 4 | 4 | 1 | 100.0% |
| knowledge-update | 5 | 4 | 3 | 1 | 75.0% |
| single-session-preference | 5 | 5 | 4 | 0 | 80.0% |
| **OVERALL** | **30** | **27** | **20** | **3** | **74.1%** |

## Comparison with Published Baselines

| System | Accuracy |
|---|---|
| Hindsight | 91.4% |
| Supermemory | 85.2% |
| Zep | 71.2% |
| GPT-4o | 60.2% |
| **ai-memory-db** | **74.1%** |

## Model Configuration
- **Extraction**: see detailed_results
- **Answer Generation**: see detailed_results
- **Judge**: see detailed_results
- **Embeddings**: nomic-embed-text