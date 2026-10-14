# CorporateBench

Evaluation code for **CorporateBench: Large-Scale Q&A Benchmarking with Temporal Knowledge Bases** (Findings of EMNLP 2026).

[Paper (arXiv)](https://arxiv.org/abs/2608.27391) · [Dataset (Hugging Face)](https://huggingface.co/datasets/epiq-ai-labs/corporatebench) · [Website/Dataset Viewer](https://corporatebench.epiqai.com/)

CorporateBench evaluates information extraction, retrieval, and question answering. It uses document collections from four synthetic companies. These collections contain between 353 and 232,692 documents.

This repository contains the evaluation code. The datasets are available on Hugging Face.

## Install

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) to manage the project and its dependencies.

Clone the repository and install the dependencies:

```bash
git clone https://github.com/Epiq-AI-Labs/CorporateBench.git
cd CorporateBench
uv sync --locked --extra openai --extra anthropic
```

You can install the dependencies for one model provider. Use `--extra openai` for OpenAI or `--extra anthropic` for Anthropic.

Set the API key in your shell:

- For OpenAI, set `OPENAI_API_KEY`.
- For Anthropic, set `ANTHROPIC_API_KEY`.

## Quickstart

Run the QA example with Zenith, the smallest company. The example runs 10 questions from each of the three tasks, for a total of 30 questions.

```bash
uv run examples/qa.py
```

The example uses `OpenAIBackend("gpt-5-nano")` with retrieval-augmented generation (RAG) and local embeddings. On the first RAG run, the runner downloads the embedding model. It then builds and saves an index for later use. The model provider charges for API calls.

The following example runs 10 questions from the KB QA task only:

```python
import asyncio

from corporatebench import CorporateBench, OpenAIBackend


async def main():
    async with CorporateBench(company="zenith") as bench:
        result = await bench.run(
            OpenAIBackend("gpt-5-nano"),
            method="rag",
            tasks=("kb_qa",),
            limit=10,
        )
        print(result.score)
        print(result.scores_by_task)


if __name__ == "__main__":
    asyncio.run(main())
```

Save this example as a Python file. Run the file with `uv run`.

If you use your own `data_dir` with an API backend, document content is sent to the configured model provider.

Use `method="kb"` to query the ground-truth knowledge base with SQL.

Use `result.to_dict()` to convert the results to a dictionary. The dictionary includes the score and any error for each question. A run can finish even if some questions fail. Check these errors when you review the overall score.

## Tasks and examples

| Task | Run | Scope |
| --- | --- | --- |
| KB QA, Topic QA, Integrated QA | `uv run examples/qa.py` | 10 questions per task, Zenith, RAG |
| Knowledge-base extraction | `uv run examples/kb_evaluation.py` | All eligible Zenith documents. Requires a request budget. |
| Topic classification | `uv run examples/topic_classification.py` | Biocure, zero-shot, one repetition, 31 requests |

The dataset contains four companies: `zenith`, `streamvibe`, `biocure`, and `pound`. Topic classification supports only Biocure and Pound.

See the [dataset reference](docs/dataset.md) for data formats, company details, task descriptions, and scoring methods.

## Paper settings and reproducibility

To run the full QA evaluation, remove the `limit` argument. Each company has 250 questions for each of the three QA tasks.

By default, retrieval returns 10 documents (`top_k=10`). The runner permits a maximum of five tool calls per question.

By default, RAG uses `sentence-transformers/all-MiniLM-L6-v2` for embeddings. The paper uses OpenAI `text-embedding-3-small` with 1,536 dimensions.

To use the paper's embedding model:

1. Install the `openai` extra.
2. Set `OPENAI_API_KEY` in your shell.
3. Create the benchmark with `CorporateBench(company="zenith", paper_embeddings=True)`.

OpenAI charges for these embedding API calls.

Use these settings for the full topic classification evaluation:

- `shots=(0, 10, 50)`
- `repetitions=5`
- `test_size=1000`
- `max_requests=15000`

For KB extraction, set `max_requests=bench.kb_evaluation_requests`.

## Running Other models

CorporateBench supports Anthropic and OpenAI endpoints out of the box. 

To connect other models or embedding methods, see the `ModelBackend` and `EmbeddingBackend` protocols. Use `CallableBackend` to connect a synchronous or asynchronous function. The function must return a string or a `ModelRun` object. For QA, it receives `system`, `prompt`, `tools`, and `max_tool_calls`. The function must manage its own tool calls.

## Additional notes

### Hugging Face offline mode and telemetry

To disable Hugging Face Hub requests and telemetry, set these variables before you start Python:

```bash
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
```

Download the data and embedding model first. Model provider API calls are not affected.

### Security

See [SECURITY.md](SECURITY.md) to report a security issue.

## Citation

```bibtex
@inproceedings{hamilton2026corporatebench,
  title = {{CorporateBench}: Large-Scale {Q\&A} Benchmarking with Temporal Knowledge Bases},
  author = {Hamilton, Sil and Sun, Albert Yu and Romero, Oscar J. and Henneking, Carl-Leander and Mimno, David and Yang, Bishan and Labutov, Igor},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2026},
  publisher = {Association for Computational Linguistics},
  address = {Budapest, Hungary},
  month = {October},
  year = {2026}
}
```
