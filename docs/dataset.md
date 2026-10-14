# Dataset reference

Canonical data: [epiq-ai-labs/corporatebench](https://huggingface.co/datasets/epiq-ai-labs/corporatebench).

## What is this?

LLMs are increasingly able to answer complex questions about enterprise-scale document collections.
But evaluation is hard: companies don't want to share internal communications, and synthetic datasets have been overly simple.
We present CorporateBench (CB), a human-validated multi-task Q&A benchmark whose scale approaches the conditions LLMs encounter in corporate communication networks, with evaluation corpora surpassing 230,000 documents.
CB evaluates LLMs across two dimensions (information extraction and knowledge base querying) through four synthetically generated firms ranging from 12 to 10,000 employees.
Each corpus is sampled from a temporally evolving knowledge base describing a consistent world, guaranteeing cross-document logical consistency even across hundreds of thousands of documents.
CB provides LLM developers a metric for corporate communication reasoning, filling a crucial gap in the benchmarking ecosystem.

## Dataset structure

```text
data/
├── kb/                 # One .kb archive per company
├── kb_qa/              # Knowledge-base QA questions
├── topic_qa/           # Document-topic QA questions
├── integrated_qa/      # Questions combining graph and topic evidence
└── additional_qa/      # Supplemental topic-hierarchy configurations
```

Each `.kb` file is a ZIP-compatible archive containing:

| Path | Contents |
| --- | --- |
| `graph.nq` | Ground-truth entities, properties, and temporal relationships in N-Quads |
| `ontologies/*.ttl` | The RDF ontologies used to validate the graph |
| `documents/*.txt` | Synthetic email, agenda, meeting-minute, and calendar documents |
| `manifest.json` | Archive format, version, component, and document-count metadata |

## Company configurations

| Configuration | Scale | Industry | Employees | Documents in `.kb` |
| --- | --- | --- | ---: | ---: |
| `zenith` | Small | Technology | 12 | 353 |
| `streamvibe` | Medium | Media | 122 | 3,925 |
| `biocure` | Large | Pharmacare | 1,110 | 26,492 |
| `pound` | Extra large | Finance | 10,210 | 232,692 |

## Tasks and evaluation

| Task | Configurations | Released protocol | Metric |
| --- | --- | --- | --- |
| KB Evaluation | All four companies | Extract entities and relationships from every document, then map them to ground truth | Per-type and macro entity F1; structural and temporal relationship F1 |
| Topic Classification | `biocure`, `pound` | 31 and 17 labels respectively; 0, 10, or 50 shots; 1,000 test documents; five repetitions | Macro F1 across topics |
| KB QA | All four companies | 250 questions per company about entities, relationships, and time | Exact match for scalar answers; set F1 for lists |
| Topic QA | All four companies | 250 questions per company about document topics | Exact match for scalar answers; set F1 for lists |
| Integrated QA | All four companies | 250 questions per company combining graph and document evidence | Exact match for scalar answers; set F1 for lists |

The three QA tasks contain 3,000 core questions. The paper evaluates them with either document retrieval (RAG, `k=10`) or SQL access to the ground-truth KB, with at most five tool calls per question. Scalar comparison is case-insensitive; list answers receive partial credit through set F1.

## QA schema

Every core QA file is a JSON object with `metadata` and `questions` fields.

| Field | Type | Description |
| --- | --- | --- |
| `metadata.dataset_type` | string | `kb_qa`, `topic_qa`, or `integrated_qa` |
| `metadata.kb` | string | Associated `.kb` filename |
| `metadata.topic_level` | integer, optional | Topic-hierarchy level used to generate topic-aware questions |
| `questions[].id` | integer | Question identifier within the file |
| `questions[].question` | string | Natural-language evaluation question |
| `questions[].answer` | boolean, integer, string, or list of strings | Deterministic ground-truth answer |
| `questions[].answer_type` | string | `bool`, `int`, `date`, `str`, or `List[str]` |
| `questions[].template` | string | Instantiated question template |
| `questions[].variables` | object | Template values used to construct the question |
| `questions[].constraints` | integer, optional | Number of constraints required to answer the question |

The supplemental files under `additional_qa/` use the same schema and provide alternate topic-hierarchy levels. Topic-classification examples and labels are read from the documents and topic graph inside the `biocure` and `pound` KB archives.

## Construction and validation

1. Define an RDF ontology for companies, departments, teams, employees, projects, tasks, meetings, and temporal relationships.
2. Generate a valid organizational hierarchy, then simulate tasks, meetings, and hierarchy changes over a 90-day quarter.
3. Annotate entities and serialize the world as an ontology-validated N-Quads graph.
4. Generate corporate documents around graph-derived evidence strings. Email structure is modeled on the Enron corpus; topics, employee personalities, and formality levels provide variation.
5. Derive QA answers through manually verified SPARQL queries rather than model generation.

## Responsible use

CorporateBench is intended for research and evaluation of retrieval, extraction, and reasoning systems. It contains synthetic organizations and communications, not real corporate records. Generated names may nevertheless coincide with real people by chance.

## Cite us

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
