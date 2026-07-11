# Research Paper Agent

Research Paper Agent is a local multi-paper RAG agent for reading, comparing, and evaluating research papers. It ingests PDFs into a local Chroma vector database, uses a LangGraph planner to select papers and retrieval actions, checks evidence coverage before answering, supports Human-in-the-loop decisions, and includes a Streamlit demo plus an offline evaluation suite.

The project is designed as a portfolio-friendly research-agent demo: the data and vector database stay local, while the code, tests, and evaluation assets can be published safely.

## Features

- PDF ingestion with LangChain `PyPDFLoader`.
- Chunking with `RecursiveCharacterTextSplitter`.
- OpenAI-compatible embeddings with `text-embedding-3-small`.
- Local Chroma vector store persisted under `db/`.
- LangGraph agent with JSON Planner, retrieval, evidence checking, answer generation, and citation cleanup.
- Automatic intent recognition and paper selection from the user question.
- Multi-paper comparison with aspect-aware retrieval.
- Human-in-the-loop flow using LangGraph checkpoint + `interrupt()` + `Command(resume=...)`.
- Streamlit single-page demo.
- Offline evaluation suite for routing, retrieval, answer structure, HITL, and citation checks.

## Repository Layout

```text
research_paper_agent/
├── ingest.py                  # Build local Chroma database from PDFs
├── agent_tools.py             # Chroma retrieval tools used by the Agent
├── paper_agent.py             # LangGraph Agent and CLI
├── test_retrieval.py          # Manual retrieval test CLI
├── streamlit_app.py           # Streamlit demo UI
├── ui_agent_adapter.py        # UI-safe adapter around the Agent API
├── test_paper_agent.py        # Agent unit tests
├── tests/                     # UI adapter tests
├── eval/                      # Offline evaluation dataset and runners
├── docs/                      # Architecture and demo notes
├── data/                      # Local PDFs, ignored by git
├── db/                        # Local Chroma database, ignored by git
└── runtime/                   # Local checkpoint/log files, ignored by git
```

## What Is Not Committed

The following files are intentionally local only:

- `.env`
- `data/`
- `db/`
- `runtime/`
- `eval/results/`
- Python caches and local IDE files

The six PDFs used during development are public papers, but they are not committed. Put your own PDFs into `data/` and run `ingest.py` locally.

## Requirements

- Python 3.10
- An OpenAI-compatible chat and embedding endpoint
- Windows, macOS, or Linux. Paths are handled with `pathlib`; the project has been tested on Windows.

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Or with Conda:

```powershell
conda create -n rag_demo python=3.10
conda activate rag_demo
python -m pip install -r requirements.txt
```

## Configuration

Copy the example environment file:

```powershell
copy .env.example .env
```

Then edit `.env`:

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://your-openai-compatible-api/v1
CHAT_MODEL=deepseek-v4-flash
EMBEDDING_MODEL=text-embedding-3-small
AGENT_TOOL_MODE=json
MAX_AGENT_ITERATIONS=3
STREAM_FINAL_ANSWER=true
SHOW_PROGRESS_ANIMATION=true
LANGGRAPH_CHECKPOINT_PATH=runtime/langgraph_checkpoints.sqlite
MAX_AUTO_SEARCH_ROUNDS=2
MAX_LOCAL_DEEP_SEARCH_ROUNDS=1
ENABLE_HUMAN_REVIEW=true
```

`OPENAI_BASE_URL` may be left empty only when using the official OpenAI API endpoint.

## Build The Vector Database

Put PDF files under `data/`, then run:

```powershell
python ingest.py
```

The script will:

- Find PDFs in `data/`.
- Load pages with `PyPDFLoader`.
- Split text into chunks.
- Embed chunks with `text-embedding-3-small`.
- Persist the Chroma database under `db/`.

The Chroma collection name is `research_papers`. Retrieval scripts and the Agent must use the same collection name as ingestion.

## Test Retrieval

After `db/` exists:

```powershell
python test_retrieval.py
```

Type a question at the prompt. The script prints the top 5 chunks with PDF name, page, distance score, and preview. Type `q`, `quit`, or `exit` to stop.

## Run The CLI Agent

```powershell
python paper_agent.py
```

Useful commands inside the CLI:

```text
papers               List available papers
trace                Toggle trace display
session              Show current thread_id
new session          Create a new thread
resume <thread_id>   Resume a checkpointed thread
clear                Delete the current checkpoint and start fresh
q                    Quit
```

## Run The Streamlit Demo

```powershell
python -m streamlit run streamlit_app.py
```

The UI shows the current paper library in the sidebar and lets the Agent automatically choose relevant papers. It does not require users to manually select papers. When the Agent pauses for Human-in-the-loop review, the UI displays the returned decision buttons and resumes the same LangGraph thread.

## Offline Evaluation

The formal frozen evaluation set is:

```text
eval/eval_cases_v1.jsonl
```

Validate the dataset and PDF manifest locally:

```powershell
python eval/validate_eval_cases.py --check-manifest
```

Run routing evaluation without real API calls:

```powershell
python eval/evaluate_routing.py `
  --cases eval/eval_cases_v1.jsonl `
  --output eval/results/routing_baseline
```

Run retrieval evaluation after `db/` and `.env` are ready:

```powershell
python eval/evaluate_retrieval.py `
  --cases eval/eval_cases_v1.jsonl `
  --output eval/results/retrieval_oracle_k5 `
  --source-mode oracle `
  --top-k 5 `
  --run-live
```

Run targeted end-to-end smoke evaluation:

```powershell
python eval/evaluate_agent.py `
  --cases eval/eval_cases_v1.jsonl `
  --output eval/results/agent_smoke `
  --run-live
```

`eval/results/` is ignored by git because these reports are local run artifacts.

## Tests

Local verification used during development:

```powershell
python -m py_compile ingest.py agent_tools.py paper_agent.py streamlit_app.py ui_agent_adapter.py test_paper_agent.py
python -m pytest -q
python eval/validate_eval_cases.py --check-manifest
```

The full local test and manifest checks assume the project PDFs are present under `data/`. In a fresh public clone, first add PDFs and run `python ingest.py`. Live evaluation commands also require a valid `.env` and a built `db/`.

## Safety Notes

- Do not commit `.env`, API keys, PDF data, Chroma databases, SQLite checkpoints, or evaluation result folders.
- `distance_score` means lower is more similar for Chroma similarity search.
- Final citations are generated and normalized deterministically by Python after model output.
- If evidence is missing, the Agent should say so instead of filling gaps from model memory.

## GitHub Release Checklist

Before publishing:

```powershell
rg "sk-[A-Za-z0-9_-]{10,}|OPENAI_API_KEY\s*=\s*sk-" `
  -g "!data/**" -g "!db/**" -g "!runtime/**" -g "!eval/results/**" -g "!.env"
python -m pytest -q
git status --short
```

Then initialize and commit:

```powershell
git init
git add .
git status --short
git commit -m "Initial public release"
```

Create an empty GitHub repository, add its remote, then push:

```powershell
git remote add origin https://github.com/<your-name>/<repo-name>.git
git branch -M main
git push -u origin main
```

## License

MIT. See [LICENSE](LICENSE).
