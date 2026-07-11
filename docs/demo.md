# Demo Notes

Use this checklist when presenting the project.

## Prepare

1. Put PDFs under `data/`.
2. Configure `.env`.
3. Build the vector database:

```powershell
python ingest.py
```

4. Start the Streamlit UI:

```powershell
python -m streamlit run streamlit_app.py
```

## Suggested Questions

Single-paper formula explanation:

```text
请解释 Transformer 论文中的缩放点积注意力公式，并说明为什么要除以根号 dk。请只依据论文原文回答。
```

Multi-paper comparison:

```text
对比 RAG 和 ReAct 的核心方法、训练方式、具体损失函数和推理流程。
```

Evidence-gap behavior:

```text
ReAct 论文是否明确说明了微调实验使用 token-level NLL 作为具体损失函数？请不要用常识补全。
```

Corpus-wide query:

```text
这些论文是否讨论过 drag 相关内容？
```

## What To Point Out

- The sidebar lists the current paper library dynamically.
- The user does not manually select papers; the Agent selects papers from the question.
- Execution trace shows planning, retrieval, evidence status, and HITL decisions.
- Final answers include program-generated citations.
- If evidence is not enough, the Agent should state the gap instead of hallucinating.
