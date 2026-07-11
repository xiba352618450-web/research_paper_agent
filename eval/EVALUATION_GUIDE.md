# Offline Evaluation Guide

This evaluation system is layered so failures can be attributed before changing the Agent.

## 1. Check Frozen Data

```powershell
python eval/validate_eval_cases.py --check-manifest
```

Stop if the manifest check fails. The frozen v1 JSONL, manifest, and PDFs must remain unchanged.

## 2. Routing

Routing is fully offline. It does not call the chat model, embeddings, or Chroma.

```powershell
python eval/evaluate_routing.py `
  --cases eval/eval_cases_v1.jsonl `
  --output eval/results/routing_baseline
```

## 3. Oracle-Source Retrieval

This isolates vector retrieval by using `expected_papers` as the target source list.

```powershell
python eval/evaluate_retrieval.py `
  --cases eval/eval_cases_v1.jsonl `
  --output eval/results/retrieval_oracle_k5 `
  --source-mode oracle `
  --top-k 5 `
  --run-live
```

Retrieval calls the embedding service. Without `--run-live`, it only writes a dry-run plan.

## 4. Predicted-Source Retrieval

This combines deterministic routing with retrieval, and should be reported separately from oracle-source retrieval.

```powershell
python eval/evaluate_retrieval.py `
  --cases eval/eval_cases_v1.jsonl `
  --output eval/results/retrieval_predicted_k5 `
  --source-mode predicted `
  --top-k 5 `
  --run-live
```

## 5. E2E Smoke

By default, E2E evaluates only:

- `eval-001`
- `eval-005`
- `eval-013`
- `eval-018`
- `eval-020`

```powershell
python eval/evaluate_agent.py `
  --cases eval/eval_cases_v1.jsonl `
  --output eval/results/agent_smoke `
  --run-live
```

E2E calls the chat model and embeddings. It temporarily sets:

- `STREAM_FINAL_ANSWER=false`
- `SHOW_PROGRESS_ANIMATION=false`

It does not edit `.env`.

## 6. Full E2E

Run all 20 cases only after reviewing the smoke report.

```powershell
python eval/evaluate_agent.py `
  --cases eval/eval_cases_v1.jsonl `
  --output eval/results/agent_full `
  --all `
  --run-live
```

Do not run multiple full E2E batches in parallel. That makes rate limits harder to diagnose.

## Resume

All runners support:

```powershell
--resume
```

Completed non-error cases are skipped. Error cases are retried.

## HITL

HITL is driven through the public `agent.resume({"action": ...})` interface. No CLI `input()` is used. If a programmatic HITL interface is unavailable, the case is marked `unsupported` rather than faked as passing.

## Scoring Notes

- Broad questions use group recall.
- Focused questions use page recall diagnostics.
- Citation semantic support is marked for manual review; the first version does not use an LLM judge.
- Exact facts are checked with deterministic rules where possible, but negative claims such as “not explicitly stated” still require semantic review.
