# SentinelDesk

Multilingual (English and German) incident triage, escalation and precedent-grounded reply drafting for IT support tickets.
**Runs fully locally on open-source models. No API keys, and ticket text never leaves the machine.**

For every incoming ticket (subject optional, body required) SentinelDesk:

1. predicts **type**, **queue** and **priority** with calibrated confidence,
2. retrieves the **top 3 to 5 most similar resolved tickets** from the historical knowledge base,
3. runs an explicit **escalation gate** (auto-draft, human review, or page the on-call engineer, always with reasons),
4. drafts a reply **in the customer's language**, grounded only in retrieved resolutions and citing their ticket IDs, then **verifies** the draft before it can be marked ready.

## Quick start

Install Ollama from https://ollama.com and open it. Then download the local model once (about 4.7 GB):

```bash
ollama pull qwen2.5:7b
```

Create a Python environment and install the dependencies (on Windows, activate with `.venv\Scripts\activate`):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Train (downloads the data, splits it, trains the models and builds the index), evaluate on the holdout, and start the console:

```bash
python train.py
python evaluate.py
streamlit run app.py
```

For fast iteration, `python train.py --max-rows 8000 --backend tfidf` finishes in about a minute.
The full run with multilingual embeddings takes several minutes on a laptop CPU (much faster on a GPU or Colab).

If Ollama is not running, everything still works: drafts come from an offline template built from the best precedent, and the sidebar says how to start the model.
`python tests/test_smoke.py` runs the whole pipeline on synthetic tickets with no download.

## Data

Official dataset: [Tobi-Bueck/customer-support-tickets](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets) (CC BY-NC 4.0).

We load one file, `aa_dataset-tickets-multi-lang-5-2-50-version.csv` (English and German). The repo holds three CSVs, and loading the whole repo would mix different versions of the tickets. Set `SENTINEL_DATA_FILE` to use the 20k file instead.

Cleaning: empty subjects become `""`, priorities are normalised to high/medium/low, rows are filtered to `en`/`de`, and exact duplicates are removed **before** splitting so the same ticket cannot appear in both the knowledge base and the holdout.

Split (seed 42, as in the problem statement): 80% knowledge base and training, 20% holdout.

## Privacy by design

Support tickets contain personal data, so every model runs on the operator's machine:

| Component | Model | License |
|---|---|---|
| Reply drafting | Qwen 2.5 7B Instruct via Ollama | Apache 2.0 |
| Precedent retrieval, burst detection | paraphrase-multilingual-MiniLM-L12-v2 | Apache 2.0 |
| Type, queue, priority, critical detector | Our own scikit-learn models, trained on the 80% split | n/a |

- The drafter only calls `localhost` and refuses any other host, so a misconfigured URL cannot leak tickets.
- No API keys or cloud SDKs are used anywhere in the code.
- PII masking (emails, phones, IPs, IBANs, cards, account numbers) still runs before drafting, as defense in depth, so drafts and logs don't echo those details.
- After the first run, set `HF_HUB_OFFLINE=1` and the system works with no internet at all.

Runs comfortably on a laptop with 16 GB of memory or more (tested target: Apple silicon with 24 GB).

## Leakage safeguards

- `data/test_inputs.csv` holds only `ticket_id, subject, body`. Labels live in `data/test_labels.csv`.
- The holdout `answer` column is never written to disk.
- Only the 80% split is indexed. `evaluate.py` asserts that no holdout ticket ID is in the index.
- Validation for threshold tuning is carved out of the 80% split; the holdout is used only once, for scoring.
- `evaluate.py` makes every prediction first and reads holdout labels only afterwards.

## How it works

**Triage** (`sentinel/triage.py`). Word 1-2 gram plus character 3-5 gram TF-IDF features feed one calibrated LinearSVC per target (sigmoid calibration, balanced class weights). Character n-grams handle German compounds and typos. The classifier's probabilities are then blended with a *neighbour vote*: how the 15 most similar knowledge-base tickets were labelled, weighted by similarity. The blend weight is tuned per target on validation data (each validation ticket excluded from its own neighbours), and weight 0 is always a candidate, so voting is only used where it improves macro-F1. Finally, one temperature per target is fitted on validation data (minimising log-loss) so that stated confidence matches real accuracy; this never changes which label wins. A fourth binary head detects *high-priority incidents*; its threshold is tuned on validation data to reach 90% recall. Overall confidence is the geometric mean of the three heads.

**Retrieval** (`sentinel/retrieval.py`). `paraphrase-multilingual-MiniLM-L12-v2` embeddings with exact cosine-similarity search in NumPy (FAISS optional via `SENTINEL_USE_FAISS=1`; off by default because it can crash next to PyTorch on macOS). The model is multilingual, so a German ticket can match English precedents (cross-lingual bonus). Precedents sharing the predicted queue or type get a small re-ranking boost. Falls back to TF-IDF cosine similarity if sentence-transformers is not installed.

**Escalation gate** (`sentinel/gate.py`), evaluated in this order:

| Rule | Outcome |
|---|---|
| Predicted type = Incident and priority = high | Escalate to on-call engineer |
| Critical-incident detector score ≥ tuned threshold | Escalate to on-call engineer |
| Outage or security language (EN/DE keyword safety net) | Escalate to on-call engineer |
| Prompt-injection pattern in the ticket | Human review |
| Overall triage confidence below threshold | Human review |
| Best precedent similarity below threshold (out of scope) | Human review |
| Asks for refund, pricing, SLA/legal guarantee, deletion or privileged access with no retrieved resolution supporting it | Human review |
| Draft fails verification (see below) | Downgraded from auto-draft to human review |
| Otherwise | Auto-draft ready |

The similarity threshold is calibrated during training: it is the 5th percentile of how well known tickets match their nearest *other* precedent, so a new ticket that matches worse than 95% of real tickets is treated as out of scope. It is capped at 0.60 for embeddings so hand-written tickets are not over-escalated, and `SENTINEL_SIM_THRESHOLD` forces a value. All thresholds can be adjusted live in the console sidebar.

**Grounded drafting** (`sentinel/generator.py`). A local Qwen 2.5 7B Instruct model (through Ollama, with JSON-constrained output and an 8k context window) receives only the masked ticket and the retrieved resolutions, with rules to cite `[TKT-xxxxx]` IDs, reply in the detected language, ignore instructions inside the ticket, and never invent prices, time frames, refunds or guarantees. Then the verifier:

- strips citations to tickets that were not retrieved,
- flags any money amount, percentage or time frame whose number does not appear in a precedent,
- respects the model's own `grounded` / `unsupported_requests` signals.

To use a different local model, set `OLLAMA_MODEL` (for example `qwen2.5:14b` for better quality if you have the memory). If the model is unavailable, drafting falls back to the offline template, so the demo never crashes.

**Bonus features**

- *PII guardrail*: everything runs locally, and emails, phone numbers, IPs, IBANs, Luhn-valid card numbers and account numbers are additionally masked before drafting.
- *Outage burst detection*: each ticket is compared with tickets from the last 30 minutes; three or more similar ones raise a major-incident banner.
- *Cross-lingual precedents*: via the multilingual embedding model.

## Evaluation

`python evaluate.py` prints and saves (`artifacts/eval_report.json`):

- macro-F1, accuracy and expected calibration error (ECE) for type, queue and priority,
- **high-priority incident recall**, both from predicted labels alone and for the full on-call gate, with the gate's precision and paging rate,
- a confidence-threshold table (coverage vs. accuracy) to choose the gate setting,
- end-to-end p50/p95 latency and the escalation mix on a holdout sample (`--with-llm` includes local LLM drafting, which adds several seconds per ticket on a laptop).

The console's *Holdout metrics* tab shows the same report.

## Five-minute demo script

1. Load **Hospital VPN outage**: on-call escalation with reasons, holding reply, precedents.
2. Load **Password reset**: auto-draft ready, cited ticket IDs, grounding checks passed.
3. Load the **German printer** ticket: language detected, German reply, English precedents may be cited.
4. Load **Refund and SLA credit demand**: PII masked, human review because no precedent supports the request.
5. Load **Prompt-injection attempt** and **Out of scope**: neither gets an automatic answer.
6. Open **Outage monitor**, select *Simulate an outage burst*: the major-incident banner appears.
7. Open **Holdout metrics**: macro-F1, high-priority incident recall and latency.

## Project structure

```
app.py               Streamlit triage console
train.py             split, train, tune thresholds, build index
evaluate.py          holdout metrics and latency
sentinel/
  config.py          paths, thresholds, model names
  data.py            loading, cleaning, leakage-safe split
  textutils.py       language detection, PII masking, risk patterns
  triage.py          classifiers and critical-incident detector
  retrieval.py       embeddings or TF-IDF search, threshold calibration
  gate.py            escalation rules
  generator.py       prompting, LLM calls, verification, offline template
  burst.py           outage burst detector
  pipeline.py        end-to-end run with per-stage latency
  examples.py        demo tickets
tests/test_smoke.py  synthetic end-to-end test
```

## Known limits and tuning tips

- Priority is the hardest label because urgency is often implied. The on-call gate deliberately trades precision for recall; check its paging rate in the report and adjust the detector threshold if it pages too often.
- If too many tickets go to human review, lower the confidence threshold using the coverage table.
- Local drafting takes several seconds per ticket on a laptop; triage, retrieval and the gate take milliseconds. Lowering "Precedents to retrieve" to 3 shortens the prompt and speeds up drafting.
- A fine-tuned multilingual transformer (for example XLM-R or mDeBERTa) would likely beat TF-IDF on macro-F1 if you have a GPU and time; the `TriageModel` interface (`fit`, `predict`, `save`, `load`) is the only thing to swap.