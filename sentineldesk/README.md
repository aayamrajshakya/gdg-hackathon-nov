# SentinelDesk

Multilingual (English and German) incident triage, escalation and precedent-grounded reply drafting for IT support tickets.

For every incoming ticket (subject optional, body required) SentinelDesk:

1. predicts **type**, **queue** and **priority** with calibrated confidence,
2. retrieves the **top 3 to 5 most similar resolved tickets** from the historical knowledge base,
3. runs an explicit **escalation gate** (auto-draft, human review, or page the on-call engineer, always with reasons),
4. drafts a reply **in the customer's language**, grounded only in retrieved resolutions and citing their ticket IDs, then **verifies** the draft before it can be marked ready.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                    # optional: add an LLM API key
python train.py                                         # download data, split, train, build index
python evaluate.py                                      # holdout metrics -> artifacts/eval_report.json
streamlit run app.py                                    # live triage console
```

For fast iteration, `python train.py --max-rows 8000 --backend tfidf` finishes in about a minute.
The full run with multilingual embeddings takes several minutes on a laptop CPU (much faster on a GPU or Colab).

No API key? Everything still works: drafts come from an offline template built from the best precedent.
`python tests/test_smoke.py` runs the whole pipeline on synthetic tickets with no download and no key.

## Data

Official dataset: [Tobi-Bueck/customer-support-tickets](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets) (CC BY-NC 4.0).

We load one file, `aa_dataset-tickets-multi-lang-5-2-50-version.csv` (English and German). The repo holds three CSVs, and loading the whole repo would mix different versions of the tickets. Set `SENTINEL_DATA_FILE` to use the 20k file instead.

Cleaning: empty subjects become `""`, priorities are normalised to high/medium/low, rows are filtered to `en`/`de`, and exact duplicates are removed **before** splitting so the same ticket cannot appear in both the knowledge base and the holdout.

Split (seed 42, as in the problem statement): 80% knowledge base and training, 20% holdout.

## Leakage safeguards

- `data/test_inputs.csv` holds only `ticket_id, subject, body`. Labels live in `data/test_labels.csv`.
- The holdout `answer` column is never written to disk.
- Only the 80% split is indexed. `evaluate.py` asserts that no holdout ticket ID is in the index.
- Validation for threshold tuning is carved out of the 80% split; the holdout is used only once, for scoring.
- `evaluate.py` makes every prediction first and reads holdout labels only afterwards.

## How it works

**Triage** (`sentinel/triage.py`). Word 1-2 gram plus character 3-5 gram TF-IDF features feed one calibrated LinearSVC per target (sigmoid calibration, balanced class weights). Character n-grams handle German compounds and typos. A fourth binary head detects *high-priority incidents*; its threshold is tuned on validation data to reach 90% recall. Overall confidence is the geometric mean of the three heads.

**Retrieval** (`sentinel/retrieval.py`). `paraphrase-multilingual-MiniLM-L12-v2` embeddings in a FAISS inner-product index. The model is multilingual, so a German ticket can match English precedents (cross-lingual bonus). Precedents sharing the predicted queue or type get a small re-ranking boost. Falls back to TF-IDF cosine similarity if sentence-transformers is not installed.

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

The similarity threshold is calibrated during training: it is the 5th percentile of how well known tickets match their nearest *other* precedent, so a new ticket that matches worse than 95% of real tickets is treated as out of scope. All thresholds can be adjusted live in the console sidebar.

**Grounded drafting** (`sentinel/generator.py`). The prompt contains only the masked ticket and the retrieved resolutions, with rules to cite `[TKT-xxxxx]` IDs, reply in the detected language, ignore instructions inside the ticket, and never invent prices, time frames, refunds or guarantees. The model returns JSON. Then the verifier:

- strips citations to tickets that were not retrieved,
- flags any money amount, percentage or time frame whose number does not appear in a precedent,
- respects the model's own `grounded` / `unsupported_requests` signals.

Supported LLMs: Anthropic (`ANTHROPIC_API_KEY`) or any OpenAI-compatible endpoint (`OPENAI_API_KEY`, optional `OPENAI_BASE_URL` for Groq, Together, Ollama and similar). Any API error falls back to the offline template, so the demo never crashes.

**Bonus features**

- *PII guardrail*: emails, phone numbers, IPs, IBANs, Luhn-valid card numbers and account numbers are masked before anything is sent to an LLM.
- *Outage burst detection*: each ticket is compared with tickets from the last 30 minutes; three or more similar ones raise a major-incident banner.
- *Cross-lingual precedents*: via the multilingual embedding model.

## Evaluation

`python evaluate.py` prints and saves (`artifacts/eval_report.json`):

- macro-F1, accuracy and expected calibration error (ECE) for type, queue and priority,
- **high-priority incident recall**, both from predicted labels alone and for the full on-call gate, with the gate's precision and paging rate,
- a confidence-threshold table (coverage vs. accuracy) to choose the gate setting,
- end-to-end p50/p95 latency and the escalation mix on a holdout sample (`--with-llm` includes LLM drafting).

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
  retrieval.py       embeddings/FAISS or TF-IDF, threshold calibration
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
- A fine-tuned multilingual transformer (for example XLM-R or mDeBERTa) would likely beat TF-IDF on macro-F1 if you have a GPU and time; the `TriageModel` interface (`fit`, `predict`, `save`, `load`) is the only thing to swap.
