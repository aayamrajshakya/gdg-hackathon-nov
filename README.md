# SentinelDesk — Autonomous Incident Triage & Grounded Resolution Copilot

SentinelDesk is a multilingual IT support ticket triage prototype built for the GDG November hackathon. It predicts ticket **type**, **routing queue**, and **priority**; retrieves relevant historical resolutions; applies an escalation/safety gate; and uses **Gemma via Ollama** to draft evidence-backed replies in English or German.

> **Status:** Prototype. Model quality and escalation thresholds should be validated before use in a real support workflow. Draft responses are for human review and are **not automatically sent**.

## Features

- **Multi-task triage:** Three calibrated TF-IDF + logistic-regression classifiers predict type, queue, and priority, with confidence estimates.
- **Historical precedent retrieval:** The `intfloat/multilingual-e5-small` sentence-transformer model embeds training tickets; cosine similarity finds the top relevant cases.
- **Human escalation gate:** Flags high-priority incidents, uncertain classification, weak precedent similarity, possible prompt injection, and sensitive requests unsupported by retrieved resolutions.
- **Grounded multilingual drafts:** Gemma produces English/German replies using retrieved historical answers and ticket-ID citations. Simple post-generation checks flag invalid citations and unsupported numeric claims.
- **Privacy guardrail:** Masks selected emails, phone numbers, IP addresses, account numbers, payment cards, and IBAN-like values before retrieval and generation.
- **Streamlit dashboard:** Displays routing predictions, confidences, latency, escalation reasons, precedent evidence, draft replies, and session ticket history. Includes basic repeated-incident/burst detection.

## Project structure

```text
.
├── app.py                 # Streamlit live demo interface
├── pipeline.py            # Classification, retrieval, escalation, Gemma drafting, checks
├── retrieval.py           # Embedding-based precedent search
├── prepare_data.py        # Clean data, generate 80/20 split
├── train_classifier.py    # Train classifiers + produce holdout metrics
├── build_index.py         # Build embeddings for historical (training) tickets only
├── common.py              # Shared text and label helpers
├── requirements.txt
├── data/                   # Local dataset and split files (generated; not committed)
├── models/                 # Trained classifiers and metrics (generated)
└── index/                  # Historical embeddings and metadata (generated)
```

## Prerequisites

- Python 3.10–3.12 recommended (a virtual environment is strongly recommended).
- [Ollama](https://ollama.com/) installed and running.
- An Ollama-compatible Gemma model downloaded locally; the exact model name must match `ollama list`.
- Enough disk space and RAM for the embedding model, classifiers, and selected Gemma model.

## Local installation

From the repository directory:

```bash
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

On Windows, activate the venv using `.venv\Scripts\activate` instead.

Check Ollama and download a model if needed:

```bash
ollama list
# Example only; choose a model your machine can support:
ollama pull gemma3:4b
export OLLAMA_MODEL=gemma3:4b
```

The code defaults to `gemma4:12b` if `OLLAMA_MODEL` is unset; use `OLLAMA_MODEL` to select **your installed model**. Start the Ollama app/service before running the pipeline.

## Dataset

Download the open [Customer Support Tickets dataset](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets) by `Tobi-Bueck` (CC BY-NC 4.0) and put a CSV into `data/`.

For faster prototyping, use `dataset-tickets-multi-lang-4-20k.csv` if available. The preparation script selects a CSV with `20k` in its filename first. **The script does not download data automatically.**

Expected input fields include `subject`, `body`, `type`, `queue`, `priority`, `language`, and `answer`.

```bash
mkdir -p data
python3 prepare_data.py
```

This creates:

- `data/kb_train.parquet`: 80% training/historical knowledge base (includes resolutions).
- `data/test_inputs.parquet`: 20% holdout inputs without answers or target labels.
- `data/test_labels.parquet`: holdout labels used **only** to calculate evaluation metrics.

**Data leakage rule:** Never train on or index holdout answers, labels, or resolutions. Index only `kb_train.parquet`.

## Train and build the retrieval index

Run from the repository root **in this order**:

```bash
python3 prepare_data.py
python3 train_classifier.py
python3 build_index.py
```

Generated files:

- `models/triage.joblib` — calibrated classifiers and vectorizer.
- `models/metrics.json` — holdout evaluation metrics.
- `index/kb_embeddings.npy` — normalized historical-ticket embeddings.
- `index/kb_meta.parquet` — historical tickets and resolutions.

To test retrieval and the end-to-end pipeline:

```bash
python3 retrieval.py
python3 pipeline.py
```

## Run the web dashboard

```bash
python3 -m streamlit run app.py
```

Open **http://localhost:8501** and paste a ticket subject/body or choose a built-in example. The app shows predictions, confidence, routing, escalation reasons, five retrieved tickets, and—when the gate permits it—a Gemma-generated draft with citations.

If no historical precedent is sufficiently similar, a possible prompt injection is detected, or the LLM fails, the application can suppress the draft and/or flag the request for human review. Even when a draft is generated, escalated items are for review only.

## Evaluation

`train_classifier.py` performs holdout evaluation and records results in `models/metrics.json`, including:

- Macro-F1 and accuracy for `type`, `queue`, and `priority`.
- Recall on truly high-priority incidents predicted as both high priority and Incident.
- Overall high-priority recall.

No metrics are hard-coded here; record the measured values after running the training script. Note that recall of the full **escalation gate** is not identical to the two-classifier incident recall metric.

## Suggested live demo

1. **High-risk:** Hospital VPN outage → check incident detection, priority and escalation.
2. **German:** Password reset → inspect language detection and German draft.
3. **Unsupported request:** Refund/guarantee demand → check safety escalation.
4. **Prompt injection:** Ticket instructing the LLM to ignore instructions → demonstrate blocking.
5. **Burst detection:** Submit three similar VPN outage reports in the same Streamlit session → inspect the outage banner (thresholds depend on similarity).

## Security and limitations

- Do not commit secrets, private ticket data, or credentials.
- PII masking uses regular expressions and may miss some sensitive information; it is not a complete privacy solution.
- Similarity and confidence thresholds are heuristic defaults and need validation/tuning.
- The current prototype stores the burst-detection history in the Streamlit session, not in a shared production database.
- A generated draft should not be sent to customers without appropriate approval.
- Use the dataset in accordance with its CC BY-NC 4.0 license.

## Tech stack

Python · Streamlit · scikit-learn · Sentence Transformers · PyTorch · pandas · NumPy · Ollama · Gemma

## Repository workflow

Work on a feature branch, push it to the shared repository, and create a pull request into `main` for team review.
