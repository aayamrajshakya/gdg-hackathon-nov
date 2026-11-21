# 🛡️ SentinelDesk

Autonomous incident triage & grounded resolution copilot, built with the Gemma family: **EmbeddingGemma** for multilingual retrieval and **Gemma 3 4B running locally through Ollama** for drafting replies that cite their sources. Every model runs on the laptop, so ticket text never leaves the machine.

A raw ticket (subject optional + body, in English or German) goes through four steps:

1. **Triage**: predicts type, queue and priority with calibrated confidence, plus a dedicated critical-incident risk.
2. **Escalation gate**: deterministic, explainable rules that decide between `AUTO_DRAFT`, `HUMAN_REVIEW` and `ESCALATE_ONCALL`.
3. **Precedent retrieval**: the top 5 most similar resolved tickets from the historical KB, across languages.
4. **Grounded draft**: a local Gemma reply in the requester's language that cites ticket IDs. It is then checked for hallucinated citations, invented numbers and the wrong language.

## Results

20 % holdout, n = 5,718, never used for training or tuning. The triage numbers do not depend on the LLM. Your own run writes them to `reports/metrics.json`.

| Head | Macro-F1 | Accuracy | ECE | TF-IDF only | EmbeddingGemma only | kNN only |
|---|---|---|---|---|---|---|
| type | 0.941 | 0.939 | 0.005 | 0.876 | 0.829 | 0.940 |
| queue | 0.787 | 0.801 | 0.070 | 0.590 | 0.344 | 0.785 |
| priority | 0.828 | 0.835 | 0.024 | 0.633 | 0.463 | 0.832 |
| critical | 0.909 | 0.948 | 0.013 | 0.808 | 0.702 | 0.899 |

High-priority Incident recall (true priority=high ∧ type=Incident, n = 995):

| | Recall | Precision |
|---|---|---|
| Predicted labels are exactly (high, Incident) | 90.2% | 78.8% |
| Escalation gate pages on-call (argmax rule ∨ calibrated risk ≥ 0.38) | 92.3% | 73.7% |

Gate outcomes on the holdout (pre-generation): 61% auto-draft · 14% human review · 25% on-call. Only 3.2% of true critical incidents end up in auto-draft. Auto-drafted tickets are routed correctly (type and queue both right) 85.7% of the time, against 76.9% overall.

**Grounded generation audit** (20 random holdout tickets, `ollama:gemma3:4b`): valid citation in TODO% of drafts, hallucinated ticket IDs TODO%, unsupported money/percent/time claims TODO%, reply in the requester's language TODO%. *(Fill in from `uv run python scripts/evaluate.py --gen 20`.)*

**Latency**: triage (language, embedding, retrieval, classification, gate) takes p50 80 ms and p95 133 ms on an 8 GB M1 Mac. The draft streams after that. Gemma 3 4B on Ollama takes about 7 s per draft on a 24 GB M5 MacBook Air.

**Retrieval**: queue precision@5 is 0.52 and type precision@5 is 0.83. 25% of retrieved precedents are in the other language, so cross-lingual matching is being used.

### Note on near-duplicates

The dataset contains many paraphrased or translated near-duplicate tickets. 41% of holdout tickets have a KB neighbour with cosine ≥ 0.95, and the kNN-over-precedents component exploits that, legitimately, since it only searches the KB. To avoid inflated headline numbers we also score the 35% of holdout tickets with no KB neighbour above 0.90:

| Head | Blended model on novel tickets | TF-IDF baseline on novel tickets |
|---|---|---|
| type | 0.878 | 0.806 |
| queue | 0.556 | 0.348 |
| priority | 0.646 | 0.449 |

Scores drop on novel tickets, as you'd expect, but the blend still beats the classic baseline by a wide margin. Reproduce everything with `uv run python scripts/evaluate.py --gen 20`. The full output is in `reports/metrics.json`.

## Architecture

```
            ┌──────────────────── everything runs locally, no data leaves the machine ────────────────────┐
 subject + body ──► clean ──► language detect (lingua) ──► EmbeddingGemma-300m ──► exact cosine index (22,869 KB tickets)
                                                               │                         │ top-20
                                                               ▼                         ▼
                                     TF-IDF LR  +  EmbeddingGemma LR  +  kNN precedent vote   (per head)
                                                 └──── blend weights + temperature tuned on KB-val ───┘
                                                               │  type · queue · priority · P(critical)
                                                               ▼
              burst detector ──►  ESCALATION GATE  ◄── safety lexicon · sensitive-action check · injection check
                                                               │
                                     PII mask ──► Gemma 3 4B via Ollama (local) + top-5 precedents
                                                               │
                                     post-checks: citations ∈ retrieved? numbers ∈ evidence? language ok?
                                                               ▼
                                              Streamlit Live Triage Console
```

| Component | Choice | Why |
|---|---|---|
| Embeddings | `google/embeddinggemma-300m` (768-d, normalised) | Multilingual with one shared space, so a German ticket can match an English precedent. Small enough to run on a laptop CPU or MPS. |
| Vector index | Exact inner-product index (one BLAS matmul, FAISS-compatible API) | Exact cosine search over 23k vectors takes a few ms. We first used FAISS but dropped it: on macOS its OpenMP runtime deadlocks against PyTorch's in the same process, and a hang during a live demo is not an option. The API matches FAISS, so an ANN index can be swapped in at scale. |
| Classifier | Per head: TF-IDF (word + char) LR, EmbeddingGemma LR, and a similarity-weighted kNN over precedents. Blend weights (macro-F1) and temperature (NLL) are fitted on a KB validation slice. | The lexical, semantic and instance-based views make different mistakes, so blending them helps. The temperature makes the reported confidence meaningful. |
| Critical head | Binary priority=high ∧ type=Incident | Gives the gate one calibrated risk probability. Its threshold is tuned for ≥90 % recall on KB validation. |
| Generator | Gemma 3 4B IT, pulled with `ollama pull gemma3:4b` and served locally by Ollama | Support tickets contain personal data, so drafting runs on the operator's machine instead of a hosted API. The model is a few GB, fits comfortably in laptop memory, and needs no API key. |
| UI | Streamlit | One file, quick to iterate on, and judges can paste tickets live. |

## Escalation gate

| Code | Severity | Fires when |
|---|---|---|
| `CRITICAL_INCIDENT` | 🔴 on-call | Predicted priority=high and type=Incident |
| `CRITICAL_RISK` | 🔴 on-call | P(critical) ≥ a recall-tuned threshold, which catches incidents the argmax missed |
| `SECURITY_INCIDENT` / `SECURITY_SENSITIVE` | 🔴 on-call / 🟠 review | Breach, ransomware, unauthorized access … (EN + DE). On-call if the ticket is predicted to be an Incident, otherwise a human must answer. |
| `CLINICAL_IMPACT` | 🔴 on-call | Clinical context (hospital, EMR, telemedicine …) and outage language on an Incident or Problem. Clinical words alone are ordinary vocabulary in this dataset, so they don't trigger it. |
| `MAJOR_OUTAGE_CLUSTER` | 🔴 on-call | The burst detector finds ≥3 near-duplicate incident tickets within 15 min |
| `LOW_CONFIDENCE` | 🟠 review | Routing confidence √(P(type)·P(queue)) is below the 15th percentile on validation |
| `NO_PRECEDENT` | 🟠 review | Top-1 precedent similarity is below the in-domain floor (out of scope) |
| `UNSUPPORTED_ACTION` | 🟠 review | The ticket asks for a refund, discount, SLA guarantee, legal action or GDPR deletion, and no retrieved resolution supports it. Legal requests always need a human. |
| `PROMPT_INJECTION` | 🟠 review | "Ignore previous instructions" and similar, in EN or DE |
| `NO_CITATION` / `HALLUCINATED_CITATION` | 🟠 review | The draft cites nothing, or cites IDs that were not retrieved. Those IDs are stripped. |
| `UNSUPPORTED_CLAIM` | 🟠 review | The draft contains money, percentages or durations that appear in no precedent and not in the ticket |
| `LANGUAGE_MISMATCH` | 🟠 review | The draft language differs from the requester's |
| `UNGROUNDED` / `PARTIAL_GROUNDING` | 🟠 review | Gemma itself reports that the precedents don't cover the request |

## Leakage controls

- The split follows the brief exactly: shuffle with `random_state=42`, assign `TKT-xxxxx` IDs, and take the first 80 % as the KB.
- The holdout is written as two files: `test_inputs.parquet` (ticket_id, subject, body) and `test_labels.parquet`. Holdout answers are never written to disk. Only `scripts/evaluate.py` reads the labels, and only to score.
- Training and every tuned threshold use only the KB. 90 % of it is for fitting and 10 % for validation. The vector index contains KB tickets only.
- `tests/test_data.py` asserts that the holdout input file has no answer or label columns and that no ticket ID appears in both splits.

## Bonus features

- **Cross-lingual precedents**: EmbeddingGemma uses one space for both languages. German tickets regularly retrieve English resolutions, and the reply is still written in German.
- **Outage burst detection**: a sliding-window similarity cluster raises a pulsing Major Incident banner and pages on-call. The sidebar has a button that simulates the burst.
- **Operator translation**: German drafts get a streamed English translation from the local Gemma model, so English-speaking operators can review them. It's labelled operator-only. The customer still receives the German reply, and the grounding checks run on that original text.
- **PII guardrail**: everything runs locally, and on top of that, emails, phone numbers, IPv4/IPv6 addresses, IBANs, card numbers, account or customer IDs and secrets are masked before any text reaches the LLM.

## Quick start

Requires Python 3.12, [uv](https://docs.astral.sh/uv/) and [Ollama](https://ollama.com). Install Ollama and open it before you start.

1. Install the dependencies:

   ```bash
   uv sync
   ```

2. Download the local drafting model (a few GB, one time):

   ```bash
   ollama pull gemma3:4b
   ```

3. Get access to EmbeddingGemma (one time). Log in at huggingface.co, accept the license on the `google/embeddinggemma-300m` model page, then run:

   ```bash
   uv run hf auth login
   ```

4. Build everything: dataset, split, embeddings, training, index and holdout evaluation (around 30 minutes on a laptop):

   ```bash
   uv run python scripts/build_all.py
   ```

5. Start the console:

   ```bash
   uv run streamlit run app.py
   ```

No API key is needed. With no `GEMINI_API_KEY` set and no MLX model downloaded, the app uses Ollama automatically, and the sidebar shows `Gemma backend: ollama:gemma3:4b`. To force the backend, set `SD_LLM_BACKEND` in `.env` (see `sentineldesk/config.py` for the accepted values).

Other entry points:

```bash
uv run python scripts/evaluate.py --gen 30
uv run pytest -q
```

The first adds a grounded-generation audit on 30 holdout tickets. The second runs the unit tests (gate, PII, grounding checks, leakage).

### Other supported backends

The code also supports Gemma through the Gemini API (`GEMINI_API_KEY`) and Gemma 4 on Apple Silicon through MLX, and the repo includes a Dockerfile for Cloud Run. This setup deliberately uses none of them: the Gemini API would send ticket text to a hosted service, and running Gemma through Ollama keeps the whole pipeline on the local machine.

## Repository layout

```
sentineldesk/
  config.py      paths, model names, thresholds (env-overridable)
  data.py        load, clean, brief-compliant split, null-safe ticket text
  embeddings.py  EmbeddingGemma encoder
  retriever.py   exact cosine precedent search
  classifier.py  multi-head blended + calibrated triage model
  gate.py        escalation rules and reason codes
  generator.py   grounded prompt, parsing, post-generation verification
  llm.py         Gemma backends (Ollama used here; Gemini API and MLX also supported)
  pii.py         PII masking
  burst.py       outage burst detector
  lang.py        language detection
  pipeline.py    end-to-end orchestration and latency accounting
  samples.py     demo tickets
scripts/         build_all, build_embeddings, train, evaluate, export_embedder, list_gemma_models
app.py           Streamlit Live Triage Console
reports/         metrics.json from the last evaluation
tests/           pytest suite
```

## Dataset & license

[Tobi-Bueck/customer-support-tickets](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets), file `aa_dataset-tickets-multi-lang-5-2-50-version.csv` (28,587 tickets, EN/DE), CC-BY-NC-4.0. The HF page lists about 61.8k rows because it counts several files. We use the single recommended file so that the multi-version files don't put near-duplicate tickets into both splits.