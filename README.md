# SentinelDesk: Confidence & Risk Escalation Gate

The escalation gate for **SentinelDesk**, our hackathon incident-triage copilot. For every incoming support ticket, the gate decides whether the AI may draft a reply on its own, or whether a human has to step in.

| Verdict | Badge shown in the UI | Meaning |
| --- | --- | --- |
| `AUTO_DRAFT` | **Auto-Draft Ready** | Safe for the AI to write a grounded reply. |
| `HUMAN_REVIEW` | **Needs Human Review** | A person should check before anything is sent. |
| `ESCALATE_ONCALL` | **Escalate to On-Call Engineer** | Urgent. Page someone now. |

Every decision comes with the reasons behind it and the list of past ticket IDs the reply generator is allowed to cite.

---

## Contents

- [What's in this repo](#whats-in-this-repo)
- [Quick start](#quick-start)
- [Using the gate in the app](#using-the-gate-in-the-app)
- [How the gate decides](#how-the-gate-decides)
- [How tuning works](#how-tuning-works)
- [Tuning with the team's real models](#tuning-with-the-teams-real-models)
- [Reading the tuning report](#reading-the-tuning-report)
- [Command-line options](#command-line-options)
- [Running the tests](#running-the-tests)
- [Troubleshooting](#troubleshooting)
- [Known limitations](#known-limitations)

---

## What's in this repo

| File | What it does |
| --- | --- |
| `sentinel_gate.py` | The gate itself. Pure Python, no dependencies. Import it from any backend (Streamlit, FastAPI, etc.). |
| `tune_gate.py` | Learns the gate's thresholds from the dataset and writes them to `gate_config.json`. It can also score the full system on the holdout set. |
| `test_sentinel_gate.py` | Tests for the gate's rules. |
| `test_tune_gate.py` | Tests for the data loading, splitting, and threshold-picking logic. |
| `requirements.txt` | Packages needed by `tune_gate.py`. The gate itself needs none. |

**Dataset:** [Tobi-Bueck/customer-support-tickets](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets) on Hugging Face (about 61.8k English and German tickets, CC-BY-NC-4.0).

---

## Quick start

### 1. Get the dataset

```bash
git lfs install
git clone https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets
```

> If you cloned without Git LFS, the CSVs are tiny placeholder files. Run `git lfs pull` inside the cloned folder. The script detects this and tells you.

### 2. Set up Python

Use a virtual environment so every teammate runs the same Python:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

> **Windows:** run these in PowerShell or Command Prompt, not an MSYS2 or MinGW terminal. If `python` points to the wrong install, use `py` instead (e.g. `py -m pip install ...`). See [Troubleshooting](#troubleshooting).

### 3. Tune the gate

```bash
python tune_gate.py --data-dir ./customer-support-tickets --eval-test
```

This produces two files:

- **`gate_config.json`**: the tuned thresholds the app loads.
- **`tuning_report.json`**: scores and details about how each rule behaves.

---

## Using the gate in the app

```python
from sentinel_gate import (
    GateConfig, HeadPrediction, RetrievedTicket, TriageResult, evaluate_gate,
)

config = GateConfig.load("gate_config.json")

# From the classifier: label + calibrated confidence per head (full probabilities optional)
triage = TriageResult(
    type=HeadPrediction("Incident", 0.91, {"Incident": 0.91, "Request": 0.05, ...}),
    queue=HeadPrediction("IT Support", 0.78),
    priority=HeadPrediction("high", 0.88, {"high": 0.88, "medium": 0.10, "low": 0.02}),
)

# From retrieval: top-k similar past tickets from the knowledge base
retrieved = [
    RetrievedTicket(ticket_id="TKT-01234", similarity=0.82, answer="Restart the VPN gateway..."),
    RetrievedTicket(ticket_id="TKT-05678", similarity=0.79, answer="Update the router firmware..."),
]

decision = evaluate_gate(subject, body, triage, retrieved, config)
```

What you get back:

| Field | Use it for |
| --- | --- |
| `decision.badge` | The status badge in the console. |
| `decision.status` | `AUTO_DRAFT`, `HUMAN_REVIEW`, or `ESCALATE_ONCALL`. |
| `decision.allow_auto_draft` | `True` only when the reply generator may run. |
| `decision.reasons` | Each rule that fired, with a readable message. Show these next to the badge. |
| `decision.supporting_ticket_ids` | The only ticket IDs the generated reply may cite. |
| `decision.requested_actions` / `unsupported_actions` | Risky actions the customer asked for, and which ones have no precedent. |
| `decision.to_dict()` | JSON-ready version for APIs and logging. |

`subject` can be `None`, empty, or a pandas `NaN`. Null subjects are handled.

---

## How the gate decides

Every rule runs on every ticket. **The most serious verdict wins**, and all reasons are reported, so the operator sees the full picture.

| Rule | Fires when... | Verdict |
| --- | --- | --- |
| `high_priority_incident` | The classifier predicts priority **high** (or **critical**) **and** type **Incident**. | On-call |
| `probable_high_priority_incident` | P(high) × P(Incident) is above a tuned cutoff, even if the top guess was something else. Catches near-misses. | On-call |
| `critical_impact_keywords` | The text mentions outages, breaches, hospitals, patients, etc. (English and German). | On-call if the classifier agrees it's urgent or an Incident/Problem; otherwise review |
| `low_classifier_confidence` | Any head (type, queue, priority) is below its confidence threshold. | Review |
| `no_precedent` | Retrieval returned nothing. | Review |
| `low_retrieval_similarity` | Even the best past ticket isn't similar enough. | Review |
| `insufficient_precedent` | Too few past tickets clear the similarity bar. | Review |
| `unsupported_action` | The customer asks for a refund, pricing, a legal or SLA guarantee, data deletion, or a security bypass, and no similar past resolution did that. | Review |
| `prompt_injection` | The ticket tries to instruct the AI ("ignore all previous instructions..."). | Review |
| `insufficient_text` | The body is empty or too short to triage. | Review |

**Why these rules?** The problem statement names three escalation triggers: high-priority incidents, low classifier or retrieval confidence, and requests unsupported by past resolutions. The remaining rules cover the two failures it warns about, missed outages and AI replies that invent refunds, prices, or guarantees, plus the adversarial and empty tickets judges will try.

**Priority levels:** the dataset has five priority values. The gate treats anything in `high_labels` (default: `high`, `critical`) as high, and the tuning script collapses the data to high / medium / low.

---

## How tuning works

The gate depends on numbers like "how confident is confident enough?" and "how similar is similar enough?" `tune_gate.py` learns them from data while following the hackathon's leakage rules.

```
Full dataset
├── 80%  Knowledge base (kb_train_df)  ← everything below happens in here
│   ├── 80%  fit    → trains the classifier, becomes the search index
│   ├── 10%  calib  → makes confidence scores honest (temperature scaling)
│   └── 10%  tune   → picks every gate threshold
└── 20%  Holdout (test_df)  ← only used for final scoring, with --eval-test
```

1. **Split.** Uses the exact 80/20 snippet from the problem statement (`random_state=42`), so ticket IDs match everyone else's. Rows with no body or missing labels are dropped *after* the split, so IDs aren't shifted.
2. **Train.** Fits a baseline classifier (TF-IDF + logistic regression per head) and a TF-IDF search index on the `fit` slice. Past tickets with no answer are left out of the index, since there's nothing to cite.
3. **Calibrate.** Adjusts each head so its confidence matches reality (a ticket at 0.8 is right about 80% of the time). The report shows calibration error before and after.
4. **Pick thresholds on the `tune` slice:**
   - **Confidence, per head:** the lowest cutoff where the auto-trusted tickets hit a target accuracy. If that would auto-trust too few tickets, it relaxes to a minimum coverage and prints a warning.
   - **Similarity floors:** set so 90% of normal tickets clear them.
   - **Probable-incident cutoff:** the strictest value that still catches 90% of true high-priority Incidents.
5. **Report.** Runs the tuned gate on the `tune` slice and records how often each rule fires.
6. **Score (optional, `--eval-test`).** Macro-F1 per head, High-Priority Incident Recall for the classifier and for the gate, and per-ticket latency, all on the holdout. Holdout labels are only used for scoring, and **holdout answers are never read or indexed**.

Thresholds are always rounded *down*, and confidence thresholds are capped at 0.99, so rounding never rejects a ticket that tuning would have accepted.

---

## Tuning with the team's real models

The classifier and search inside `tune_gate.py` are **stand-ins** so the script works on its own. Thresholds only fit the models they were tuned with, so once the real models are ready, **re-tune with them**, and re-tune again any time either one changes.

### Rule 1: train only on the `fit` slice

If the real classifier has already seen the `tune` slice, it will look more accurate than it is, and the gate will trust it too much. Use the same carve the script uses:

```python
from tune_gate import load_and_split

kb, test, raw = load_and_split("path/to/aa_dataset-tickets-multi-lang-5-2-50-version.csv")
kb = kb.sample(frac=1.0, random_state=7).reset_index(drop=True)
fit = kb.iloc[: int(0.8 * len(kb))]   # train the classifier and build the index on THIS only
```

Everyone on the team should load data through `load_and_split()`, so ticket IDs match across the classifier, the index, and the citations.

### Rule 2: wrap the models in the same shape

```python
class TeamTriage:
    classes = {"type": [...], "queue": [...], "priority": [...]}   # label names, in column order

    def predict_heads(self, texts):
        # {"type": probs, "queue": probs, "priority": probs}
        # each probs: numpy array, shape (len(texts), len(classes[head]))
        ...

class TeamRetriever:
    def retrieve_topk(self, texts, k=5):
        # (indexes, similarities), each a numpy array of shape (len(texts), k), best first
        ...

    def to_hits(self, idx_row, sim_row):
        # list of RetrievedTicket(ticket_id, similarity, answer) for one ticket
        ...
```

Then, inside `tune()` in `tune_gate.py`, replace the lines that build `BaselineTriage` and `BaselineRetriever` with the team's versions. If the team's classifier already outputs calibrated probabilities, the calibration step can be skipped. Leaving it on is harmless.

### Checklist before the demo

- [ ] Final classifier and retriever are frozen.
- [ ] Both were trained on the `fit` slice only.
- [ ] `tune_gate.py --eval-test` was run with those exact models.
- [ ] `high_incident_recall_oncall` in the report looks good (around 0.9+).
- [ ] The app loads the newest `gate_config.json`.

---

## Reading the tuning report

`tuning_report.json` contains:

| Key | What it tells you |
| --- | --- |
| `calibration` | Temperature and calibration error (ECE) per head, before and after. |
| `heads` | Each head's chosen threshold, the share of tickets it auto-trusts (`coverage`), accuracy on that share, and whether the target had to be relaxed. |
| `retrieval` | Typical best-match similarity and the chosen similarity floors. |
| `soft_high_incident` | The probable-incident cutoff, the recall it reaches, and how many extra non-incidents it pages. |
| `gate_on_tune` | Share of tickets per verdict, and gate recall on true high-priority Incidents. |
| `rule_fire_rates` | How often each rule fires. **Start here if too few tickets get auto-drafted.** |
| `critical_keyword_precision` | Of the tickets flagged by emergency keywords, the share that really were high priority. |
| `holdout` | (with `--eval-test`) Macro-F1 per head, classifier and gate incident recall, and latency. |

**If almost nothing gets auto-drafted:** find the rule with the biggest fire rate in `rule_fire_rates`. Queue confidence is the usual cause, since the dataset has 52 queues and noisy labels. Loosen it with `--acc-queue` or `--min-coverage`.

**If emergencies slip through:** raise `--target-hi-recall`.

---

## Command-line options

| Option | Default | What it does |
| --- | --- | --- |
| `--data-dir PATH` | *(required, or `--csv`)* | Folder containing the cloned dataset. Finds the recommended CSV automatically. |
| `--csv PATH` | | Use a specific CSV file instead. |
| `--out PATH` | `gate_config.json` | Where to save the tuned config. |
| `--report PATH` | `tuning_report.json` | Where to save the report. |
| `--k N` | `5` | Past tickets retrieved per ticket. |
| `--acc-type` | `0.85` | Target accuracy for auto-trusted type predictions. |
| `--acc-queue` | `0.80` | Target accuracy for auto-trusted queue predictions. |
| `--acc-priority` | `0.80` | Target accuracy for auto-trusted priority predictions. |
| `--min-coverage` | `0.4` | Each head auto-trusts at least this share of tickets, even if that means relaxing its accuracy target. |
| `--sim-quantile` | `0.10` | Similarity floors are set so this share of normal tickets fall below them. |
| `--target-hi-recall` | `0.90` | Share of true high-priority Incidents the gate must send to on-call. |
| `--eval-test` | off | Also score everything on the 20% holdout. |
| `--save-models PATH` | | Save the baseline classifier and index with joblib, e.g. for the UI. |

Example, a looser setup that auto-drafts more:

```bash
python tune_gate.py --data-dir ./customer-support-tickets --acc-queue 0.7 --min-coverage 0.5 --eval-test
```

---

## Running the tests

The tests are written for pytest:

```bash
python -m pip install pytest
python -m pytest
```

They cover every gate rule (including German input, null subjects, and config save/load), the official split, priority normalization, Git LFS detection, calibration, and the threshold-picking logic.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'numpy'`, but pip says "Requirement already satisfied"**

`pip` and `python` belong to different Python installs. Always install through the Python you run with:

```bash
python -m pip install -r requirements.txt
```

To see which Python is running:

```bash
python -c "import sys; print(sys.executable)"
```

**The path contains `msys64`**

That's MSYS2's own Python, usually installed alongside a C toolchain. Don't uninstall it.
- If you're in an MSYS2 or MinGW terminal, type `exit` and open PowerShell instead.
- In PowerShell, use the `py` launcher: `py -m pip install -r requirements.txt`, then `py tune_gate.py ...`.
- To fix it permanently, move your regular Python entries above the `C:\msys64\...` entries in the Windows `Path` environment variable, then reopen the terminal.

**`... is a Git LFS pointer, not the data`**

Run `git lfs install && git lfs pull` inside the cloned dataset folder.

**`No CSV files found under ...`**

Check the `--data-dir` path, or point at a file directly with `--csv`.

---

## Known limitations

- **Risky-request detection uses keyword patterns** (English and German). It's fast and explainable, but it can miss paraphrases like "make this right financially." An LLM classifier could replace `detect_actions()` without changing anything else.
- **Emergency keywords can over-flag.** For example, a billing question from a hospital customer goes to human review. Check `critical_keyword_precision` in the report, or set `use_critical_keywords` to `false` in `gate_config.json`.
- **Thresholds belong to specific models.** A config tuned with the baseline models isn't valid for a different classifier or embedding model. Re-tune whenever either changes.
- **The dataset is synthetic and its labels are noisy.** Expect modest Macro-F1, especially for queue. That's why the gate leans on several signals instead of trusting the classifier alone.
