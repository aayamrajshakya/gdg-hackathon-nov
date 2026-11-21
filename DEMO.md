# SentinelDesk: 5-minute demo script

**Before going on stage**
- `uv run streamlit run app.py`, open http://localhost:8501, and wait for the sidebar to show the Gemma backend.
- Triage one throwaway ticket so every model is warm.
- Close other heavy apps. On an 8 GB Mac, local Gemma 4 and the browser compete for RAM.
- If `GEMINI_API_KEY` is set, the hosted Gemma 3 27B answers in about 3 s; otherwise MLX Gemma 4 takes 15–25 s.

| Time | Action | What to say |
|---|---|---|
| 0:00 | Show the empty console | "Helpdesks drown in a mixed inbox, and LLM auto-replies hallucinate. SentinelDesk triages, decides whether AI may answer at all, and only drafts replies grounded in resolved tickets, with citations." |
| 0:30 | **🚨 Hospital VPN outage** → Triage | The badge turns red within about 200 ms, before Gemma has written anything. Point out `CLINICAL_IMPACT`: the classifier said *medium*, but clinical context plus outage language pages on-call anyway, as a defence in depth. The draft streams in as an internal note for the on-call engineer, with cited interim steps. |
| 1:30 | **🧾 Rechnungsfrage (DE)** | Language is auto-detected and the reply comes back in German. Open the Precedents tab: similarity bars and resolution snippets, with the cited tickets marked. Mention the cross-lingual matching (EN/DE flags). Under the German draft, the **🇬🇧 English translation** streams in for operators (it is not sent to the customer). |
| 2:15 | **🧨 Prompt injection + refund demand** | `PROMPT_INJECTION` and `NO_PRECEDENT` both fire. The caption shows 🔒 3 PII items masked before the LLM saw the text. The draft doesn't promise the $4,999 or the 20 %. Any invented number would be flagged as `UNSUPPORTED_CLAIM`. |
| 3:00 | **💡 Pre-sales SAP / SLA question** | `UNSUPPORTED_ACTION`: no precedent supports an uptime SLA. Gemma marks its own draft PARTIAL and says a specialist will follow up instead of inventing an SAP connector. |
| 3:30 | Sidebar → **Simulate 4 VPN failure tickets** | The pulsing MAJOR INCIDENT banner appears. Each ticket alone looked routine (`HUMAN_REVIEW`), but together they form a cluster in EN and DE. |
| 4:00 | Metrics (README or `reports/metrics.json`) | Holdout macro-F1: type 0.94, queue 0.79, priority 0.83. The gate catches 92 % of high-priority incidents while escalating 22 %. Be upfront about the near-duplicate analysis: on *novel* tickets the numbers fall to 0.88 / 0.56 / 0.65, and the blend still beats TF-IDF by 8–20 points. |
| 4:30 | Wrap up | "All Gemma: EmbeddingGemma for retrieval and classification features, Gemma 4 on-device via MLX or Gemma 27B on Google AI. Every guardrail can be explained, and every threshold was tuned on the KB, never on the holdout." |

**Likely judge questions**
- *Leakage?* Holdout answers are never written to disk. Labels live in a separate file that only `evaluate.py` reads. All thresholds and blend weights come from a validation slice of the KB.
- *Why is kNN so strong?* The dataset contains paraphrased and translated near-duplicates (41 % of holdout tickets have a KB neighbour with similarity ≥ 0.95). We report the novel-ticket slice separately so the headline numbers aren't inflated.
- *Calibration?* Each head has a temperature fitted on validation. Holdout ECE is 0.005 for type, 0.07 for queue and 0.02 for priority.
- *Why not FAISS?* Its OpenMP runtime deadlocks against PyTorch's on macOS. Exact BLAS search over 23k vectors takes 2 ms, and the interface matches FAISS so it can be swapped at scale.
- *Unseen ticket types?* `NO_PRECEDENT` uses a similarity floor that sits between in-domain tickets and out-of-scope probes, so the system abstains instead of improvising.
