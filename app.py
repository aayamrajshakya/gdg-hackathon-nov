"""SentinelDesk - Live Triage Console (Streamlit).

    uv run streamlit run app.py
"""

from __future__ import annotations

import html
import json
import re
import time

import pandas as pd
import streamlit as st

from sentineldesk import config, gate
from sentineldesk.generator import display_text
from sentineldesk.samples import DEMO_TICKETS, OUTAGE_BURST

st.set_page_config(page_title="SentinelDesk · Triage Console", page_icon="🛡️", layout="wide",
                   initial_sidebar_state="expanded")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
:root {
  --sd-panel: #131a26; --sd-border: rgba(148,163,184,.16);
  --sd-text: #e6e9ef; --sd-muted: #94a3b8; --sd-accent: #6366f1; --sd-accent-2: #22d3ee;
  --sd-green: #22c55e; --sd-amber: #f59e0b; --sd-red: #ef4444;
}
html, body, .stMarkdown, .stTextInput, .stTextArea, button {font-family: 'Inter', system-ui, sans-serif;}
.block-container {padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1360px;}
h1, h2, h3 {letter-spacing: -.02em;}

/* header */
.sd-header {display:flex; align-items:center; justify-content:space-between; gap:16px; flex-wrap:wrap;
            padding: 4px 0 18px 0; border-bottom: 1px solid var(--sd-border); margin-bottom: 18px;}
.sd-header h1 {margin:0; padding:0; font-size: 1.65rem; font-weight: 700;}
.sd-header .sub {color: var(--sd-muted); font-size: .92rem; margin-top: 2px;}
.sd-chips {display:flex; gap:8px; flex-wrap:wrap;}
.sd-chip {display:inline-flex; align-items:center; gap:6px; font-size:.78rem; font-weight:500;
          padding:5px 10px; border-radius:999px; border:1px solid var(--sd-border); background: var(--sd-panel);}
.sd-dot {width:8px; height:8px; border-radius:50%; display:inline-block;}
.sd-dot.ok {background: var(--sd-green); box-shadow: 0 0 0 3px rgba(34,197,94,.18);}
.sd-dot.bad {background: var(--sd-red); box-shadow: 0 0 0 3px rgba(239,68,68,.18);}

/* KPI strip */
.sd-kpis {display:grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap:12px; margin-bottom: 18px;}
@media (max-width: 900px) {.sd-kpis {grid-template-columns: repeat(2, minmax(0,1fr));}}
.sd-kpi {background: var(--sd-panel); border:1px solid var(--sd-border); border-radius:12px; padding:14px 16px;}
.sd-kpi .k {font-size:.72rem; text-transform:uppercase; letter-spacing:.08em; color: var(--sd-muted); font-weight:600;}
.sd-kpi .v {font-size:1.55rem; font-weight:700; margin-top:4px; font-variant-numeric: tabular-nums;}
.sd-kpi .h {font-size:.78rem; color: var(--sd-muted); margin-top:2px;}
.sd-kpi.g .v {color:#4ade80;} .sd-kpi.a .v {color:#fbbf24;} .sd-kpi.r .v {color:#f87171;}

/* status banner */
.sd-badge {display:flex; gap:14px; align-items:flex-start; border-radius: 12px; padding: 14px 18px;
           margin: 4px 0 12px 0; border:1px solid;}
.sd-badge .icon {font-size: 1.4rem; line-height:1.2;}
.sd-badge h3 {margin: 0; padding:0; font-size: 1.08rem; font-weight: 650;}
.sd-badge p {margin: 3px 0 0 0; color: var(--sd-muted); font-size: .88rem;}
.sd-auto   {border-color: rgba(34,197,94,.45);  background: linear-gradient(90deg, rgba(34,197,94,.14), rgba(34,197,94,.03));}
.sd-auto h3 {color:#4ade80;}
.sd-review {border-color: rgba(245,158,11,.45); background: linear-gradient(90deg, rgba(245,158,11,.14), rgba(245,158,11,.03));}
.sd-review h3 {color:#fbbf24;}
.sd-oncall {border-color: rgba(239,68,68,.5);   background: linear-gradient(90deg, rgba(239,68,68,.18), rgba(239,68,68,.04));}
.sd-oncall h3 {color:#f87171;}
.sd-outage {background: linear-gradient(90deg, rgba(127,29,29,.85), rgba(153,27,27,.55)); color:#fff;
            border-radius:12px; padding:14px 18px; margin-bottom:16px; border:1px solid #f87171;
            animation: pulse 1.8s infinite;}
.sd-outage ul {margin:6px 0 0 0; padding-left: 20px; font-size: .88rem; opacity:.92;}
@keyframes pulse {0%{box-shadow:0 0 0 0 rgba(239,68,68,.55);}70%{box-shadow:0 0 0 14px rgba(239,68,68,0);}100%{box-shadow:0 0 0 0 rgba(239,68,68,0);}}
@media (prefers-reduced-motion: reduce) {.sd-outage {animation:none;}}

.sd-reasons {display:flex; flex-direction:column; gap:6px; margin: 0 0 14px 0;}
.sd-reason {font-size:.86rem; display:flex; align-items:center; gap:8px; padding:7px 10px; border-radius:8px;
            background: var(--sd-panel); border:1px solid var(--sd-border);}
.sd-reason.crit {border-left: 3px solid var(--sd-red);}
.sd-reason.warn {border-left: 3px solid var(--sd-amber);}
.sd-code {font-family: 'JetBrains Mono', ui-monospace, monospace; font-size:.74rem; padding:2px 7px; border-radius:6px;
          background: rgba(148,163,184,.14); white-space:nowrap;}
.sd-cite {font-family: 'JetBrains Mono', ui-monospace, monospace; font-size:.78rem; background: rgba(99,102,241,.22);
          color:#c7d2fe; border:1px solid rgba(99,102,241,.45); padding:0 6px; border-radius:6px;}

/* native st.metric tiles */
[data-testid="stMetric"] {background: var(--sd-panel); border:1px solid var(--sd-border); border-radius:12px;
                          padding: 12px 14px; height: 100%;}
[data-testid="stMetricLabel"] p {font-size:.72rem !important; text-transform:uppercase; letter-spacing:.07em;
                                 color: var(--sd-muted); font-weight:600;}
[data-testid="stMetricValue"] {font-size: 1.35rem !important; font-weight: 700;}

/* reply / precedent cards */
.sd-reply {background: var(--sd-panel); border:1px solid var(--sd-border); border-radius:12px; padding:16px 18px;
           white-space: pre-wrap; line-height:1.6; font-size: .95rem;}
.sd-translation {border-style: dashed; background: transparent;}
.sd-checks {display:flex; flex-wrap:wrap; gap:8px; margin: 12px 0 6px 0;}
.sd-check {font-size:.8rem; padding:5px 10px; border-radius:999px; border:1px solid var(--sd-border); background: var(--sd-panel);}
.sd-check.ok  {border-color: rgba(34,197,94,.45); color:#86efac;}
.sd-check.bad {border-color: rgba(239,68,68,.45); color:#fca5a5;}
.sd-check.neu {color: var(--sd-muted);}
.sd-prec {background: var(--sd-panel); border:1px solid var(--sd-border); border-radius:12px; padding:14px 16px;
          margin-bottom:12px; transition: border-color .15s ease;}
.sd-prec:hover {border-color: rgba(99,102,241,.55);}
.sd-prec .top {display:flex; justify-content:space-between; align-items:center; gap:10px; flex-wrap:wrap;}
.sd-prec .meta {font-size:.78rem; color: var(--sd-muted);}
.sd-prec .subj {font-weight:600; margin: 4px 0 2px 0;}
.sd-prec .res {margin-top:8px; padding-top:8px; border-top:1px solid var(--sd-border); font-size:.88rem;}
.sd-bar {height:6px; border-radius:3px; background:rgba(148,163,184,.16); margin:8px 0;}
.sd-bar > div {height:6px; border-radius:3px; background: linear-gradient(90deg, var(--sd-accent), var(--sd-accent-2));}
.sd-small {font-size:.82rem; color: var(--sd-muted);}
.sd-sim {font-family:'JetBrains Mono', monospace; font-size:.8rem; color:#a5b4fc;}

/* empty state */
.sd-empty {text-align:center; padding: 34px 20px; border:1px dashed var(--sd-border); border-radius:14px;
           background: radial-gradient(ellipse at top, rgba(99,102,241,.12), transparent 70%);}
.sd-empty .big {font-size: 2.2rem;}
.sd-empty h3 {margin: 6px 0 4px 0;}
.sd-empty p {color: var(--sd-muted); margin:0 auto; max-width: 560px;}
.sd-steps {display:grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap:12px; margin-top:18px; text-align:left;}
@media (max-width: 800px) {.sd-steps {grid-template-columns: 1fr;}}
.sd-step {background: var(--sd-panel); border:1px solid var(--sd-border); border-radius:10px; padding:12px 14px; font-size:.86rem;}
.sd-step b {display:block; margin-bottom:2px;}
.sd-step span {color: var(--sd-muted);}

/* streamlit chrome */
section[data-testid="stSidebar"] h3 {font-size:.72rem; text-transform:uppercase; letter-spacing:.09em;
                                     color: var(--sd-muted); margin-top: .9rem; font-weight:600;}
.stTabs [data-baseweb="tab-list"] {gap: 4px; border-bottom: 1px solid var(--sd-border);}
.stTabs [data-baseweb="tab"] {padding: 8px 14px;}
[data-testid="stForm"] {background: var(--sd-panel); border:1px solid var(--sd-border); border-radius:14px; padding: 18px;}
.sd-section {font-size:.75rem; text-transform:uppercase; letter-spacing:.09em; color: var(--sd-muted);
             font-weight:600; margin: 24px 0 8px 0;}
#MainMenu, footer {visibility:hidden;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

STATUS_UI = {
    gate.AUTO_DRAFT: ("sd-auto", "✅", "Auto-Draft Ready",
                      "Confident triage, strong precedents, draft passed all grounding checks."),
    gate.HUMAN_REVIEW: ("sd-review", "🟠", "Human Review Required",
                        "An agent must review before anything is sent."),
    gate.ESCALATE_ONCALL: ("sd-oncall", "🚨", "Escalate to On-Call Engineer",
                           "Critical incident - page on-call now. Draft is for the engineer only."),
}
STATUS_SHORT = {gate.AUTO_DRAFT: "✅ Auto-draft", gate.HUMAN_REVIEW: "🟠 Review",
                gate.ESCALATE_ONCALL: "🚨 On-call"}
LANG_FLAG = {"en": "🇬🇧 EN", "de": "🇩🇪 DE"}
UNCERTAIN_BELOW = 0.60  # display only: labels under this confidence are marked as guesses
UNCERTAIN_HELP = ("The model is less than 60% confident in this label - treat it as a guess. "
                  "This marker is display-only; the gate's thresholds are unchanged.")


def label_tile(col, name: str, pred: dict, upper: bool = False) -> None:
    """Metric tile for a predicted label; flags low-confidence predictions as uncertain."""
    value = pred["label"].upper() if upper else pred["label"]
    conf = pred["confidence"]
    if conf < UNCERTAIN_BELOW:
        col.metric(f"{name} · ⚠️ uncertain", value, f"{conf:.0%} conf · a guess",
                   delta_color="off", help=UNCERTAIN_HELP)
    else:
        col.metric(name, value, f"{conf:.0%} conf", delta_color="off")


def inbox_label(pred: dict) -> str:
    return pred["label"] + (" (?)" if pred["confidence"] < UNCERTAIN_BELOW else "")


@st.cache_resource(show_spinner="Loading EmbeddingGemma, triage model, precedent index and Gemma…")
def load_desk():
    from sentineldesk.pipeline import SentinelDesk

    desk = SentinelDesk()
    if desk.llm:
        desk.llm.warmup()  # load local Gemma weights now, not on the first ticket
    return desk


def esc(s: str) -> str:
    return html.escape(s or "")


def highlight_citations(reply: str) -> str:
    return re.sub(r"\[?(TKT-\d{5})\]?", r'<span class="sd-cite">\1</span>', esc(reply))


def check_pill(ok: bool | None, text: str) -> str:
    cls = "neu" if ok is None else ("ok" if ok else "bad")
    return f'<span class="sd-check {cls}">{text}</span>'


def section(title: str) -> None:
    st.markdown(f'<div class="sd-section">{title}</div>', unsafe_allow_html=True)


# ------------------------------------------------------------------------------- state
ss = st.session_state
ss.setdefault("history", [])
ss.setdefault("subject", "")
ss.setdefault("body", "")
ss.setdefault("result", None)
ss.setdefault("pending_draft", False)

try:
    desk = load_desk()
except FileNotFoundError:
    st.error("Artifacts missing. Run `uv run python scripts/build_all.py` first.")
    st.stop()


def load_sample(name: str) -> None:
    ss.subject, ss.body = DEMO_TICKETS[name]


def clear_input() -> None:
    ss.subject, ss.body = "", ""


# ------------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("## 🛡️ SentinelDesk")
    st.caption("Multilingual incident triage · escalation gate · Gemma-grounded replies")
    llm = desk.llm

    st.markdown("### Demo scenarios")
    pick = st.selectbox("Load a sample ticket", ["—"] + list(DEMO_TICKETS), label_visibility="collapsed")
    if st.button("Load sample", icon=":material/inbox:", width="stretch", disabled=pick == "—"):
        load_sample(pick)
        st.rerun()

    st.markdown("### Outage burst simulation")
    if st.button("Simulate 4 VPN failure tickets", icon=":material/bolt:", width="stretch"):
        now = time.time()
        for i, (s, b) in enumerate(OUTAGE_BURST):
            r = desk.triage(s, b, generate=False, now=now + i)
            ss.history.insert(0, r)
        st.rerun()
    if st.button("Reset burst window", icon=":material/restart_alt:", width="stretch"):
        desk.burst.reset()
        st.rerun()

    st.markdown("### Output")
    gen_on = st.toggle("Generate Gemma draft", value=True)
    translate_on = st.toggle("English translation of German drafts", value=True,
                             help="Operator-only translation; the customer still gets the German reply.")

    with st.expander("Gate thresholds", icon=":material/tune:"):
        t = desk.thresholds
        t["min_confidence"] = st.slider("Min routing confidence", 0.0, 1.0,
                                        float(t["min_confidence"]), 0.01)
        t["min_similarity"] = st.slider("Min precedent similarity", 0.0, 1.0,
                                        float(t["min_similarity"]), 0.01)
        t["critical_risk"] = st.slider("Critical-incident risk", 0.0, 1.0,
                                       float(t["critical_risk"]), 0.01)
        st.caption("Defaults were tuned on a validation slice of the KB (never the holdout).")

    with st.expander("System", icon=":material/memory:"):
        if llm:
            st.success(f"Gemma backend: `{llm.describe()}`")
        else:
            st.error("No Gemma backend - set GEMINI_API_KEY, or install mlx-lm / Ollama.")
        st.caption(f"Embeddings: `{config.EMBED_MODEL}`")
        st.caption(f"KB: {len(desk.retriever.kb):,} resolved tickets")

    if ss.history and st.button("Clear session", icon=":material/delete_sweep:", width="stretch",
                                type="tertiary"):
        ss.history, ss.result, ss.pending_draft = [], None, False
        st.rerun()

# ------------------------------------------------------------------------------- header
backend_chip = (f'<span class="sd-chip"><span class="sd-dot ok"></span>Gemma · {esc(llm.describe())}</span>'
                if llm else '<span class="sd-chip"><span class="sd-dot bad"></span>No LLM backend</span>')
st.markdown(
    f'<div class="sd-header"><div><h1>🛡️ Live Triage Console</h1>'
    f'<div class="sd-sub sub">Route, gate and draft grounded replies for EN / DE support tickets.</div></div>'
    f'<div class="sd-chips">{backend_chip}'
    f'<span class="sd-chip">📚 {len(desk.retriever.kb):,} precedents</span>'
    f'<span class="sd-chip">🧬 {esc(config.EMBED_MODEL.split("/")[-1])}</span></div></div>',
    unsafe_allow_html=True)

# session KPIs
H = ss.history
n = len(H)
n_auto = sum(h.gate.status == gate.AUTO_DRAFT for h in H)
n_review = sum(h.gate.status == gate.HUMAN_REVIEW for h in H)
n_oncall = sum(h.gate.status == gate.ESCALATE_ONCALL for h in H)
avg_lat = (sum(h.latency_ms.get("triage", 0) for h in H) / n) if n else 0
st.markdown(
    '<div class="sd-kpis">'
    f'<div class="sd-kpi"><div class="k">Tickets triaged</div><div class="v">{n}</div>'
    f'<div class="h">avg triage {avg_lat:.0f} ms</div></div>'
    f'<div class="sd-kpi g"><div class="k">Auto-draft</div><div class="v">{n_auto}</div>'
    f'<div class="h">{(n_auto / n if n else 0):.0%} deflection</div></div>'
    f'<div class="sd-kpi a"><div class="k">Human review</div><div class="v">{n_review}</div>'
    f'<div class="h">{(n_review / n if n else 0):.0%} of session</div></div>'
    f'<div class="sd-kpi r"><div class="k">On-call escalations</div><div class="v">{n_oncall}</div>'
    f'<div class="h">{(n_oncall / n if n else 0):.0%} of session</div></div>'
    '</div>', unsafe_allow_html=True)

burst_now = next((r.burst for r in ss.history[:1] if r.burst.get("active")), None)
if burst_now:
    members = "".join(f"<li>{esc(m['subject'])}</li>" for m in burst_now["members"][-4:])
    st.markdown(
        f'<div class="sd-outage" role="alert"><b>🚨 MAJOR INCIDENT - possible outage.</b> '
        f'{burst_now["size"]} similar tickets in the last {burst_now["window_min"]:.0f} min '
        f'(queue: {esc(burst_now["queue"])}). On-call paged.<ul>{members}</ul></div>',
        unsafe_allow_html=True)

# ------------------------------------------------------------------------------- input
with st.form("ticket", border=False):
    c1, c2 = st.columns([1, 3], gap="medium")
    with c1:
        st.text_input("Subject", key="subject", placeholder="Optional - e.g. VPN down")
        st.caption("Language is auto-detected; replies match the customer's language. "
                   "PII is masked before the LLM sees it.")
    with c2:
        st.text_area("Ticket body", key="body", height=150,
                     placeholder="Paste any customer ticket in English or German…")
    b1, b2 = st.columns([4, 1])
    submitted = b1.form_submit_button("Triage ticket", icon=":material/play_arrow:", type="primary",
                                      width="stretch")
    b2.form_submit_button("Clear", icon=":material/close:", width="stretch", on_click=clear_input)

if submitted:
    if not (ss.body.strip() or ss.subject.strip()):
        st.toast("Please enter a ticket body first.", icon="⚠️")
    else:
        r = desk.analyze(ss.subject, ss.body)  # ~50-200 ms: triage, retrieval, gate
        ss.result = r
        ss.pending_draft = gen_on
        ss.history.insert(0, r)
        st.rerun()

r = ss.result
if r is None:
    st.markdown(
        '<div class="sd-empty"><div class="big">🛰️</div><h3>No ticket triaged yet</h3>'
        '<p>Paste a ticket above, or try one of the demo scenarios below. SentinelDesk classifies it, '
        'retrieves resolved precedents, decides whether it is safe to auto-draft, and writes a cited reply.</p>'
        '<div class="sd-steps">'
        '<div class="sd-step"><b>1 · Triage</b><span>Type, queue, priority and critical-risk in ~100 ms.</span></div>'
        '<div class="sd-step"><b>2 · Gate</b><span>Auto-draft, human review, or page on-call.</span></div>'
        '<div class="sd-step"><b>3 · Draft</b><span>Gemma reply grounded in cited precedents.</span></div>'
        '</div></div>', unsafe_allow_html=True)
    samples = list(DEMO_TICKETS)[:4]
    if samples:
        section("Quick start")
        qc = st.columns(len(samples))
        for col, name in zip(qc, samples):
            col.button(name, key=f"qs_{name}", width="stretch", on_click=load_sample, args=(name,))
else:
    cls, icon, title, sub = STATUS_UI[r.gate.status]
    if ss.pending_draft:
        sub = "Provisional - Gemma is drafting; grounding checks run when it finishes."
    section(f"Result · {esc(r.ref)}")
    st.markdown(f'<div class="sd-badge {cls}" role="status"><div class="icon">{icon}</div>'
                f'<div><h3>{title}</h3><p>{sub}</p></div></div>', unsafe_allow_html=True)
    if r.gate.reasons:
        items = "".join(
            f'<div class="sd-reason {"crit" if reason.severity == gate.CRITICAL else "warn"}">'
            f'<span class="sd-code">{reason.code}</span>{esc(reason.message)}</div>'
            for reason in r.gate.reasons)
        st.markdown(f'<div class="sd-reasons">{items}</div>', unsafe_allow_html=True)

    p = r.predictions
    m = st.columns(6)
    label_tile(m[0], "Type", p["type"])
    label_tile(m[1], "Queue", p["queue"])
    label_tile(m[2], "Priority", p["priority"], upper=True)
    m[3].metric("Critical risk", f"{r.critical_risk:.0%}",
                f"threshold {desk.thresholds['critical_risk']:.0%}", delta_color="off")
    m[4].metric("Language", LANG_FLAG.get(r.language, r.language.upper()),
                f"reply in {r.reply_language.upper()}", delta_color="off")
    m[5].metric("Latency", f"{r.latency_ms['total'] / 1000:.2f}s",
                f"triage {r.latency_ms['triage']:.0f}ms",
                delta_color="off")
    lat = " · ".join(f"{k} {v:.0f}ms" for k, v in r.latency_ms.items()
                     if k not in ("total", "triage", "translate"))
    st.caption(f"Routing confidence {r.routing_confidence:.0%} · {lat}"
               + (f" · 🔒 {r.pii['masked']} PII item(s) masked before LLM" if r.pii["masked"] else ""))

    tab_draft, tab_prec, tab_probs, tab_json = st.tabs(
        [":material/edit_note: Draft reply", f":material/library_books: Precedents ({len(r.precedents)})",
         ":material/bar_chart: Probabilities", ":material/data_object: JSON"])

    with tab_draft:
        d = r.draft
        if ss.pending_draft:
            st.caption("✍️ Gemma is drafting from the retrieved precedents…")
            box = st.empty()
            acc = ""
            for piece in desk.draft_stream(r):  # updates r.draft / r.gate when done
                acc += piece
                box.markdown(f'<div class="sd-reply">{highlight_citations(display_text(acc))}▌</div>',
                             unsafe_allow_html=True)
            ss.pending_draft = False
            st.rerun()
        elif d is None:
            st.warning(r.llm_error or "Draft generation disabled.", icon=":material/info:")
        else:
            if r.gate.status == gate.ESCALATE_ONCALL:
                st.caption("⚠️ Internal draft for the on-call engineer - not sent automatically.")
            elif r.gate.status == gate.HUMAN_REVIEW:
                st.caption("⚠️ Draft requires agent approval before sending.")
            st.markdown(f'<div class="sd-reply">{highlight_citations(d.reply)}</div>',
                        unsafe_allow_html=True)
            st.markdown(
                '<div class="sd-checks">'
                + check_pill(bool(d.cited), f"{'✓' if d.cited else '✗'} Citations: {esc(', '.join(d.cited)) or 'none'}")
                + check_pill(not d.unsupported_claims,
                             "✓ No unsupported numbers" if not d.unsupported_claims
                             else f"✗ Unsupported numbers: {esc(', '.join(d.unsupported_claims))}")
                + check_pill(None, f"Grounding: {esc(str(d.grounding))}")
                + check_pill(None, f"{esc(d.backend)}:{esc(d.model)} · {d.latency_ms / 1000:.1f}s")
                + '</div>', unsafe_allow_html=True)
            if d.invalid_citations:
                st.error(f"Removed hallucinated citations: {', '.join(d.invalid_citations)}")
            if d.grounding_note:
                st.caption(f"Grounding note: {d.grounding_note}")

            a1, a2, _ = st.columns([1, 1, 3])
            a1.download_button("Download reply", d.reply, file_name=f"{r.ref}-reply.txt",
                               icon=":material/download:", width="stretch")
            with a2.popover("Copy text", icon=":material/content_copy:", width="stretch"):
                st.code(d.reply, language=None, wrap_lines=True)

            if translate_on and desk.needs_translation(r):
                section("🇬🇧 English translation · operators only, not sent to the customer")
                if d.translation_en:
                    st.markdown(f'<div class="sd-reply sd-translation">'
                                f'{highlight_citations(d.translation_en)}</div>', unsafe_allow_html=True)
                elif d.translation_error:
                    st.caption(f"Translation unavailable: {d.translation_error[:160]}")
                else:
                    tbox = st.empty()
                    tbox.caption("Translating with Gemma…")
                    acc = ""
                    for piece in desk.translate_stream(r):  # fills d.translation_en when done
                        acc += piece
                        tbox.markdown(f'<div class="sd-reply sd-translation">'
                                      f'{highlight_citations(acc)}▌</div>', unsafe_allow_html=True)
                    st.rerun()

    with tab_prec:
        if not r.precedents:
            st.info("No precedents retrieved for this ticket.", icon=":material/search_off:")
        for prec in r.precedents:
            used = r.draft is not None and prec.ticket_id in r.draft.cited
            st.markdown(
                f'<div class="sd-prec"><div class="top"><div><b>{prec.ticket_id}</b> '
                f'{"<span class=sd-cite>cited</span>" if used else ""}</div>'
                f'<span class="sd-sim">sim {prec.similarity:.3f}</span></div>'
                f'<div class="meta">{LANG_FLAG.get(prec.language, prec.language)} · {esc(prec.type)} · '
                f'{esc(prec.queue)} · {esc(prec.priority)}</div>'
                f'<div class="sd-bar"><div style="width:{max(0, prec.similarity) * 100:.0f}%"></div></div>'
                f'<div class="subj">{esc(prec.subject) or "<i>(no subject)</i>"}</div>'
                f'<span class="sd-small">{esc(prec.body[:260])}{"…" if len(prec.body) > 260 else ""}</span>'
                f'<div class="res"><b>Resolution:</b> {esc(prec.answer[:420])}'
                f'{"…" if len(prec.answer) > 420 else ""}</div></div>',
                unsafe_allow_html=True)

    with tab_probs:
        cols = st.columns(4)
        for col, head in zip(cols, ("type", "queue", "priority", "critical")):
            with col, st.container(border=True):
                st.markdown(f"**{head.capitalize()}**")
                probs = p[head]["probs"]
                st.bar_chart(pd.DataFrame({"probability": list(probs.values())},
                                          index=list(probs.keys())), horizontal=True, height=220,
                             color="#6366f1")

    with tab_json:
        payload = json.dumps(r.to_dict(), indent=2, ensure_ascii=False, default=str)
        st.download_button("Download JSON", payload, file_name=f"{r.ref}.json", mime="application/json",
                           icon=":material/download:")
        st.code(payload, language="json")

# ------------------------------------------------------------------------------- history
if ss.history:
    hc1, hc2 = st.columns([3, 2], vertical_alignment="bottom")
    with hc1:
        section(f"Session inbox · {len(ss.history)} ticket(s)")
    with hc2:
        filt = st.segmented_control(
            "Filter", ["All", "✅ Auto-draft", "🟠 Review", "🚨 On-call"], default="All",
            label_visibility="collapsed", key="inbox_filter") or "All"
    rows = [{
        "status": STATUS_SHORT.get(h.gate.status, h.gate.status), "ref": h.ref,
        "subject": (h.subject or h.body)[:80],
        "type": inbox_label(h.predictions["type"]),
        "queue": inbox_label(h.predictions["queue"]), "priority": inbox_label(h.predictions["priority"]),
        "risk": h.critical_risk, "lang": LANG_FLAG.get(h.language, h.language),
        "burst": "🚨" if h.burst.get("active") else "",
    } for h in ss.history[:50]]
    if filt != "All":
        rows = [x for x in rows if x["status"] == filt]
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(
            df, width="stretch", hide_index=True,
            column_config={
                "status": st.column_config.TextColumn("Status", width="small"),
                "ref": st.column_config.TextColumn("Ref", width="small"),
                "subject": st.column_config.TextColumn("Subject", width="large"),
                "risk": st.column_config.ProgressColumn("Critical risk", format="percent",
                                                        min_value=0.0, max_value=1.0, width="small"),
                "lang": st.column_config.TextColumn("Lang", width="small"),
                "burst": st.column_config.TextColumn("Burst", width="small"),
            })
        st.download_button("Export inbox CSV", df.to_csv(index=False), file_name="sentineldesk-inbox.csv",
                           mime="text/csv", icon=":material/table_view:")
    else:
        st.caption("No tickets match this filter.")
