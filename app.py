"""SentinelDesk - Live Triage Console (Streamlit).

    uv run streamlit run app.py
"""

from __future__ import annotations

import html
import json
import re
import time

import streamlit as st

from sentineldesk import config, gate
from sentineldesk.generator import display_text
from sentineldesk.samples import DEMO_TICKETS, OUTAGE_BURST

st.set_page_config(page_title="SentinelDesk", page_icon="🛡️", layout="wide")

# Colour roles follow the dataviz reference palette (dark mode): one accent ramp for magnitude
# (confidence, similarity, probability), status colours only for state, always with icon + label.
CSS = """
<style>
:root {
  --sd-page:#111110; --sd-surface:#1a1a19; --sd-surface-2:#222220; --sd-border:rgba(255,255,255,.10);
  --sd-ink:#f0efec; --sd-ink-2:#c3c2b7; --sd-muted:#898781; --sd-grid:#2c2c2a;
  --sd-accent:#3987e5; --sd-good:#0ca30c; --sd-warn:#fab219; --sd-crit:#d03b3b;
  --sd-sidebar:#141413; --sd-chip-bg:rgba(255,255,255,.08); --sd-deemph:#4b4a46;
  --sd-font: system-ui, -apple-system, "Segoe UI", sans-serif;
}
/* only our own components get the system font; Streamlit's Material icon font stays intact */
[class^="sd-"], [class*=" sd-"] { font-family: var(--sd-font); }
.block-container { padding-top: 1.1rem; max-width: 1360px; }
header[data-testid="stHeader"] { background: transparent; }
[data-testid="stAppDeployButton"], footer { display: none !important; }
[data-testid="stSidebar"] { background: var(--sd-sidebar); border-right: 1px solid var(--sd-border); }
[data-testid="stForm"] { background: var(--sd-surface); border: 1px solid var(--sd-border);
  border-radius: 14px; padding: 14px 18px 16px; }
.stButton button, [data-testid="stFormSubmitButton"] button { border-radius: 10px; font-weight: 600; }
[data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid var(--sd-border); }
[data-baseweb="tab"] { padding: 8px 12px; }
.sd-muted { color: var(--sd-muted); } .sd-ink2 { color: var(--sd-ink-2); }

/* ---------- header */
.sd-hero { display:flex; align-items:center; justify-content:space-between; gap:16px; flex-wrap:wrap;
  margin: 0 0 14px; }
.sd-brand { display:flex; align-items:center; gap:12px; }
.sd-logo { width:44px; height:44px; border-radius:12px; display:grid; place-items:center; font-size:24px;
  background: color-mix(in srgb, var(--sd-accent) 22%, var(--sd-surface)); border:1px solid var(--sd-border); }
.sd-title { font-size: 1.55rem; font-weight: 700; color: var(--sd-ink); line-height: 1.15; }
.sd-subtitle { font-size: .86rem; color: var(--sd-muted); margin-top: 2px; }
.sd-chips { display:flex; gap:8px; flex-wrap:wrap; }
.sd-chip { display:inline-flex; align-items:center; gap:7px; font-size:.78rem; color: var(--sd-ink-2);
  background: var(--sd-surface); border:1px solid var(--sd-border); border-radius: 999px; padding: 5px 11px; }
.sd-dot { width:8px; height:8px; border-radius:50%; display:inline-block; flex:none; }
.sd-dot.good { background: var(--sd-good); } .sd-dot.warn { background: var(--sd-warn); }
.sd-dot.crit { background: var(--sd-crit); } .sd-dot.off { background: var(--sd-muted); }

/* ---------- outage banner */
.sd-outage { border:1px solid color-mix(in srgb, var(--sd-crit) 55%, transparent); border-left: 4px solid var(--sd-crit);
  background: color-mix(in srgb, var(--sd-crit) 14%, var(--sd-surface)); border-radius: 12px; padding: 12px 16px;
  margin: 0 0 14px; color: var(--sd-ink); }
.sd-outage-head { display:flex; align-items:center; gap:10px; flex-wrap:wrap; }
.sd-outage-meta { color: var(--sd-ink-2); font-size: .86rem; }
.sd-outage ul { margin: 6px 0 0 22px; padding: 0; color: var(--sd-ink-2); font-size: .86rem; }
.sd-pulse { width:10px; height:10px; border-radius:50%; background: var(--sd-crit); animation: sdpulse 1.6s infinite; }
@keyframes sdpulse { 0%{box-shadow:0 0 0 0 rgba(208,59,59,.7);} 70%{box-shadow:0 0 0 10px rgba(208,59,59,0);}
  100%{box-shadow:0 0 0 0 rgba(208,59,59,0);} }

/* ---------- status banner (+ reasons) */
.sd-badge { --st: var(--sd-accent); border-radius: 14px; margin: 16px 0 12px; overflow: hidden;
  border: 1px solid color-mix(in srgb, var(--st) 45%, transparent); border-left: 4px solid var(--st);
  background: color-mix(in srgb, var(--st) 11%, var(--sd-surface)); }
.sd-st-good { --st: var(--sd-good); } .sd-st-warn { --st: var(--sd-warn); } .sd-st-crit { --st: var(--sd-crit); }
.sd-badge-main { display:flex; align-items:center; gap:14px; padding: 14px 18px; }
.sd-badge-icon { width:34px; height:34px; border-radius:50%; display:grid; place-items:center; flex:none;
  background: var(--st); color:#fff; font-weight:800; font-size:18px; }
.sd-st-warn .sd-badge-icon { color:#0b0b0b; }
.sd-badge-title { font-size: 1.12rem; font-weight: 700; color: var(--sd-ink); }
.sd-badge-sub { font-size: .87rem; color: var(--sd-ink-2); margin-top: 2px; }
.sd-badge-ref { margin-left:auto; font: 500 .75rem ui-monospace, monospace; color: var(--sd-muted); }
.sd-reasons { border-top: 1px solid var(--sd-border); padding: 8px 18px 12px; display:grid; gap:6px; }
.sd-reason { display:flex; align-items:baseline; gap:9px; font-size: .86rem; color: var(--sd-ink-2); }
.sd-reason .sd-dot { position: relative; top: 1px; }
.sd-code { font: 600 .74rem ui-monospace, monospace; color: var(--sd-ink); padding: 2px 7px; border-radius: 6px;
  background: var(--sd-chip-bg); border: 1px solid var(--sd-border); white-space: nowrap; }
.sd-live { display:inline-block; width:8px; height:8px; border-radius:50%; background: var(--sd-accent);
  margin-right:6px; animation: sdblink 1s infinite alternate; }
@keyframes sdblink { from{opacity:.25;} to{opacity:1;} }

/* ---------- stat tiles */
.sd-tiles { display:grid; grid-template-columns: 1fr 1.55fr 1fr 1fr 1fr 1fr; gap: 10px; margin: 2px 0 8px; }
@media (max-width: 1100px) { .sd-tiles { grid-template-columns: repeat(3, minmax(0,1fr)); } }
.sd-tile { background: var(--sd-surface); border:1px solid var(--sd-border); border-radius: 12px; padding: 11px 13px 10px;
  min-width: 0; }
.sd-tile-label { display:flex; align-items:center; justify-content:space-between; gap:6px; font-size:.76rem;
  color: var(--sd-muted); font-weight: 500; }
.sd-tile-value { font-size: 1.32rem; font-weight: 650; color: var(--sd-ink); line-height: 1.2; margin: 4px 0 8px;
  overflow-wrap: anywhere; }
.sd-tile-sub { font-size: .74rem; color: var(--sd-muted); margin-top: 6px; }
.sd-pill { font-size:.66rem; font-weight:700; padding: 2px 7px; border-radius: 999px; white-space: nowrap; }
.sd-pill-warn { background: var(--sd-warn); color: #0b0b0b; }
.sd-pill-crit { background: var(--sd-crit); color: #fff; }
.sd-pill-accent { background: color-mix(in srgb, var(--sd-accent) 25%, transparent); color: var(--sd-ink); }
.sd-meter { --c: var(--sd-accent); position: relative; height: 6px; border-radius: 3px;
  background: color-mix(in srgb, var(--c) 22%, transparent); }
.sd-meter > i { display:block; height:100%; border-radius:3px; background: var(--c); }
.sd-tick { position:absolute; top:-3px; width:2px; height:12px; border-radius:1px; background: var(--sd-ink-2); }
.sd-details { font-size: .78rem; color: var(--sd-muted); margin: 4px 0 6px; }
.sd-details b { color: var(--sd-ink-2); font-weight: 600; }

/* ---------- draft */
.sd-note { font-size:.84rem; color: var(--sd-ink-2); margin: 6px 0 8px; display:flex; gap:8px; align-items:center; }
.sd-card { background: var(--sd-surface); border:1px solid var(--sd-border); border-radius: 14px; overflow:hidden; margin-bottom: 10px; }
.sd-card-head { display:flex; justify-content:space-between; gap:10px; flex-wrap:wrap; padding: 9px 16px;
  border-bottom: 1px solid var(--sd-border); font-size: .78rem; color: var(--sd-muted); }
.sd-card-head b { color: var(--sd-ink-2); font-weight: 600; }
.sd-reply { padding: 16px 18px; line-height: 1.6; color: var(--sd-ink); font-size: .95rem; }
.sd-card > .sd-reply { border: 0; }
.sd-stream { background: var(--sd-surface); border:1px solid var(--sd-border); border-radius: 14px; }
.sd-translation { color: var(--sd-ink-2); }
.sd-card-tr { border-style: dashed; }
.sd-cite { font: 600 .76rem ui-monospace, monospace; background: color-mix(in srgb, var(--sd-accent) 30%, transparent);
  color: var(--sd-ink); border: 1px solid color-mix(in srgb, var(--sd-accent) 60%, transparent);
  padding: 1px 6px; border-radius: 6px; white-space: nowrap; }
.sd-checks { display:flex; gap:8px; flex-wrap:wrap; margin: 4px 0 10px; }
.sd-check { display:inline-flex; align-items:center; gap:7px; font-size:.8rem; color: var(--sd-ink-2);
  background: var(--sd-surface); border:1px solid var(--sd-border); border-radius: 999px; padding: 5px 11px; }
.sd-check b { color: var(--sd-ink); font-weight: 600; }
.sd-ic { width:18px; height:18px; border-radius:50%; display:grid; place-items:center; font-size:11px; font-weight:800; flex:none; }
.sd-ic.good { background: var(--sd-good); color:#fff; } .sd-ic.warn { background: var(--sd-warn); color:#0b0b0b; }
.sd-ic.crit { background: var(--sd-crit); color:#fff; } .sd-ic.off { background: var(--sd-grid); color: var(--sd-ink-2); }

/* ---------- precedents */
.sd-legend { font-size:.78rem; color: var(--sd-muted); margin: 4px 0 10px; }
.sd-precs { display:grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap: 10px; }
@media (max-width: 1100px) { .sd-precs { grid-template-columns: 1fr; } }
.sd-prec { background: var(--sd-surface); border:1px solid var(--sd-border); border-radius: 12px; padding: 12px 14px; min-width:0; }
.sd-prec.cited { border-left: 3px solid var(--sd-accent); }
.sd-prec-top { display:flex; align-items:center; gap:8px; margin-bottom: 7px; }
.sd-prec-id { font: 700 .82rem ui-monospace, monospace; color: var(--sd-ink); }
.sd-prec-sim { margin-left:auto; font-weight: 650; color: var(--sd-ink); font-size: .9rem; }
.sd-prec-meta { font-size:.76rem; color: var(--sd-muted); margin: 8px 0 6px; }
.sd-prec-subj { font-weight: 600; color: var(--sd-ink); font-size:.9rem; }
.sd-prec-body { font-size:.82rem; color: var(--sd-muted); margin: 3px 0 8px; }
.sd-prec-res { font-size:.85rem; color: var(--sd-ink-2); }
.sd-prec-res b { color: var(--sd-ink); }
.sd-prec details { margin-top: 6px; font-size: .82rem; color: var(--sd-ink-2); }
.sd-prec summary { cursor: pointer; color: var(--sd-accent); }
.sd-tag { font-size:.68rem; color: var(--sd-ink-2); border:1px solid var(--sd-border); border-radius: 999px; padding: 1px 7px; }

/* ---------- probabilities */
.sd-probs { display:grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: 10px; }
@media (max-width: 1100px) { .sd-probs { grid-template-columns: repeat(2, minmax(0,1fr)); } }
.sd-prob-card { background: var(--sd-surface); border:1px solid var(--sd-border); border-radius: 12px; padding: 12px 14px; }
.sd-prob-title { font-size:.82rem; font-weight: 650; color: var(--sd-ink); margin-bottom: 8px; }
.sd-prob-row { margin: 7px 0; }
.sd-prob-line { display:flex; justify-content:space-between; gap:8px; font-size:.78rem; color: var(--sd-ink-2); margin-bottom: 3px; }
.sd-prob-line b { color: var(--sd-ink); font-weight: 650; }
.sd-bar { height: 8px; border-radius: 0 4px 4px 0; background: var(--sd-grid); }
.sd-bar > i { display:block; height:100%; border-radius: 0 4px 4px 0; background: var(--sd-deemph); }
.sd-bar.top > i { background: var(--sd-accent); }

/* ---------- inbox table */
.sd-table-wrap { border:1px solid var(--sd-border); border-radius: 12px; overflow-x: auto; background: var(--sd-surface); }
.sd-table { width:100%; border-collapse: collapse; font-size: .82rem; color: var(--sd-ink-2); }
.sd-table th { text-align:left; font-weight:600; font-size:.72rem; color: var(--sd-muted); padding: 9px 12px;
  border-bottom: 1px solid var(--sd-border); white-space: nowrap; }
.sd-table td { padding: 8px 12px; border-bottom: 1px solid var(--sd-border); white-space: nowrap; }
.sd-table tr:last-child td { border-bottom: 0; }
.sd-table tbody tr:hover { background: var(--sd-chip-bg); }
.sd-table td.ref { font: 500 .76rem ui-monospace, monospace; color: var(--sd-ink); }
.sd-table td.subj { white-space: normal; min-width: 260px; color: var(--sd-ink); }
.sd-status { display:inline-flex; align-items:center; gap:7px; color: var(--sd-ink); font-weight: 600; }
.sd-risk { display:flex; align-items:center; gap:8px; min-width: 110px; }
.sd-risk .sd-meter { flex: 1; }
.sd-q { color: var(--sd-warn); font-weight: 700; margin-left: 2px; }

/* ---------- empty state + sidebar */
.sd-empty { display:grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap: 12px; margin-top: 16px; }
@media (max-width: 900px) { .sd-empty { grid-template-columns: 1fr; } }
.sd-feature { background: var(--sd-surface); border:1px solid var(--sd-border); border-radius: 14px; padding: 16px; }
.sd-feature-k { font-size: 1.4rem; } .sd-feature-t { font-weight: 650; color: var(--sd-ink); margin: 6px 0 4px; }
.sd-feature-d { font-size: .84rem; color: var(--sd-muted); line-height: 1.45; }
.sd-side { background: var(--sd-surface); border:1px solid var(--sd-border); border-radius: 12px; padding: 10px 12px;
  font-size:.8rem; display:grid; gap:7px; }
.sd-side-k { color: var(--sd-muted); font-size:.72rem; }
.sd-side-v { color: var(--sd-ink); font: 500 .76rem ui-monospace, monospace; overflow-wrap:anywhere; }
.sd-section { font-size:.72rem; letter-spacing:.06em; text-transform: uppercase; color: var(--sd-muted);
  margin: 16px 0 6px; font-weight: 600; }
</style>
"""
st.html(CSS)

# Light theme: same roles, light steps (dataviz reference palette), plus overrides for Streamlit's
# own widgets so native and custom elements always match.
LIGHT_CSS = """
<style>
:root {
  --sd-page:#f6f6f3; --sd-surface:#ffffff; --sd-surface-2:#f1f0ec; --sd-border:rgba(11,11,11,.12);
  --sd-ink:#0b0b0b; --sd-ink-2:#52514e; --sd-muted:#6f6e69; --sd-grid:#e1e0d9;
  --sd-accent:#2a78d6; --sd-sidebar:#efeee9; --sd-chip-bg:rgba(11,11,11,.05); --sd-deemph:#b9b7ae;
}
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] { background: var(--sd-page); color: var(--sd-ink); }
[data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label, [data-testid="stCheckbox"] p,
[data-testid="stMarkdownContainer"] > p, [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p,
[data-testid="stSliderTickBarMin"], [data-testid="stSliderTickBarMax"], [data-testid="stAlert"] p,
[data-baseweb="tab"] p { color: var(--sd-ink-2); }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: var(--sd-muted); }
[data-baseweb="tab"][aria-selected="true"] p { color: var(--sd-accent); }
[data-testid="stTextInputRootElement"], [data-testid="stTextAreaRootElement"], [data-baseweb="input"],
[data-baseweb="textarea"], [data-baseweb="base-input"] { background: #ffffff !important; border-color: rgba(11,11,11,.18) !important; }
[data-testid="stTextInputRootElement"] input, [data-testid="stTextAreaRootElement"] textarea {
  background: #ffffff !important; color: var(--sd-ink) !important; -webkit-text-fill-color: var(--sd-ink) !important;
  caret-color: var(--sd-ink); }
input::placeholder, textarea::placeholder { color: #8d8b84 !important; -webkit-text-fill-color: #8d8b84 !important; }
[data-testid="stBaseButton-secondary"], [data-testid="stButtonGroup"] button {
  background: var(--sd-surface) !important; color: var(--sd-ink) !important; border: 1px solid rgba(11,11,11,.16) !important; }
[data-testid="stBaseButton-secondary"] p, [data-testid="stButtonGroup"] button p { color: var(--sd-ink) !important; }
[data-testid="stBaseButton-secondary"]:hover, [data-testid="stButtonGroup"] button:hover {
  border-color: var(--sd-accent) !important; background: color-mix(in srgb, var(--sd-accent) 6%, #fff) !important; }
[data-testid="stSidebarCollapseButton"] button, [data-testid="stMainMenu"] button,
[data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapsedControl"] button { color: var(--sd-ink-2) !important; }
[data-testid="stCode"] pre { background: var(--sd-surface-2) !important; }
[data-testid="stBaseButton-primaryFormSubmit"] p, [data-testid="stBaseButton-primary"] p { color: #ffffff !important; }
[data-testid="stCheckbox"] label:has(input:not(:checked)) > div:first-of-type { background: #c9c7bf !important; }
[data-testid="stAlertContainer"] p { color: var(--sd-ink) !important; }
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentInfo"]) { background: color-mix(in srgb, var(--sd-accent) 10%, #fff) !important; }
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentError"]) { background: color-mix(in srgb, var(--sd-crit) 10%, #fff) !important; }
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentWarning"]) { background: color-mix(in srgb, var(--sd-warn) 14%, #fff) !important; }
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentSuccess"]) { background: color-mix(in srgb, var(--sd-good) 10%, #fff) !important; }
</style>
"""

# status -> (css kind, icon, title, subtitle). Icon + label always accompany the colour.
STATUS_UI = {
    gate.AUTO_DRAFT: ("good", "✓", "Auto-draft ready",
                      "Confident triage, strong precedents, and the draft passed every grounding check."),
    gate.HUMAN_REVIEW: ("warn", "!", "Human review required",
                        "An agent must review before anything is sent."),
    gate.ESCALATE_ONCALL: ("crit", "!", "Escalate to on-call engineer",
                           "Critical incident: page on-call now. The draft is an internal note for the engineer."),
}
LANG_NAME = {"en": "English", "de": "German", "fr": "French", "es": "Spanish", "nl": "Dutch",
             "it": "Italian", "pt": "Portuguese"}
LANG_FLAG = {"en": "🇬🇧", "de": "🇩🇪", "fr": "🇫🇷", "es": "🇪🇸", "nl": "🇳🇱", "it": "🇮🇹", "pt": "🇵🇹"}
HEAD_TITLE = {"type": "Type", "queue": "Queue", "priority": "Priority", "critical": "Critical incident"}
UNCERTAIN_BELOW = 0.60  # display only: labels under this confidence are marked as guesses
UNCERTAIN_HELP = ("The model is less than 60% confident in this label - treat it as a guess. "
                  "This marker is display-only; the gate's thresholds are unchanged.")


# ------------------------------------------------------------------------------- helpers
def H(fragment: str) -> None:
    """Render an HTML fragment. Fragments never contain blank lines, so markdown leaves them alone."""
    st.markdown(fragment, unsafe_allow_html=True)


def esc(s: str) -> str:
    return html.escape(s or "").replace("$", "&#36;")  # '$' would otherwise trigger LaTeX


def multiline(s: str) -> str:
    return esc(s).replace("\n", "<br>")


def highlight_citations(text: str) -> str:
    return re.sub(r"\[?(TKT-\d{5})\]?", r'<span class="sd-cite">\1</span>', multiline(text))


def meter(frac: float, color: str = "var(--sd-accent)", tick: float | None = None, title: str = "") -> str:
    w = max(0.0, min(1.0, frac)) * 100
    t = f'<span class="sd-tick" style="left:calc({tick * 100:.1f}% - 1px)"></span>' if tick is not None else ""
    return f'<div class="sd-meter" style="--c:{color}" title="{esc(title)}"><i style="width:{w:.1f}%"></i>{t}</div>'


def tile(label: str, value: str, sub: str, meter_html: str = "", pill: str = "", title: str = "") -> str:
    return (f'<div class="sd-tile" title="{esc(title)}"><div class="sd-tile-label"><span>{label}</span>{pill}</div>'
            f'<div class="sd-tile-value">{value}</div>{meter_html}<div class="sd-tile-sub">{sub}</div></div>')


def label_tile(name: str, pred: dict, cap: bool = False) -> str:
    """Stat tile for a predicted label with a confidence meter; low confidence is flagged as a guess."""
    value = esc(pred["label"].capitalize() if cap else pred["label"])
    conf = pred["confidence"]
    if conf < UNCERTAIN_BELOW:
        return tile(name, value, f"{conf:.0%} confidence · treat as a guess",
                    meter(conf, "var(--sd-warn)", title=f"{conf:.0%} calibrated confidence"),
                    pill='<span class="sd-pill sd-pill-warn">⚠ uncertain</span>', title=UNCERTAIN_HELP)
    return tile(name, value, f"{conf:.0%} confidence", meter(conf, title=f"{conf:.0%} calibrated confidence"))


def check(kind: str, icon: str, label: str, value: str) -> str:
    return f'<span class="sd-check"><span class="sd-ic {kind}">{icon}</span>{label} <b>{value}</b></span>'


@st.cache_resource(show_spinner="Loading EmbeddingGemma, triage model, precedent index and Gemma…")
def load_desk():
    from sentineldesk.pipeline import SentinelDesk

    desk = SentinelDesk()
    if desk.llm:
        desk.llm.warmup()  # load local Gemma weights now, not on the first ticket
    return desk


def _load_scenario() -> None:
    name = st.session_state.get("scenario")
    if name:
        st.session_state.subject, st.session_state.body = DEMO_TICKETS[name]
    st.session_state.scenario = None


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

llm = desk.llm
kb_size = len(desk.retriever.kb)
if llm:
    prim = llm.primary
    where = {"mlx": "on-device · MLX", "gemini": "Gemini API", "ollama": "Ollama"}.get(prim.name, prim.name)
    backend_chip = f'<span class="sd-dot good"></span>{esc(prim.model.split("/")[-1])} · {where}'
else:
    backend_chip = '<span class="sd-dot crit"></span>No Gemma backend'

# ------------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.toggle("☀️ Light theme", value=False, key="light_theme")  # dark is the designed default
    H('<div class="sd-section">System</div>')
    if llm:
        fallbacks = " → ".join(f"{b.name}:{b.model.split('/')[-1]}" for b in llm.backends[1:]) or "none"
        H(f'<div class="sd-side"><div><div class="sd-side-k">Gemma backend</div>'
          f'<div class="sd-side-v">{esc(llm.primary.name)}:{esc(llm.primary.model)}</div></div>'
          f'<div><div class="sd-side-k">Fallbacks</div><div class="sd-side-v">{esc(fallbacks)}</div></div>'
          f'<div><div class="sd-side-k">Embeddings</div><div class="sd-side-v">{esc(config.EMBED_MODEL)}</div></div>'
          f'<div><div class="sd-side-k">Knowledge base</div><div class="sd-side-v">{kb_size:,} resolved tickets</div></div></div>')
    else:
        st.error("No Gemma backend - set GEMINI_API_KEY, or install mlx-lm / Ollama.")

    H('<div class="sd-section">Outage simulation</div>')
    if st.button("Simulate 4 VPN failure tickets", width="stretch"):
        now = time.time()
        for i, (s, b) in enumerate(OUTAGE_BURST):
            r = desk.triage(s, b, generate=False, now=now + i)
            ss.history.insert(0, r)
        st.rerun()
    if st.button("Reset burst window", width="stretch"):
        desk.burst.reset()
        st.rerun()

    H('<div class="sd-section">Gate thresholds</div>')
    t = desk.thresholds
    t["min_confidence"] = st.slider("Min routing confidence", 0.0, 1.0, float(t["min_confidence"]), 0.01)
    t["min_similarity"] = st.slider("Min precedent similarity", 0.0, 1.0, float(t["min_similarity"]), 0.01)
    t["critical_risk"] = st.slider("Critical-incident risk", 0.0, 1.0, float(t["critical_risk"]), 0.01)
    st.caption("Defaults were tuned on a validation slice of the KB, never the holdout.")

    H('<div class="sd-section">Options</div>')
    gen_on = st.toggle("Generate Gemma draft", value=True)
    translate_on = st.toggle("English translation of German drafts", value=True,
                             help="Operator-only translation; the customer still gets the German reply.")

if ss.get("light_theme"):
    st.html(LIGHT_CSS)

# ------------------------------------------------------------------------------- header
H(f'<div class="sd-hero"><div class="sd-brand"><div class="sd-logo">🛡️</div><div>'
  f'<div class="sd-title">SentinelDesk</div>'
  f'<div class="sd-subtitle">Live triage console · multilingual incident triage, escalation gate and grounded replies</div>'
  f'</div></div><div class="sd-chips"><span class="sd-chip">{backend_chip}</span>'
  f'<span class="sd-chip">EmbeddingGemma-300m</span><span class="sd-chip">KB · {kb_size:,} tickets</span></div></div>')

burst_now = next((r.burst for r in ss.history[:1] if r.burst.get("active")), None)
if burst_now:
    members = "".join(f"<li>{esc(m['subject'])}</li>" for m in burst_now["members"][-4:])
    H(f'<div class="sd-outage"><div class="sd-outage-head"><span class="sd-pulse"></span>'
      f'<b>Major incident: possible outage</b><span class="sd-outage-meta">{burst_now["size"]} similar tickets '
      f'in the last {burst_now["window_min"]:.0f} min · queue {esc(burst_now["queue"])} · on-call paged</span></div>'
      f'<ul>{members}</ul></div>')

# ------------------------------------------------------------------------------- input
st.pills("Try a scenario", list(DEMO_TICKETS), key="scenario", on_change=_load_scenario)
with st.form("ticket"):
    c1, c2 = st.columns([1, 3])
    with c1:
        st.text_input("Subject (optional)", key="subject", placeholder="Leave empty if there is none")
    with c2:
        st.text_area("Ticket body", key="body", height=120,
                     placeholder="Paste any customer ticket in English or German…")
    submitted = st.form_submit_button("Triage ticket ▶", type="primary", width="stretch")

if submitted:
    if not (ss.body.strip() or ss.subject.strip()):
        st.warning("Please enter a ticket body.")
    else:
        r = desk.analyze(ss.subject, ss.body)  # ~50-200 ms: triage, retrieval, gate
        ss.result = r
        ss.pending_draft = gen_on
        ss.history.insert(0, r)
        st.rerun()

r = ss.result
if r is None:
    H('<div class="sd-empty">'
      '<div class="sd-feature"><div class="sd-feature-k">⚡</div><div class="sd-feature-t">Triage in about 0.1 s</div>'
      '<div class="sd-feature-d">Type, queue, priority and critical-incident risk from EmbeddingGemma plus calibrated '
      'classic ML, with a confidence meter on every label.</div></div>'
      '<div class="sd-feature"><div class="sd-feature-k">🛡️</div><div class="sd-feature-t">Explainable escalation</div>'
      '<div class="sd-feature-d">Every decision shows its reason codes: critical incidents, low confidence, no '
      'precedent, unsupported requests, prompt injection, outage bursts.</div></div>'
      '<div class="sd-feature"><div class="sd-feature-k">📚</div><div class="sd-feature-t">Grounded drafts</div>'
      '<div class="sd-feature-d">Gemma 4 replies in the customer\'s language, cites past tickets, and is checked '
      'for invented ticket IDs, numbers and the wrong language.</div></div></div>')
    H('<div class="sd-legend" style="margin-top:12px">Pick a scenario above or paste a ticket, then press '
      '<b>Triage ticket</b>.</div>')
else:
    kind, icon, title, sub = STATUS_UI[r.gate.status]
    if ss.pending_draft:
        sub = '<span class="sd-live"></span>Provisional - Gemma is drafting; grounding checks run when it finishes.'
    reasons = "".join(
        f'<div class="sd-reason"><span class="sd-dot {"crit" if x.severity == gate.CRITICAL else "warn"}"></span>'
        f'<span class="sd-code">{esc(x.code)}</span><span>{esc(x.message)}</span></div>' for x in r.gate.reasons)
    H(f'<div class="sd-badge sd-st-{kind}"><div class="sd-badge-main"><div class="sd-badge-icon">{icon}</div>'
      f'<div><div class="sd-badge-title">{title}</div><div class="sd-badge-sub">{sub}</div></div>'
      f'<div class="sd-badge-ref">{esc(r.ref)}</div></div>'
      + (f'<div class="sd-reasons">{reasons}</div>' if reasons else "") + '</div>')

    p, thr, lat = r.predictions, desk.thresholds["critical_risk"], r.latency_ms
    risk_hot = r.critical_risk >= thr
    risk_tile = tile("Critical risk", f"{r.critical_risk:.0%}", f"pages on-call at {thr:.0%}",
                     meter(r.critical_risk, "var(--sd-crit)" if risk_hot else "var(--sd-accent)", tick=thr,
                           title=f"Calibrated P(high-priority incident) = {r.critical_risk:.0%}; tick = threshold"),
                     pill='<span class="sd-pill sd-pill-crit">above threshold</span>' if risk_hot else "")
    lang_tile = tile("Language", f"{LANG_FLAG.get(r.language, '🌐')} {LANG_NAME.get(r.language, r.language.upper())}",
                     f"reply in {LANG_NAME.get(r.reply_language, r.reply_language)}",
                     meter(r.language_confidence, title=f"Detection confidence {r.language_confidence:.0%}"))
    gen_ms = lat.get("generate")
    lat_sub = ("draft streaming…" if ss.pending_draft else
               f"+ Gemma draft {gen_ms / 1000:.1f} s" if gen_ms else "no draft generated")
    lat_title = " · ".join(f"{k} {v:.0f} ms" for k, v in lat.items() if k not in ("total", "triage"))
    lat_tile = tile("Triage latency", f"{lat['triage'] / 1000:.2f} s", lat_sub, title=lat_title)
    H('<div class="sd-tiles">' + label_tile("Type", p["type"]) + label_tile("Queue", p["queue"])
      + label_tile("Priority", p["priority"], cap=True) + risk_tile + lang_tile + lat_tile + '</div>')
    pii = (f' · 🔒 <b>{r.pii["masked"]}</b> PII item(s) masked before the LLM' if r.pii["masked"] else "")
    H(f'<div class="sd-details">Routing confidence <b>{r.routing_confidence:.0%}</b>{pii} · {esc(lat_title)}</div>')

    tab_draft, tab_prec, tab_probs, tab_json = st.tabs(
        ["✍️ Draft reply", f"📚 Precedents ({len(r.precedents)})", "📊 Probabilities", "{ } JSON"])

    with tab_draft:
        d = r.draft
        if ss.pending_draft:
            H('<div class="sd-note"><span class="sd-live"></span>✍️ Gemma is drafting from the retrieved precedents…</div>')
            box = st.empty()
            acc = ""
            for piece in desk.draft_stream(r):  # updates r.draft / r.gate when done
                acc += piece
                box.markdown(f'<div class="sd-stream sd-reply">{highlight_citations(display_text(acc))}▌</div>',
                             unsafe_allow_html=True)
            ss.pending_draft = False
            st.rerun()
        elif d is None:
            st.info(r.llm_error or "Draft generation is turned off in the sidebar.")
        else:
            if r.gate.status == gate.ESCALATE_ONCALL:
                H('<div class="sd-note">⚠️ Internal draft for the on-call engineer. Not sent automatically.</div>')
            elif r.gate.status == gate.HUMAN_REVIEW:
                H('<div class="sd-note">⚠️ Draft needs agent approval before sending.</div>')
            H(f'<div class="sd-card"><div class="sd-card-head"><span><b>Reply to customer</b> · '
              f'{LANG_FLAG.get(d.reply_language, "")} {LANG_NAME.get(d.reply_language, d.reply_language)}</span>'
              f'<span>{esc(d.model.split("/")[-1])} · {esc(d.backend)} · {d.latency_ms / 1000:.1f} s</span></div>'
              f'<div class="sd-reply">{highlight_citations(d.reply)}</div></div>')
            ground_kind = {"FULL": "good", "PARTIAL": "warn", "NONE": "crit"}.get(d.grounding, "off")
            lang_bad = any(x.code == "LANGUAGE_MISMATCH" for x in d.reasons)
            H('<div class="sd-checks">'
              + (check("good", "✓", "Citations", ", ".join(d.cited)) if d.cited
                 else check("warn", "✗", "Citations", "none"))
              + (check("crit", "✗", "Unsupported numbers", esc(", ".join(d.unsupported_claims)))
                 if d.unsupported_claims else check("good", "✓", "Unsupported numbers", "none"))
              + check(ground_kind, {"good": "✓", "warn": "!", "crit": "✗"}.get(ground_kind, "?"),
                      "Gemma self-check", esc(d.grounding))
              + check("warn" if lang_bad else "good", "✗" if lang_bad else "✓", "Language",
                      LANG_NAME.get(d.reply_language, d.reply_language))
              + '</div>')
            if d.invalid_citations:
                st.error(f"Removed hallucinated citations: {', '.join(d.invalid_citations)}")
            if d.grounding_note:
                H(f'<div class="sd-details">Gemma\'s grounding note: {esc(d.grounding_note)}</div>')
            if translate_on and desk.needs_translation(r):
                head = ('<div class="sd-card-head"><span><b>🇬🇧 English translation</b> · for operators only, '
                        'not sent to the customer</span></div>')
                if d.translation_en:
                    H(f'<div class="sd-card sd-card-tr">{head}<div class="sd-reply sd-translation">'
                      f'{highlight_citations(d.translation_en)}</div></div>')
                elif d.translation_error:
                    st.caption(f"Translation unavailable: {d.translation_error[:160]}")
                else:
                    tbox = st.empty()
                    tbox.markdown('<div class="sd-note"><span class="sd-live"></span>Translating with Gemma…</div>',
                                  unsafe_allow_html=True)
                    acc = ""
                    for piece in desk.translate_stream(r):  # fills d.translation_en when done
                        acc += piece
                        tbox.markdown(f'<div class="sd-card sd-card-tr">{head}<div class="sd-reply sd-translation">'
                                      f'{highlight_citations(acc)}▌</div></div>', unsafe_allow_html=True)
                    st.rerun()

    with tab_prec:
        floor = desk.thresholds["min_similarity"]
        H(f'<div class="sd-legend">Cosine similarity to this ticket (EmbeddingGemma). The tick on each meter marks '
          f'the gate\'s precedent floor ({floor:.2f}); below it the ticket counts as having no precedent.</div>')
        cards = []
        for prec in r.precedents:
            used = r.draft is not None and prec.ticket_id in r.draft.cited
            tags = "".join(f'<span class="sd-tag">{esc(tg)}</span> ' for tg in prec.tags[:4])
            more = (f'<details><summary>Full resolution</summary>{multiline(prec.answer)}</details>'
                    if len(prec.answer) > 360 else "")
            cards.append(
                f'<div class="sd-prec{" cited" if used else ""}"><div class="sd-prec-top">'
                f'<span class="sd-prec-id">{prec.ticket_id}</span>'
                + ('<span class="sd-pill sd-pill-accent">cited</span>' if used else "")
                + f'<span>{LANG_FLAG.get(prec.language, prec.language)}</span>'
                f'<span class="sd-prec-sim">{prec.similarity:.2f}</span></div>'
                + meter(prec.similarity, tick=floor, title=f"Similarity {prec.similarity:.3f}")
                + f'<div class="sd-prec-meta">{esc(prec.type)} · {esc(prec.queue)} · {esc(prec.priority)} priority '
                f'&nbsp;{tags}</div>'
                f'<div class="sd-prec-subj">{esc(prec.subject) or "<i>(no subject)</i>"}</div>'
                f'<div class="sd-prec-body">{esc(prec.body[:220])}{"…" if len(prec.body) > 220 else ""}</div>'
                f'<div class="sd-prec-res"><b>Resolution:</b> {esc(prec.answer[:360])}'
                f'{"…" if len(prec.answer) > 360 else ""}</div>{more}</div>')
        H('<div class="sd-precs">' + "".join(cards) + '</div>')

    with tab_probs:
        H('<div class="sd-legend">Calibrated class probabilities per head, ranked. The predicted class is '
          'highlighted.</div>')
        cards = []
        for head in ("type", "queue", "priority", "critical"):
            rows = []
            for i, (cls, pr) in enumerate(p[head]["probs"].items()):
                rows.append(f'<div class="sd-prob-row" title="{esc(cls)}: {pr:.1%}"><div class="sd-prob-line">'
                            f'<span>{esc(cls)}</span><b>{pr:.0%}</b></div>'
                            f'<div class="sd-bar{" top" if i == 0 else ""}"><i style="width:{pr * 100:.1f}%"></i></div></div>')
            cards.append(f'<div class="sd-prob-card"><div class="sd-prob-title">{HEAD_TITLE[head]}</div>'
                         + "".join(rows) + '</div>')
        H('<div class="sd-probs">' + "".join(cards) + '</div>')

    with tab_json:
        st.code(json.dumps(r.to_dict(), indent=2, ensure_ascii=False, default=str), language="json")

# ------------------------------------------------------------------------------- history
if ss.history:
    H('<div class="sd-section" style="margin-top:22px">Session inbox</div>')
    thr = desk.thresholds["critical_risk"]
    st_kind = {gate.AUTO_DRAFT: ("good", "Auto-draft"), gate.HUMAN_REVIEW: ("warn", "Review"),
               gate.ESCALATE_ONCALL: ("crit", "On-call")}

    def cell(pred: dict) -> str:
        q = '<span class="sd-q" title="under 60% confidence">?</span>' if pred["confidence"] < UNCERTAIN_BELOW else ""
        return f"{esc(pred['label'])}{q}"

    rows = []
    for h in ss.history[:25]:
        k, lbl = st_kind[h.gate.status]
        hot = h.critical_risk >= thr
        rows.append(
            f'<tr><td class="ref">{esc(h.ref)}</td>'
            f'<td><span class="sd-status"><span class="sd-dot {k}"></span>{lbl}</span></td>'
            f'<td>{cell(h.predictions["type"])}</td><td>{cell(h.predictions["queue"])}</td>'
            f'<td>{cell(h.predictions["priority"])}</td>'
            f'<td><div class="sd-risk">{meter(h.critical_risk, "var(--sd-crit)" if hot else "var(--sd-accent)", tick=thr)}'
            f'<span>{h.critical_risk:.0%}</span></div></td>'
            f'<td>{LANG_FLAG.get(h.language, esc(h.language))}</td>'
            f'<td class="subj">{esc((h.subject or h.body)[:90])}</td>'
            f'<td>{"🚨" if h.burst.get("active") else ""}</td></tr>')
    H('<div class="sd-table-wrap"><table class="sd-table"><thead><tr><th>Ref</th><th>Status</th><th>Type</th>'
      '<th>Queue</th><th>Priority</th><th>Critical risk</th><th>Lang</th><th>Subject</th><th>Burst</th></tr></thead>'
      '<tbody>' + "".join(rows) + '</tbody></table></div>')
