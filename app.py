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

st.set_page_config(page_title="SentinelDesk", page_icon="🛡️", layout="wide")

CSS = """
<style>
.block-container {padding-top: 1.6rem; max-width: 1400px;}
.sd-badge {border-radius: 12px; padding: 14px 18px; margin: 4px 0 10px 0; color: #fff;}
.sd-badge h3 {margin: 0; font-size: 1.25rem; color: #fff;}
.sd-badge p {margin: 4px 0 0 0; opacity: .92; font-size: .9rem;}
.sd-auto {background: linear-gradient(90deg,#0f7b4a,#16a34a);}
.sd-review {background: linear-gradient(90deg,#b45309,#d97706);}
.sd-oncall {background: linear-gradient(90deg,#991b1b,#dc2626);}
.sd-outage {background:#7f1d1d; color:#fff; border-radius:10px; padding:12px 16px; margin-bottom:12px;
            border:2px solid #fca5a5; animation: pulse 1.6s infinite;}
@keyframes pulse {0%{box-shadow:0 0 0 0 rgba(220,38,38,.6);}70%{box-shadow:0 0 0 12px rgba(220,38,38,0);}100%{box-shadow:0 0 0 0 rgba(220,38,38,0);}}
.sd-reason {font-size:.88rem; margin:2px 0;}
.sd-code {font-family: ui-monospace, monospace; font-size:.78rem; padding:1px 6px; border-radius:6px;
          background: rgba(127,127,127,.18); margin-right:6px;}
.sd-cite {font-family: ui-monospace, monospace; font-size:.8rem; background:#1d4ed8; color:#fff;
          padding:1px 6px; border-radius:6px;}
.sd-reply {border:1px solid rgba(127,127,127,.35); border-radius:10px; padding:14px 16px;
           white-space: pre-wrap; line-height:1.5;}
.sd-translation {border-style: dashed; opacity: .92;}
.sd-prec {border:1px solid rgba(127,127,127,.3); border-radius:10px; padding:10px 14px; margin-bottom:10px;}
.sd-prec .meta {font-size:.8rem; opacity:.75;}
.sd-bar {height:6px; border-radius:3px; background:rgba(127,127,127,.2); margin:4px 0 6px 0;}
.sd-bar > div {height:6px; border-radius:3px; background:#2563eb;}
.sd-small {font-size:.8rem; opacity:.75;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

STATUS_UI = {
    gate.AUTO_DRAFT: ("sd-auto", "✅ Auto-Draft Ready",
                      "Confident triage, strong precedents, draft passed all grounding checks."),
    gate.HUMAN_REVIEW: ("sd-review", "🟠 Human Review Required",
                        "An agent must review before anything is sent."),
    gate.ESCALATE_ONCALL: ("sd-oncall", "🚨 Escalate to On-Call Engineer",
                           "Critical incident - page on-call now. Draft is for the engineer only."),
}
LANG_FLAG = {"en": "🇬🇧 EN", "de": "🇩🇪 DE"}


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

# ------------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("## 🛡️ SentinelDesk")
    st.caption("Multilingual incident triage · escalation gate · Gemma-grounded replies")
    llm = desk.llm
    if llm:
        st.success(f"Gemma backend: `{llm.describe()}`")
    else:
        st.error("No Gemma backend - set GEMINI_API_KEY, or install mlx-lm / Ollama.")
    st.caption(f"Embeddings: `{config.EMBED_MODEL}` · KB: {len(desk.retriever.kb):,} resolved tickets")

    st.markdown("### Demo scenarios")
    pick = st.selectbox("Load a sample ticket", ["—"] + list(DEMO_TICKETS))
    if st.button("Load sample", width="stretch", disabled=pick == "—"):
        ss.subject, ss.body = DEMO_TICKETS[pick]
        st.rerun()

    st.markdown("### Outage burst simulation")
    if st.button("Simulate 4 VPN failure tickets", width="stretch"):
        now = time.time()
        for i, (s, b) in enumerate(OUTAGE_BURST):
            r = desk.triage(s, b, generate=False, now=now + i)
            ss.history.insert(0, r)
        st.rerun()
    if st.button("Reset burst window", width="stretch"):
        desk.burst.reset()
        st.rerun()

    st.markdown("### Gate thresholds")
    t = desk.thresholds
    t["min_confidence"] = st.slider("Min routing confidence", 0.0, 1.0,
                                    float(t["min_confidence"]), 0.01)
    t["min_similarity"] = st.slider("Min precedent similarity", 0.0, 1.0,
                                    float(t["min_similarity"]), 0.01)
    t["critical_risk"] = st.slider("Critical-incident risk", 0.0, 1.0,
                                   float(t["critical_risk"]), 0.01)
    st.caption("Defaults were tuned on a validation slice of the KB (never the holdout).")
    gen_on = st.toggle("Generate Gemma draft", value=True)
    translate_on = st.toggle("English translation of German drafts", value=True,
                             help="Operator-only translation; the customer still gets the German reply.")

# ------------------------------------------------------------------------------- header
st.markdown("# Live Triage Console")
burst_now = next((r.burst for r in ss.history[:1] if r.burst.get("active")), None)
if burst_now:
    members = "".join(f"<li>{esc(m['subject'])}</li>" for m in burst_now["members"][-4:])
    st.markdown(
        f'<div class="sd-outage"><b>🚨 MAJOR INCIDENT - possible outage.</b> '
        f'{burst_now["size"]} similar tickets in the last {burst_now["window_min"]:.0f} min '
        f'(queue: {esc(burst_now["queue"])}). On-call paged.<ul style="margin:6px 0 0 0">{members}</ul></div>',
        unsafe_allow_html=True)

# ------------------------------------------------------------------------------- input
with st.form("ticket"):
    c1, c2 = st.columns([1, 3])
    with c1:
        st.text_input("Subject (optional)", key="subject")
    with c2:
        st.text_area("Ticket body", key="body", height=140,
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
    st.info("Load a demo scenario from the sidebar or paste a ticket, then press **Triage ticket**.")
else:
    cls, title, sub = STATUS_UI[r.gate.status]
    if ss.pending_draft:
        sub = "Provisional - Gemma is drafting; grounding checks run when it finishes."
    st.markdown(f'<div class="sd-badge {cls}"><h3>{title}</h3><p>{sub}</p></div>',
                unsafe_allow_html=True)
    if r.gate.reasons:
        for reason in r.gate.reasons:
            icon = "🔴" if reason.severity == gate.CRITICAL else "🟠"
            st.markdown(f'<div class="sd-reason">{icon} <span class="sd-code">{reason.code}</span>'
                        f'{esc(reason.message)}</div>', unsafe_allow_html=True)

    p = r.predictions
    m = st.columns(6)
    m[0].metric("Type", p["type"]["label"], f"{p['type']['confidence']:.0%} conf", delta_color="off")
    m[1].metric("Queue", p["queue"]["label"], f"{p['queue']['confidence']:.0%} conf", delta_color="off")
    m[2].metric("Priority", p["priority"]["label"].upper(), f"{p['priority']['confidence']:.0%} conf",
                delta_color="off")
    m[3].metric("Critical risk", f"{r.critical_risk:.0%}",
                f"threshold {desk.thresholds['critical_risk']:.0%}", delta_color="off")
    m[4].metric("Language", LANG_FLAG.get(r.language, r.language.upper()),
                f"reply in {r.reply_language.upper()}", delta_color="off")
    m[5].metric("Latency", f"{r.latency_ms['total'] / 1000:.2f}s",
                f"triage {r.latency_ms['triage']:.0f}ms",
                delta_color="off")
    lat = " · ".join(f"{k} {v:.0f}ms" for k, v in r.latency_ms.items()
                     if k not in ("total", "triage", "translate"))
    st.caption(f"Routing confidence {r.routing_confidence:.0%} · {lat} · ref {r.ref}"
               + (f" · 🔒 {r.pii['masked']} PII item(s) masked before LLM" if r.pii["masked"] else ""))

    tab_draft, tab_prec, tab_probs, tab_json = st.tabs(
        ["✍️ Draft reply", f"📚 Precedents ({len(r.precedents)})", "📊 Probabilities", "{ } JSON"])

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
            st.warning(r.llm_error or "Draft generation disabled.")
        else:
            if r.gate.status == gate.ESCALATE_ONCALL:
                st.caption("⚠️ Internal draft for the on-call engineer - not sent automatically.")
            elif r.gate.status == gate.HUMAN_REVIEW:
                st.caption("⚠️ Draft requires agent approval before sending.")
            st.markdown(f'<div class="sd-reply">{highlight_citations(d.reply)}</div>',
                        unsafe_allow_html=True)
            if translate_on and desk.needs_translation(r):
                st.markdown("**🇬🇧 English translation** "
                            "<span class='sd-small'>· for operators only, not sent to the customer</span>",
                            unsafe_allow_html=True)
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
            ch = st.columns(4)
            ch[0].markdown(f"**Citations** {'✅' if d.cited else '❌'} "
                           f"{', '.join(d.cited) or 'none'}")
            ch[1].markdown(f"**Unsupported numbers** {'❌ ' + ', '.join(d.unsupported_claims) if d.unsupported_claims else '✅ none'}")
            ch[2].markdown(f"**Self-reported grounding** `{d.grounding}`")
            ch[3].markdown(f"**Model** `{d.backend}:{d.model}` · {d.latency_ms / 1000:.1f}s")
            if d.invalid_citations:
                st.error(f"Removed hallucinated citations: {', '.join(d.invalid_citations)}")
            if d.grounding_note:
                st.caption(f"Grounding note: {d.grounding_note}")

    with tab_prec:
        for prec in r.precedents:
            used = r.draft is not None and prec.ticket_id in r.draft.cited
            st.markdown(
                f'<div class="sd-prec"><b>{prec.ticket_id}</b> {"<span class=sd-cite>cited</span>" if used else ""}'
                f' &nbsp;<span class="meta">{LANG_FLAG.get(prec.language, prec.language)} · {esc(prec.type)} · '
                f'{esc(prec.queue)} · {esc(prec.priority)} · similarity {prec.similarity:.3f}</span>'
                f'<div class="sd-bar"><div style="width:{max(0, prec.similarity) * 100:.0f}%"></div></div>'
                f'<b>{esc(prec.subject) or "<i>(no subject)</i>"}</b><br>'
                f'<span class="sd-small">{esc(prec.body[:260])}{"…" if len(prec.body) > 260 else ""}</span><br>'
                f'<b>Resolution:</b> {esc(prec.answer[:420])}{"…" if len(prec.answer) > 420 else ""}</div>',
                unsafe_allow_html=True)

    with tab_probs:
        cols = st.columns(4)
        for col, head in zip(cols, ("type", "queue", "priority", "critical")):
            with col:
                st.markdown(f"**{head}**")
                probs = p[head]["probs"]
                st.bar_chart(pd.DataFrame({"probability": list(probs.values())},
                                          index=list(probs.keys())), horizontal=True, height=220)

    with tab_json:
        st.code(json.dumps(r.to_dict(), indent=2, ensure_ascii=False, default=str), language="json")

# ------------------------------------------------------------------------------- history
if ss.history:
    st.markdown("### Session inbox")
    rows = [{
        "ref": h.ref, "status": h.gate.status, "type": h.predictions["type"]["label"],
        "queue": h.predictions["queue"]["label"], "priority": h.predictions["priority"]["label"],
        "risk": f"{h.critical_risk:.0%}", "lang": h.language,
        "subject": (h.subject or h.body)[:70], "burst": "🚨" if h.burst.get("active") else "",
    } for h in ss.history[:25]]
    st.dataframe(rows, width="stretch", hide_index=True)
