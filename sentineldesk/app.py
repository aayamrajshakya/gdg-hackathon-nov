"""SentinelDesk live triage console.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import html
import json
from dataclasses import replace

import pandas as pd
import streamlit as st

from sentinel import config
from sentinel.burst import BurstDetector
from sentinel.examples import BURST_TICKETS, EXAMPLES
from sentinel.gate import AUTO, ON_CALL, REVIEW
from sentinel.pipeline import RunSettings, SentinelDesk
from sentinel.textutils import build_text

st.set_page_config(page_title="SentinelDesk", layout="wide")

st.markdown(
    """
<style>
.sd-status {border: 1px solid; border-left-width: 8px; border-radius: 0; padding: 14px 18px; margin: 0 0 16px;}
.sd-status-title {font-size: 1.25rem; font-weight: 600; margin-bottom: 4px;}
.sd-status ul {margin: 4px 0 0 18px; padding: 0;}
.sd-oncall {background: #FEF3F2; border-color: #B42318; color: #7A271A;}
.sd-review {background: #FFFAEB; border-color: #B54708; color: #7A2E0E;}
.sd-auto {background: #ECFDF3; border-color: #067647; color: #054F31;}
.sd-banner {background: #7A271A; color: #FFFFFF; padding: 12px 18px; margin-bottom: 14px;}
.sd-field {padding: 6px 0;}
.sd-field-name {font-size: 0.85rem; color: #4A5568;}
.sd-field-value {font-size: 1.15rem; font-weight: 600; line-height: 1.3;}
.sd-field-conf {font-size: 0.85rem; color: #4A5568;}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading triage models and the precedent index…")
def load_desk() -> SentinelDesk:
    return SentinelDesk()


@st.cache_resource
def load_monitor(backend: str) -> BurstDetector:
    return BurstDetector(sim_threshold=config.BURST_SIM[backend])


try:
    desk = load_desk()
except FileNotFoundError as exc:
    st.error(f"{exc}")
    st.code("python train.py\nstreamlit run app.py", language="bash")
    st.stop()

monitor = load_monitor(desk.retriever.backend)
defaults = desk.default_settings()
for key, value in {"subject": "", "body": "", "result": None}.items():
    st.session_state.setdefault(key, value)


def load_example() -> None:
    example = EXAMPLES.get(st.session_state.example_choice)
    if example:
        st.session_state.subject = example["subject"]
        st.session_state.body = example["body"]


def triage_and_log(subject: str, body: str, settings: RunSettings) -> dict:
    result = desk.run(subject, body, settings)
    qvec = result.pop("query_vec")
    monitor.add(qvec, {
        "subject": subject or body[:70],
        "text": build_text(subject, body),
        "type": result["triage"]["type"]["label"],
        "priority": result["triage"]["priority"]["label"],
        "status": result["decision"]["label"],
    })
    return result


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Escalation gate")
    min_conf = st.slider("Minimum triage confidence", 0.0, 1.0, float(defaults.min_confidence), 0.01)
    sim_thr = st.slider("Minimum precedent similarity", 0.0, 1.0, float(defaults.sim_threshold), 0.01)
    use_detector = st.toggle("Critical-incident detector", True)
    crit_thr = st.slider("Detector threshold", 0.0, 1.0, float(round(defaults.critical_threshold, 2)), 0.01,
                         help="Tuned on validation data to catch 90% of high-priority incidents.")
    use_keywords = st.toggle("Outage keyword safety net", True)

    st.header("Retrieval and drafting")
    top_k = st.slider("Precedents to retrieve", 3, 5, config.TOP_K)
    mask = st.toggle("Mask PII before the LLM call", True)
    llm_available = desk.generator.provider != "template"
    use_llm = st.toggle("Draft with the LLM", llm_available, disabled=not llm_available)
    st.caption(f"Drafting engine: {desk.generator.describe()}")
    st.caption(f"Retrieval: {desk.retriever.backend}, {len(desk.retriever.meta):,} historical tickets")
    if desk.meta:
        st.caption(f"Models trained {desk.meta.get('trained_at', 'unknown')}")

settings = RunSettings(
    min_confidence=min_conf, sim_threshold=sim_thr, critical_threshold=crit_thr,
    use_critical_detector=use_detector, use_outage_keywords=use_keywords,
    top_k=top_k, mask_pii=mask, use_llm=use_llm,
)

# ---------------------------------------------------------------- header + outage banner
st.title("SentinelDesk")
st.caption("Triage incoming tickets, escalate what matters, and draft replies grounded in past resolutions.")

alert = monitor.current_alert()
if alert:
    st.markdown(
        f'<div class="sd-banner"><strong>Major incident suspected:</strong> {alert["size"]} similar tickets in '
        f'the last {config.BURST_WINDOW_MIN} minutes. Common terms: '
        f'{html.escape(", ".join(alert["keywords"]))}. Open the outage monitor for details.</div>',
        unsafe_allow_html=True,
    )

console_tab, outage_tab, metrics_tab = st.tabs(["Triage console", "Outage monitor", "Holdout metrics"])


def field(name: str, value: str, conf: float | None = None) -> str:
    conf_html = f'<div class="sd-field-conf">{conf:.0%} confidence</div>' if conf is not None else ""
    return (f'<div class="sd-field"><div class="sd-field-name">{html.escape(name)}</div>'
            f'<div class="sd-field-value">{html.escape(value)}</div>{conf_html}</div>')


def render_result(res: dict) -> None:
    decision, tri, draft = res["decision"], res["triage"], res["draft"]
    css = {ON_CALL: "sd-oncall", REVIEW: "sd-review", AUTO: "sd-auto"}[decision["status"]]
    reasons = "".join(f"<li>{html.escape(r)}</li>" for r in decision["reasons"])
    st.markdown(
        f'<div class="sd-status {css}"><div class="sd-status-title">{html.escape(decision["label"])}</div>'
        f"<ul>{reasons}</ul></div>",
        unsafe_allow_html=True,
    )

    cols = st.columns(4)
    for col, key, name in zip(cols, ("type", "queue", "priority"), ("Type", "Queue", "Priority")):
        col.markdown(field(name, tri[key]["label"], tri[key]["confidence"]), unsafe_allow_html=True)
    cols[3].markdown(field("End-to-end latency", f"{res['latency_ms']['total']:.0f} ms"), unsafe_allow_html=True)

    pii = ", ".join(f"{v} {k.lower()}" for k, v in res["pii_masked"].items()) or "none found"
    st.caption(
        f"Language detected: {res['language'].upper()}  |  overall confidence {tri['confidence']:.2f}  |  "
        f"critical score {tri['critical_score']:.2f}  |  PII masked: {pii}"
    )

    st.subheader("Draft reply")
    with st.container(border=True):
        st.markdown(draft["reply"].replace("\n", "  \n"))
    cited = ", ".join(draft["cited_ticket_ids"]) or "none"
    st.caption(f"Written by: {draft['engine']}  |  cites: {cited}")
    if draft.get("note"):
        st.warning(draft["note"])
    if draft["issues"]:
        for issue in draft["issues"]:
            st.warning(f"Grounding check: {issue}")
    else:
        st.success("Grounding checks passed: every citation is a retrieved ticket and no unsupported figures appear.")

    st.subheader("Historical precedents")
    for h in res["hits"]:
        mark = "  |  cited" if h["ticket_id"] in draft["cited_ticket_ids"] else ""
        title = (f"{h['ticket_id']}  |  similarity {h['similarity']:.2f}  |  {h['queue']}  |  "
                 f"{h['language'].upper()}{mark}")
        with st.expander(title):
            st.markdown(f"**Customer issue:** {h['subject']} {h['body'][:400]}")
            st.markdown(f"**Agent resolution:** {h['answer'][:700]}")
            st.caption(f"type {h['type']}  |  priority {h['priority']}")

    lat = res["latency_ms"]
    st.caption("Latency by stage: " + "  |  ".join(f"{k} {v:.0f} ms" for k, v in lat.items() if k != "total"))
    with st.expander("Raw pipeline output"):
        st.json(res, expanded=False)


# ---------------------------------------------------------------- triage console
with console_tab:
    left, right = st.columns([5, 7], gap="large")
    with left:
        st.selectbox("Load an example ticket", ["Choose an example…", *EXAMPLES],
                     key="example_choice", on_change=load_example)
        st.text_input("Subject (optional)", key="subject")
        st.text_area("Ticket body", key="body", height=260, placeholder="Paste the customer's message here.")
        if st.button("Triage ticket", type="primary"):
            if not (st.session_state.subject.strip() or st.session_state.body.strip()):
                st.warning("Add a ticket body (the subject is optional), then select Triage ticket again.")
            else:
                with st.spinner("Classifying, retrieving precedents and drafting…"):
                    st.session_state.result = triage_and_log(
                        st.session_state.subject, st.session_state.body, settings
                    )
                st.rerun()
    with right:
        if st.session_state.result is None:
            st.info("Paste a ticket or load an example, then select Triage ticket.")
        else:
            render_result(st.session_state.result)

# ---------------------------------------------------------------- outage monitor
with outage_tab:
    st.write(
        f"Each triaged ticket is compared with tickets from the last {config.BURST_WINDOW_MIN} minutes. "
        f"When {config.BURST_MIN_TICKETS} or more describe the same failure, a major-incident banner appears."
    )
    c1, c2, _ = st.columns([2, 2, 6])
    if c1.button("Simulate an outage burst"):
        quick = replace(settings, use_llm=False)
        with st.spinner("Triaging a burst of VPN failure tickets…"):
            for ticket in BURST_TICKETS:
                triage_and_log(ticket["subject"], ticket["body"], quick)
        st.rerun()
    if c2.button("Clear monitor"):
        monitor.clear()
        st.rerun()
    if alert:
        st.error(f"{alert['size']} related tickets. Recent subjects: " + "; ".join(alert["subjects"]))
    log = monitor.log()
    if log:
        st.dataframe(pd.DataFrame(log), hide_index=True)
    else:
        st.info("No tickets triaged yet. Triage a few tickets or simulate an outage burst.")

# ---------------------------------------------------------------- holdout metrics
with metrics_tab:
    report_path = config.ARTIFACT_DIR / "eval_report.json"
    if not report_path.exists():
        st.info("No evaluation report yet. Run the command below, then reload this page.")
        st.code("python evaluate.py", language="bash")
    else:
        rep = json.loads(report_path.read_text())
        cols = st.columns(4)
        for col, t in zip(cols, config.TARGETS):
            col.metric(f"{t.title()} macro-F1", f"{rep['triage'][t]['macro_f1']:.3f}")
        hpi = rep["high_priority_incident"]
        cols[3].metric("High-priority incident recall", f"{hpi['recall_on_call_gate']:.1%}",
                       help="Share of true high-priority incidents the on-call gate escalates.")
        st.caption(
            f"Holdout tickets: {rep['n_test']:,}  |  recall from predicted labels alone "
            f"{hpi['recall_predicted_labels']:.1%}  |  gate precision {hpi['precision_on_call_gate']:.1%}  |  "
            f"p50 latency {rep['end_to_end']['latency_p50_ms']} ms, p95 {rep['end_to_end']['latency_p95_ms']} ms"
        )
        st.subheader("Confidence threshold trade-off")
        st.dataframe(pd.DataFrame(rep["confidence_coverage"]), hide_index=True)
        with st.expander("Full report"):
            st.json(rep, expanded=False)
