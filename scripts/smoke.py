"""Run the demo scenarios end to end and print a compact summary (sanity check before a demo)."""

import argparse
import time

import _bootstrap  # noqa: F401

from sentineldesk.pipeline import SentinelDesk
from sentineldesk.samples import DEMO_TICKETS, OUTAGE_BURST

ap = argparse.ArgumentParser()
ap.add_argument("--gen", action="store_true", help="also generate Gemma drafts")
ap.add_argument("--only", default="", help="substring filter on scenario name")
args = ap.parse_args()

desk = SentinelDesk(load_llm=args.gen)
print("thresholds:", desk.thresholds, "| llm:", desk.llm.describe() if desk.llm else None)
for name, (s, b) in DEMO_TICKETS.items():
    if args.only and args.only.lower() not in name.lower():
        continue
    r = desk.triage(s, b, generate=args.gen, translate=args.gen)
    p = r.predictions
    print(f"\n### {name}\n  {r.gate.status}  lang={r.language}->{r.reply_language}  "
          f"type={p['type']['label']}({p['type']['confidence']:.2f}) queue={p['queue']['label']}"
          f"({p['queue']['confidence']:.2f}) prio={p['priority']['label']}({p['priority']['confidence']:.2f}) "
          f"risk={r.critical_risk:.2f} route={r.routing_confidence:.2f} top_sim={r.precedents[0].similarity:.3f}")
    print("  precedents:", [(x.ticket_id, round(x.similarity, 3), x.language) for x in r.precedents])
    for reason in r.gate.reasons:
        print(f"  - {reason.code}: {reason.message}")
    if r.draft:
        print(f"  draft ({r.draft.backend}:{r.draft.model}, {r.draft.latency_ms / 1000:.1f}s, "
              f"grounding={r.draft.grounding}, cited={r.draft.cited}):")
        print("   ", r.draft.reply.replace("\n", "\n    "))
        if r.draft.translation_en:
            print("  english translation (operator only):")
            print("   ", r.draft.translation_en.replace("\n", "\n    "))
    print("  latency:", r.latency_ms)

desk.burst.reset()
now = time.time()
print("\n### burst simulation")
for i, (s, b) in enumerate(OUTAGE_BURST):
    r = desk.triage(s, b, generate=False, now=now + i)
    print(f"  {i + 1}: size={r.burst['size']} active={r.burst['active']} status={r.gate.status} "
          f"type={r.predictions['type']['label']}")
