"""Smoke test on synthetic tickets: no download, no LLM, TF-IDF backend.

Run:  python tests/test_smoke.py     (or: pytest tests/)
"""
from __future__ import annotations

import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TMP = Path(tempfile.mkdtemp(prefix="sentinel_smoke_"))
os.environ["SENTINEL_DATA_DIR"] = str(TMP / "data")
os.environ["SENTINEL_ARTIFACT_DIR"] = str(TMP / "artifacts")
os.environ["LLM_PROVIDER"] = "template"
sys.path.insert(0, str(ROOT))

SCENARIOS = [
    ("Incident", "Technical Support", "high",
     "VPN outage", "The VPN is down for all users since this morning, TLS handshake failed.",
     "We restarted the VPN gateway and renewed the TLS certificate. Please reconnect the client.",
     "VPN Ausfall", "Das VPN ist seit heute Morgen für alle Nutzer ausgefallen, TLS Fehler.",
     "Wir haben das VPN-Gateway neu gestartet und das Zertifikat erneuert. Bitte verbinden Sie sich neu."),
    ("Request", "Customer Service", "low",
     "Password reset", "I forgot my password and the reset email does not arrive.",
     "Please use the self-service reset link and check your spam folder. We re-sent the email.",
     "Passwort zurücksetzen", "Ich habe mein Passwort vergessen und die E-Mail kommt nicht an.",
     "Bitte nutzen Sie den Link zum Zurücksetzen und prüfen Sie den Spam-Ordner."),
    ("Problem", "Billing and Payments", "medium",
     "Duplicate charge", "My invoice shows the subscription charge twice this month.",
     "We reviewed the billing record and reversed the duplicate payment within 5 business days.",
     "Doppelte Abbuchung", "Auf meiner Rechnung wurde das Abo diesen Monat zweimal abgebucht.",
     "Wir haben die doppelte Zahlung storniert."),
    ("Change", "IT Support", "medium",
     "Printer driver update", "Please update the printer driver, after the Windows update it prints blank pages.",
     "We rolled back the driver and installed the vendor's latest version.",
     "Druckertreiber", "Nach dem Windows-Update druckt der Drucker nur leere Seiten.",
     "Wir haben den Treiber zurückgesetzt und neu installiert."),
    ("Request", "Sales and Pre-Sales", "low",
     "Pricing question", "What does the enterprise license cost for 200 users?",
     "Our sales team will send a tailored quote after a short call.",
     "Preisanfrage", "Was kostet die Enterprise-Lizenz für 200 Nutzer?",
     "Unser Vertrieb sendet Ihnen ein individuelles Angebot."),
    ("Incident", "Product Support", "high",
     "Data loss after sync", "After the last sync we lost data in all project folders, urgent.",
     "We restored the folders from the nightly backup and paused the sync job.",
     "Datenverlust", "Nach der Synchronisierung sind Daten in allen Projektordnern verloren.",
     "Wir haben die Ordner aus dem nächtlichen Backup wiederhergestellt."),
]
FILLERS = {"en": ["Thanks in advance.", "Please help.", "This affects our team.", "", "We use the latest version."],
           "de": ["Vielen Dank im Voraus.", "Bitte helfen Sie uns.", "Das betrifft unser Team.", "", "Wir nutzen die neueste Version."]}


def make_toy_csv(path: Path, n: int = 1500, seed: int = 0) -> None:
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        sc = rng.choice(SCENARIOS)
        lang = rng.choice(["en", "de"])
        subj, body, ans = (sc[3], sc[4], sc[5]) if lang == "en" else (sc[6], sc[7], sc[8])
        rows.append({
            "subject": "" if rng.random() < 0.1 else subj,  # some missing subjects, like the real data
            "body": f"{body} {rng.choice(FILLERS[lang])} Ref {i}.",
            "answer": ans, "type": sc[0], "queue": sc[1], "priority": sc[2], "language": lang,
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def run(*args: str) -> None:
    subprocess.run([sys.executable, *args], cwd=ROOT, env=os.environ.copy(), check=True)


def test_guardrails() -> None:
    from sentinel.generator import verify
    from sentinel.textutils import build_text, detect_language, mask_pii

    assert detect_language("Hallo, seit dem Update funktioniert der Drucker nicht mehr.") == "de"
    assert detect_language("Hello, since the update the printer is not working.") == "en"
    assert build_text(None, "body only") == "body only"
    assert build_text(float("nan"), "  x ") == "x"

    masked, found = mask_pii(
        "Mail jane@example.com, call +49 170 1234567, IP 10.0.0.12, card 4111 1111 1111 1111, "
        "IBAN DE89 3704 0044 0532 0130 00, customer no. AC-778812, on 2024-01-15 for customer service."
    )
    for tag in ("EMAIL", "PHONE", "IP", "CARD", "IBAN", "ACCOUNT"):
        assert tag in found, (tag, masked)
    assert "2024-01-15" in masked and "customer service" in masked

    hits = [{"ticket_id": "TKT-00001", "answer": "Refund within 5 business days.", "body": ""}]
    reply, cited, issues = verify("We refund in 3 business days [TKT-00001] [TKT-99999].", [], hits, True)
    assert cited == ["TKT-00001"] and "TKT-99999" not in reply
    assert any("not retrieved" in i for i in issues) and any("figures" in i for i in issues)


def test_end_to_end() -> None:
    csv = TMP / "toy.csv"
    make_toy_csv(csv)
    run("train.py", "--data", str(csv), "--backend", "tfidf")
    run("evaluate.py", "--pipeline-sample", "40")

    from sentinel.examples import EXAMPLES
    from sentinel.gate import AUTO, ON_CALL
    from sentinel.pipeline import SentinelDesk

    desk = SentinelDesk(TMP / "artifacts")
    results = {name: desk.run(ex["subject"], ex["body"]) for name, ex in EXAMPLES.items()}
    for name, r in results.items():
        print(f"  {r['decision']['status']:<18} {r['latency_ms']['total']:>7.1f} ms  {name}")

    assert results["Hospital VPN outage (EN, should page on-call)"]["decision"]["status"] == ON_CALL
    assert results["Password reset (EN, routine)"]["decision"]["status"] == AUTO
    assert results["Prompt-injection attempt (EN)"]["decision"]["status"] != AUTO
    assert results["Out of scope (EN)"]["decision"]["status"] != AUTO
    assert results["Drucker druckt leere Seiten (DE)"]["language"] == "de"
    pii = results["Refund and SLA credit demand with PII (EN)"]["pii_masked"]
    assert {"EMAIL", "PHONE", "IBAN", "ACCOUNT"} <= set(pii), pii
    assert "answer" not in pd.read_csv(TMP / "data" / "test_inputs.csv").columns


if __name__ == "__main__":
    test_guardrails()
    print("guardrail checks passed")
    test_end_to_end()
    print(f"end-to-end smoke test passed (artifacts in {TMP})")
