from sentineldesk.generator import build_prompt, parse, verify
from sentineldesk.retriever import Precedent


def prec(tid, answer, lang="en"):
    return Precedent(tid, 0.8, "VPN issue", "VPN drops", answer, "Incident", "IT Support", "high", lang, [])


PRECS = [prec("TKT-00001", "Please restart the VPN client and re-enter your credentials."),
         prec("TKT-00002", "We reset the authentication server; expect recovery within 2 hours.", "de")]


def test_parse_format():
    reply, g, note = parse("REPLY:\nHello there [TKT-00001].\nGROUNDING: PARTIAL - pricing not covered")
    assert reply == "Hello there [TKT-00001]." and g == "PARTIAL" and "pricing" in note


def test_parse_markdown_bold_headers():
    reply, g, _ = parse("**REPLY:** Hi [TKT-00001]\n**GROUNDING:** FULL - all covered")
    assert reply.startswith("Hi") and g == "FULL"


def test_parse_without_reply_header():
    raw = "Hello,\n\nRestart the client [TKT-00001].\n\nBest regards\n\nGROUNDING: FULL - all from resolutions."
    reply, g, note = parse(raw)
    assert g == "FULL" and "GROUNDING" not in reply and reply.startswith("Hello,")


def test_parse_missing_grounding_is_unknown():
    reply, g, _ = parse("REPLY: Hi there")
    assert reply == "Hi there" and g == "UNKNOWN"


def test_verify_strips_hallucinated_citation_and_flags_numbers():
    reply = ("Please restart the VPN client [TKT-00001]. Recovery within 2 hours [TKT-00002]. "
             "You will get a refund of $500 within 24 hours [TKT-99999].")
    out, cited, invalid, unsupported, reasons = verify(reply, PRECS, "vpn broken", "en")
    assert cited == ["TKT-00001", "TKT-00002"] and invalid == ["TKT-99999"]
    assert "TKT-99999" not in out
    assert any("500" in u for u in unsupported) and any("24" in u for u in unsupported)
    assert not any("2 hours" == u for u in unsupported)
    assert {"HALLUCINATED_CITATION", "UNSUPPORTED_CLAIM"} <= {r.code for r in reasons}


def test_verify_language_mismatch_and_no_citation():
    reply = "Bitte starten Sie den VPN-Client neu und melden Sie sich erneut an. Vielen Dank für Ihre Geduld."
    _, cited, _, _, reasons = verify(reply, PRECS, "vpn", "en")
    codes = {r.code for r in reasons}
    assert not cited and {"NO_CITATION", "LANGUAGE_MISMATCH"} <= codes


def test_prompt_contains_only_retrieved_ids_and_language():
    p = build_prompt(subject="", body="VPN kaputt", reply_lang="de", precedents=PRECS)
    assert "TKT-00001" in p and "TKT-00002" in p and "German" in p and "(no subject)" in p


def test_display_text_hides_protocol_while_streaming():
    from sentineldesk.generator import display_text

    assert display_text("REPLY:\nHello") == "Hello"
    assert display_text("**REPLY:** Hello there\n\nBest\nGROUNDING: FULL - x") == "Hello there\n\nBest"
    assert display_text("Hello there\nGROU") == "Hello there"  # partial header held back


def test_translation_prompt_and_cleanup():
    from sentineldesk.generator import build_translation_prompt, clean_translation

    p = build_translation_prompt("Guten Tag, bitte starten Sie neu [TKT-00001].", "de")
    assert "German" in p and "English" in p and "[TKT-00001]" in p
    assert clean_translation("English translation: Hello, please restart [TKT-00001].") == \
        "Hello, please restart [TKT-00001]."
    assert clean_translation('"Hello"') == "Hello"
