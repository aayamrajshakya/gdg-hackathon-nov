import pandas as pd
import pytest

from sentineldesk import config, data


def test_ticket_text_handles_missing_subject():
    assert data.ticket_text(None, "body") == "body"
    assert data.ticket_text(float("nan"), " body ") == "body"
    assert data.ticket_text("", "body") == "body"
    assert data.ticket_text("Subj", "body") == "Subj\nbody"


def test_split_is_brief_compliant():
    df = pd.DataFrame({"subject": list("abcdefghij"), "body": list("ABCDEFGHIJ")})
    kb, test = data.make_split(df)
    assert len(kb) == 8 and len(test) == 2
    assert kb.ticket_id.iloc[0] == "TKT-00000" and test.ticket_id.iloc[0] == "TKT-00008"
    kb2, _ = data.make_split(df)
    assert kb.subject.tolist() == kb2.subject.tolist()  # deterministic (seed 42)


@pytest.mark.skipif(not config.TEST_INPUTS_PATH.exists(), reason="artifacts not built")
def test_holdout_files_contain_no_answers_or_overlap():
    t_in = pd.read_parquet(config.TEST_INPUTS_PATH)
    t_lab = pd.read_parquet(config.TEST_LABELS_PATH)
    kb = pd.read_parquet(config.KB_PATH)
    assert set(t_in.columns) == {"ticket_id", "subject", "body"}
    assert "answer" not in t_lab.columns
    assert not set(t_in.ticket_id) & set(kb.ticket_id)


def test_embedding_format_handles_null_subject():
    from sentineldesk.embeddings import format_ticket

    assert format_ticket(float("nan"), "body") == "title: none | text: body"
    assert format_ticket(None, None) == "title: none | text: "
