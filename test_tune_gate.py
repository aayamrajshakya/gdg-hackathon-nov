import os
import tempfile

import numpy as np
import pandas as pd

from tune_gate import (
    choose_conf_threshold, choose_soft_threshold, clean_split, is_lfs_pointer,
    make_split, normalize_priority, softmax_with_temperature, fit_temperature,
)


def test_normalize_priority_collapses_to_three_levels():
    s = pd.Series(["High", "critical", "medium", "low", "very_low", None])
    out = normalize_priority(s)
    assert out[:5].tolist() == ["high", "high", "medium", "low", "low"] and pd.isna(out.iloc[5])


def test_clean_split_drops_rows_with_missing_labels():
    df = pd.DataFrame({"subject": ["a", "b"], "body": ["hello there", "hello again"],
                       "type": ["Incident", None], "queue": ["A", "B"],
                       "priority": ["high", "low"], "answer": ["x", "y"]})
    assert len(clean_split(df)) == 1


def test_is_lfs_pointer_detects_unpulled_file():
    d = tempfile.mkdtemp()
    p = os.path.join(d, "x.csv")
    with open(p, "w") as f:
        f.write("version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 123\n")
    assert is_lfs_pointer(p)
    with open(p, "w") as f:
        f.write("subject,body\n")
    assert not is_lfs_pointer(p)


def test_make_split_matches_problem_statement():
    df = pd.DataFrame({"body": [f"b{i}" for i in range(100)]})
    kb, test = make_split(df)
    ref = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    assert len(kb) == 80 and len(test) == 20
    assert kb["body"].tolist() == ref["body"][:80].tolist()
    assert kb["ticket_id"].iloc[0] == "TKT-00000" and test["ticket_id"].iloc[0] == "TKT-00080"


def test_clean_split_handles_nulls():
    df = pd.DataFrame({"subject": [None, "s"], "body": ["hello there", None],
                       "type": ["Incident", "Request"], "queue": ["A", "B"],
                       "priority": ["critical", "low"], "answer": ["a", "b"]})
    out = clean_split(df)
    assert len(out) == 1 and out["subject"].iloc[0] == "" and out["priority"].iloc[0] == "high"


def test_choose_conf_threshold_meets_target():
    conf = np.array([0.95, 0.9, 0.8, 0.7, 0.6, 0.5])
    correct = np.array([1, 1, 1, 0, 0, 0], bool)
    r = choose_conf_threshold(conf, correct, target_accuracy=0.9, fallback_coverage=0.5)
    assert r["threshold"] == 0.8 and r["met_target"]


def test_choose_conf_threshold_falls_back_to_coverage():
    conf = np.array([0.9, 0.8, 0.7, 0.6])
    correct = np.array([0, 1, 0, 1], bool)
    r = choose_conf_threshold(conf, correct, target_accuracy=0.99, fallback_coverage=0.5)
    assert not r["met_target"] and r["threshold"] == 0.8 and r["coverage"] == 0.5


def test_choose_soft_threshold_reaches_target_recall():
    # 4 true high incidents; hard rule catches 2; need 1 more for 75% recall.
    joint = np.array([0.9, 0.8, 0.40, 0.10, 0.5, 0.05])
    hard = np.array([1, 1, 0, 0, 0, 0], bool)
    pos = np.array([1, 1, 1, 1, 0, 0], bool)
    r = choose_soft_threshold(joint, hard, pos, target_recall=0.75)
    assert r["threshold"] == 0.40 and r["recall"] == 0.75


def test_choose_soft_threshold_none_when_hard_rule_suffices():
    r = choose_soft_threshold(np.array([0.9, 0.1]), np.array([1, 0], bool),
                              np.array([1, 0], bool), target_recall=0.9)
    assert r["threshold"] is None


def test_temperature_scaling_reduces_overconfidence():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 2000)
    logits = rng.normal(0, 1, (2000, 3))
    logits[np.arange(2000), y] += 1.0
    logits *= 5                       # make it badly overconfident
    T = fit_temperature(logits, y)
    assert T > 1.5
    p = softmax_with_temperature(logits, T)
    assert np.allclose(p.sum(1), 1)


def test_threshold_rounding_never_rounds_up():
    from tune_gate import floor4
    assert floor4(0.99996) == 0.9999 and floor4(0.5) == 0.5


def test_choose_conf_threshold_enforces_min_coverage():
    conf = np.linspace(1.0, 0.1, 10)
    correct = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0], bool)   # only top-1 meets 0.9
    r = choose_conf_threshold(conf, correct, target_accuracy=0.9, fallback_coverage=0.4)
    assert r["coverage"] == 0.4 and r["relaxed"]
