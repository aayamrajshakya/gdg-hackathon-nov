from sentineldesk.pii import mask


def test_masks_common_pii():
    r = mask("Mail john.doe@example.com, call +1 617 555 0199, server 10.0.12.7, "
             "IBAN DE89 3704 0044 0532 0130 00, account number: ACC-99812345, password: hunter2")
    for raw in ("john.doe@example.com", "617 555 0199", "10.0.12.7", "DE89", "ACC-99812345", "hunter2"):
        assert raw not in r.text, raw
    assert r.counts["EMAIL"] == 1 and r.counts["IPV4"] == 1 and r.counts["IBAN"] == 1
    assert r.total >= 6


def test_leaves_ordinary_numbers_alone():
    text = "Since version 5.2 in 2024 we have 250 users and it takes 30 seconds."
    assert mask(text).text == text


def test_none_and_empty():
    assert mask("").text == "" and mask(None).text == ""
