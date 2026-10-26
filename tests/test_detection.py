import pytest

from detection import CIREEDetector, split_sentences


def test_split_sentences():
    text = "Wildlife populations fell 60 percent.  We can flip that script! “Why?” she asked."
    assert split_sentences(text) == [
        "Wildlife populations fell 60 percent.",
        "We can flip that script!",
        "“Why?” she asked.",
    ]


def test_split_keeps_decimals_and_lowercase_continuations():
    assert split_sentences("It grew 2.5 times. e.g. this stays.") == ["It grew 2.5 times. e.g. this stays."]


def test_split_empty():
    assert split_sentences("   ") == []


@pytest.mark.slow
def test_detector_labels_fact():
    det = CIREEDetector()
    fact, narrative = det.predict([
        "In the last 50 years, wildlife populations have declined by 60 percent.",
        "We can flip that script and change it.",
    ])
    assert fact["fact_probability"] > narrative["fact_probability"]
