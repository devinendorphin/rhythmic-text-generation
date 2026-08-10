"""Tests for completion-log ingest, against the shipped davinci-002 exports."""

import json
import pathlib

import pytest

from musicality.completion_log import CompletionLog, Generation

GENERATIONS = pathlib.Path(__file__).resolve().parent.parent / "examples" / "generations"
GENESIS = GENERATIONS / "davinci002_genesis_20260809.json"
ALTMAN = GENERATIONS / "davinci002_altman_20260810.json"


def test_loads_shipped_export_with_known_shape():
    log = CompletionLog.load(ALTMAN)
    assert len(log) == 306
    assert log.model == "davinci-002"
    assert all(g.model == "davinci-002" for g in log)
    assert all(g.output for g in log)


def test_generation_exposes_params_by_key():
    gen = Generation(output="the cat sat on the mat", params={"temperature": 1.5})
    assert gen["temperature"] == 1.5
    assert gen["top_p"] is None
    assert gen.words == 6


def test_load_all_merges_and_keeps_every_generation():
    merged = CompletionLog.load_all(GENESIS, ALTMAN)
    assert len(merged) == len(CompletionLog.load(GENESIS)) + len(CompletionLog.load(ALTMAN))
    # Both exports are the same model, so it survives the merge unhyphenated.
    assert merged.model == "davinci-002"


def test_usable_drops_short_and_empty_outputs():
    log = CompletionLog(generations=[
        Generation(output="the cat sat on the mat"),          # 6 words
        Generation(output=""),                                 # empty call
        Generation(output=" ".join(["word"] * 40)),            # 40 words
    ])
    assert len(log.usable(min_words=20)) == 1
    assert len(log.usable(min_words=5)) == 2


def test_grouped_by_bins_in_parameter_order():
    log = CompletionLog(generations=[
        Generation(output="a", params={"temperature": 1.5}),
        Generation(output="b", params={"temperature": 1.0}),
        Generation(output="c", params={"temperature": 1.5}),
    ])
    groups = log.grouped_by("temperature")
    assert list(groups) == [1.0, 1.5]
    assert [len(v) for v in groups.values()] == [1, 2]


def test_swept_finds_only_varying_parameters():
    # The shipped sweep varies temperature and top_p; max_tokens is pinned at 150.
    log = CompletionLog.load(ALTMAN)
    assert set(log.swept()) == {"temperature", "top_p"}


def test_shipped_sweep_spans_the_full_temperature_range():
    log = CompletionLog.load_all(GENESIS, ALTMAN).usable()
    temps = sorted(log.grouped_by("temperature"))
    assert temps[0] == 1 and temps[-1] == 2
    assert len(temps) == 11


def test_rejects_non_log_json():
    with pytest.raises(ValueError, match="not a completion log"):
        CompletionLog.from_dict({"model": "davinci-002"})


def test_text_concatenates_outputs():
    log = CompletionLog(generations=[Generation(output="one"), Generation(output="two")])
    assert log.text(joiner=" ") == "one two"
