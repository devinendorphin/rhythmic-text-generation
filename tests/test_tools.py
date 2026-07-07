"""Sanity tests: each tool must recover known properties of canonical texts."""

import json
import pathlib
import subprocess
import sys

import pytest

from musicality import ALL_TOOLS
from musicality.alliteration import AlliterationAnalyzer
from musicality.cadence import CadenceAnalyzer
from musicality.euphony import EuphonyScorer
from musicality.fingerprint import RhythmicFingerprint
from musicality.meter import MeterDetector
from musicality.rhyme import RhymeAnalyzer
from musicality.sonority import SonorityAnalyzer
from musicality.stress import StressExtractor
from musicality.syllables import SyllableProfiler
from musicality.variability import RhythmVariability
from musicality import phonology

EXAMPLES = pathlib.Path(__file__).resolve().parent.parent / "examples"
SONNET = (EXAMPLES / "sonnet18.txt").read_text()
PROSE = (EXAMPLES / "prose.txt").read_text()
TYGER = (EXAMPLES / "tyger.txt").read_text()


def test_syllable_counts_known_words():
    assert phonology.syllable_count("cat") == 1
    assert phonology.syllable_count("language") == 2
    assert phonology.syllable_count("symmetry") == 3
    assert phonology.syllable_count("universally") == 5


def test_syllable_profile_shape():
    report = SyllableProfiler().analyze(SONNET)
    assert report["word_count"] > 100
    assert 1.0 <= report["syllables_per_word"] <= 2.0
    assert 0 < report["monosyllabic_ratio"] < 1


def test_stress_sequence_binary_and_function_words_demoted():
    report = StressExtractor().analyze("the cat sat on the mat")
    # Function words unstressed, content monosyllables stressed.
    assert report["sequence"] == "011001"
    assert set(report["sequence"]) <= {"0", "1"}


def test_meter_detects_iambic_pentameter_in_sonnet():
    report = MeterDetector().analyze(SONNET)
    assert report["best_foot"] == "iamb"
    assert "pentameter" in report["meter_name"]
    assert report["metricality"] > 0.7


def test_meter_detects_trochaic_tyger():
    report = MeterDetector().analyze(TYGER)
    assert report["best_foot"] in ("trochee", "iamb")  # catalectic lines blur phase
    assert report["metricality"] > 0.7


def test_verse_more_metrical_than_prose():
    sonnet = MeterDetector().analyze(SONNET)["metricality"]
    prose = MeterDetector().analyze(PROSE)["metricality"]
    assert sonnet > prose


def test_variability_reports_all_measures():
    report = RhythmVariability().analyze(SONNET)
    assert report["npvi_inter_stress"] is not None
    assert 0 < report["stress_entropy_bits"] <= report["max_entropy_bits"]
    assert report["dominant_period"] in range(1, 9)


def test_sonnet_stress_more_periodic_than_prose():
    sonnet = RhythmVariability().analyze(SONNET)
    prose = RhythmVariability().analyze(PROSE)
    assert sonnet["periodicity_strength"] > prose["periodicity_strength"]


def test_sonority_scale_and_ratios():
    report = SonorityAnalyzer().analyze(PROSE)
    assert 1 <= report["mean_sonority"] <= 10
    total = (
        report["vowel_ratio"]
        + report["sonorant_consonant_ratio"]
        + report["obstruent_ratio"]
    )
    assert total == pytest.approx(1.0, abs=0.02)  # glide/H rounding


def test_rhyme_scheme_of_sonnet_quatrain():
    # Second quatrain: shines/declines, dimmed/untrimmed. (The first
    # quatrain's temperate/date is a historic rhyme that no longer holds
    # phonetically, and the analyzer rightly rejects it.)
    quatrain = "\n".join(SONNET.splitlines()[4:8])
    report = RhymeAnalyzer().analyze(quatrain)
    assert report["rhyme_scheme"] == "ABAB"
    assert report["end_rhyme_ratio"] == 1.0


def test_rhyme_is_phonetic_not_orthographic():
    report = RhymeAnalyzer().analyze("it was rough\nhe had stuff")
    assert report["rhyme_scheme"] == "AA"


def test_alliteration_peter_piper():
    dense = AlliterationAnalyzer().analyze(
        "Peter Piper picked a peck of pickled peppers"
    )
    plain = AlliterationAnalyzer().analyze(PROSE)
    assert dense["alliteration_density"] > 0.7
    assert dense["alliteration_density"] > plain["alliteration_density"]
    assert dense["top_onsets"].get("P", 0) >= 5


def test_cadence_measures_prose():
    report = CadenceAnalyzer().analyze(PROSE)
    assert report["sentence_count"] >= 4
    assert report["mean_sentence_syllables"] > 5
    assert "dominant_sentence_period" in report


def test_euphony_mellifluous_beats_harsh():
    smooth = EuphonyScorer().analyze(
        "the low melodious moan of a lone lily in the meadow"
    )
    harsh = EuphonyScorer().analyze(
        "strict tax code checks blocked six struck stock texts"
    )
    assert smooth["euphony_index"] > harsh["euphony_index"]
    assert 0 <= harsh["euphony_index"] <= 100


def test_fingerprint_vector_and_self_similarity():
    fp = RhythmicFingerprint()
    report = fp.analyze(SONNET)
    assert report["dimensions"] == 14
    self_cmp = fp.compare(SONNET, SONNET)
    assert self_cmp["cosine_similarity"] == pytest.approx(1.0)
    assert self_cmp["euclidean_distance"] == pytest.approx(0.0)


def test_fingerprint_separates_verse_from_prose():
    fp = RhythmicFingerprint()
    verse_pair = fp.compare(SONNET, TYGER)["cosine_similarity"]
    cross_pair = fp.compare(SONNET, PROSE)["cosine_similarity"]
    assert verse_pair > cross_pair


def test_all_tools_handle_empty_and_tiny_input():
    for cls in ALL_TOOLS.values():
        for text in ("", "the", "!!!"):
            report = cls().analyze(text)
            assert isinstance(report, dict)  # error dict is fine, crash is not


def test_cli_analyze_and_compare():
    out = subprocess.run(
        [sys.executable, "-m", "musicality", "analyze",
         str(EXAMPLES / "sonnet18.txt"), "-t", "meter"],
        capture_output=True, text=True, check=True,
        cwd=EXAMPLES.parent,
    )
    assert json.loads(out.stdout)["meter"]["best_foot"] == "iamb"

    out = subprocess.run(
        [sys.executable, "-m", "musicality", "compare",
         str(EXAMPLES / "sonnet18.txt"), str(EXAMPLES / "prose.txt")],
        capture_output=True, text=True, check=True,
        cwd=EXAMPLES.parent,
    )
    assert 0 <= json.loads(out.stdout)["cosine_similarity"] <= 1
