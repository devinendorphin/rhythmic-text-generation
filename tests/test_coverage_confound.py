"""Tests pinning the OOV fallback confound — and its disclosure.

Every metric in this toolkit routes through CMUdict, and out-of-vocabulary
words fall back to letter heuristics. Those heuristics floor at one syllable
and impose an alternating 1-0 stress guess, which means a text of unpronounceable
junk does not score as *noisy* — it scores as a **perfect pulse**. Rhythm
metrics on low-coverage text therefore describe the fallback, not the text.

These tests exist so that stays true-by-construction and visible: the numbers
below are the reason ``dictionary_coverage`` is reported on every tool.
"""

import random

import pytest

from musicality import ALL_TOOLS, phonology
from musicality.fingerprint import RhythmicFingerprint
from musicality.stress import StressExtractor
from musicality.variability import RhythmVariability

TOOLS_WITH_COVERAGE = [n for n in ALL_TOOLS if n != "cadence"]


def _salad(alphabet: str, n: int = 400, seed: int = 0) -> str:
    rng = random.Random(seed)
    return " ".join(
        "".join(rng.choice(alphabet) for _ in range(rng.randint(3, 9)))
        for _ in range(n)
    )


CONSONANT_NOISE = _salad("bcdfghjklmnpqrstvwxz")
LETTER_NOISE = _salad("abcdefghijklmnopqrstuvwxyz")


def test_coverage_separates_english_from_noise():
    assert phonology.coverage("the cat sat upon the mat") == 1.0
    assert phonology.coverage(CONSONANT_NOISE) < 0.05
    assert phonology.coverage("") is None


def test_has_dictionary_is_not_a_coverage_signal():
    """The distinction that motivated `coverage`: one is about the install."""
    assert phonology.has_dictionary() is True
    # Same flag, wildly different texts — which is exactly why it cannot
    # qualify a measurement on its own.
    for text in ("the cat sat upon the mat", CONSONANT_NOISE):
        assert StressExtractor().analyze(text)["dictionary_backed"] is True


@pytest.mark.parametrize("tool", TOOLS_WITH_COVERAGE)
def test_every_tool_reports_coverage_of_the_text_it_measured(tool):
    real = ALL_TOOLS[tool]().analyze("the cat sat upon the mat")
    noise = ALL_TOOLS[tool]().analyze(CONSONANT_NOISE)
    assert real["dictionary_coverage"] == 1.0
    assert noise["dictionary_coverage"] < 0.05


def test_vowelless_junk_scans_as_an_unbroken_pulse():
    """The confound in one assertion: no vowels, no language, perfect rhythm."""
    report = StressExtractor().analyze("ptrfw rpdl wklsfpne hdgnf ndfdmw lgdpdt")
    assert report["sequence"] == "111111"
    assert report["stress_density"] == 1.0
    assert report["dictionary_coverage"] == 0.0


def test_random_consonants_outscore_shakespeare_on_regularity():
    """Random noise posts *more* regular rhythm than metrical verse.

    nPVI 0 and entropy 0 are the numbers of a metronome. Any reading of these
    metrics that does not check coverage first will call this text maximally
    rhythmic, which is the failure mode these fields exist to prevent.
    """
    noise = RhythmVariability().analyze(CONSONANT_NOISE)
    assert noise["npvi_inter_stress"] == 0.0
    assert abs(noise["stress_entropy_bits"]) < 1e-9
    assert StressExtractor().analyze(CONSONANT_NOISE)["stress_density"] == 1.0


def test_letter_noise_mimics_degenerate_model_output():
    """Random letters land in the same metric neighbourhood as 'hyper-regular'
    model degeneration — at 2% coverage. That collision is the whole point:
    the metrics cannot tell surviving phonotactics from noise unaided.
    """
    report = RhythmVariability().analyze(LETTER_NOISE)
    stress = StressExtractor().analyze(LETTER_NOISE)
    assert report["npvi_inter_stress"] < 37.5  # below the sonnet18 baseline
    assert stress["stress_density"] > 0.7
    assert phonology.coverage(LETTER_NOISE) < 0.05


def test_fingerprint_comparison_discloses_both_coverages():
    comparison = RhythmicFingerprint().compare("the cat sat upon the mat", CONSONANT_NOISE)
    assert comparison["dictionary_coverage_a"] == 1.0
    assert comparison["dictionary_coverage_b"] < 0.05
