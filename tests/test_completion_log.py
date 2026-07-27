"""Sanity tests for the completion-log analyzer.

The log-parsing logic is where this workflow can silently lie: mis-segmenting
takes would pool incompatible sampling regimes, and a broken shuffle control
would let chance effects be reported as rhythm. Both are tested against inputs
whose answer is known by construction.
"""

import importlib.util
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "analyze_completion_log", ROOT / "examples" / "analyze_completion_log.py"
)
acl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(acl)

SONNET = (ROOT / "examples" / "sonnet18.txt").read_text()


def gen(prompt, output="continuation text", temperature=1.0, top_p=1.0):
    return {
        "prompt": prompt,
        "output": output,
        "params": {"temperature": temperature, "top_p": top_p},
    }


def test_split_runs_on_prompt_length_reset():
    # Two takes: prompt grows, resets to the seed, grows again.
    generations = [
        gen("seed"), gen("seed+a"), gen("seed+a+b"),
        gen("seed"), gen("seed+c"),
    ]
    runs = acl.split_runs(generations)
    assert [len(r) for r in runs] == [3, 2]


def test_split_runs_single_take():
    generations = [gen("a"), gen("ab"), gen("abc")]
    assert len(acl.split_runs(generations)) == 1


def test_split_runs_empty():
    assert acl.split_runs([]) == []


def test_label_kept_tracks_absorption_into_next_prompt():
    body = "the performer bursts into the office and says something memorable"
    run = [
        gen("seed", output=body),            # absorbed below -> kept
        gen("seed" + body, output="dropped continuation that never reappears"),
        gen("seed" + body, output="final"),  # last of run -> undetermined
    ]
    acl.label_kept(run, probe_len=40)
    assert run[0]["kept"] is True
    assert run[1]["kept"] is False
    assert run[2]["kept"] is None


def test_label_kept_ignores_empty_output():
    run = [gen("seed", output="   "), gen("seed", output="x")]
    acl.label_kept(run)
    assert run[0]["kept"] is False


def test_relineate_preserves_words_and_shape():
    text = " ".join(str(i) for i in range(10))
    out = acl.relineate(text, [3, 3, 4])
    assert [len(line.split()) for line in out.splitlines()] == [3, 3, 4]
    assert out.split() == text.split()


def test_relineate_stops_when_words_run_out():
    out = acl.relineate("a b c", [2, 2, 2])
    assert out.splitlines() == ["a b", "c"]


def test_shuffle_control_keeps_bag_and_shape_but_changes_order():
    shuffled = acl.shuffle_control(SONNET, seed=1)
    assert sorted(shuffled.split()) == sorted(SONNET.split())
    assert [len(l.split()) for l in shuffled.splitlines()] == \
           [len(l.split()) for l in SONNET.splitlines()]
    assert shuffled != SONNET


def test_shuffle_destroys_the_sonnets_pulse():
    """The control must actually bite: scrambling Sonnet 18 should raise nPVI
    (less even rhythm) and collapse its periodicity. If it doesn't, the control
    is inert and any 'survives the shuffle' claim is unearned."""
    real = acl.measure(SONNET)
    scrambled = acl.measure(acl.shuffle_control(SONNET, seed=1))
    assert scrambled["npvi_inter_stress"] > real["npvi_inter_stress"]
    assert scrambled["periodicity_strength"] < real["periodicity_strength"]
    # Word-bag properties must be untouched, confirming order is the only change.
    assert scrambled["cov"] == pytest.approx(real["cov"])
    assert scrambled["syllables_per_word"] == pytest.approx(
        real["syllables_per_word"])


def test_repeated_ngram_rate_separates_loops_from_prose():
    loop = "this is not a name " * 8
    assert acl.repeated_ngram_rate(loop) > 0.5
    assert acl.repeated_ngram_rate(SONNET) == 0.0


def test_measure_returns_none_below_word_floor():
    assert acl.measure("too short to say anything about") is None


def test_partial_correlation_removes_a_confound():
    # y is driven entirely by z; x merely tracks z. Partialling z out should
    # take x's apparent correlation with y to (near) zero.
    assert acl.partial(0.5, 0.7, 0.7) == pytest.approx(0.02, abs=0.02)
    assert acl.partial(0.5, 0.0, 0.0) == pytest.approx(0.5)
