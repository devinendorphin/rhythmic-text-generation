"""Segment-and-compare workflow for analyzing model-generated extracts.

Runs the full musicality toolkit over a text file — whole and, optionally,
sliced into named segments (e.g. distinct generation regimes of a base-model
sample) — and prints a metric table alongside verse/prose baselines, plus
rhythmic-fingerprint similarities.

Usage:
    python examples/analyze_extract.py extract.txt
    python examples/analyze_extract.py extract.txt \
        -s "word_salad:105-143" -s "coherent:286-350"

Segments are 1-indexed inclusive line ranges of the input file. This is the
workflow used to find that base-model degeneration into nonce-word salad is
rhythmically HYPER-regular (nPVI below metrical verse, stress density ~0.8),
not noisy: phonotactics and pulse survive the collapse of semantics.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from musicality import ALL_TOOLS, phonology
from musicality.fingerprint import RhythmicFingerprint

EXAMPLES = pathlib.Path(__file__).resolve().parent

# (column label, tool, result key) for the summary table.
COLUMNS = [
    ("syll/word", "syllables", "syllables_per_word"),
    ("monosyl_ratio", "syllables", "monosyllabic_ratio"),
    ("stress_density", "stress", "stress_density"),
    ("metricality", "meter", "metricality"),
    ("best_foot", "meter", "best_foot"),
    ("npvi_stress", "variability", "npvi_inter_stress"),
    ("entropy_bits", "variability", "stress_entropy_bits"),
    ("period", "variability", "dominant_period"),
    ("period_strength", "variability", "periodicity_strength"),
    ("mean_sonority", "sonority", "mean_sonority"),
    ("assonance", "rhyme", "assonance_density"),
    ("consonance", "rhyme", "consonance_density"),
    ("alliteration", "alliteration", "alliteration_density"),
    ("sent_len_cv", "cadence", "sentence_length_cv"),
    ("euphony", "euphony", "euphony_index"),
    # end_rhyme_ratio is deliberately omitted: on documents with hundreds of
    # lines it saturates from chance tail-collisions. Run the rhyme tool on
    # stanza-sized excerpts instead.
]


def dictionary_coverage(text: str) -> tuple[float | None, int]:
    """Share of tokens found in CMUdict — a lexicality/coherence signal.

    Coherent English runs ~95-99%; phonotactically-plausible nonce-word
    salad drops toward 60%.
    """
    ws = phonology.words(text)
    if not ws:
        return None, 0
    known = sum(1 for w in ws if phonology.phones_for(w) is not None)
    return round(known / len(ws), 4), len(ws)


def metric_row(text: str) -> dict:
    coverage, n_words = dictionary_coverage(text)
    reports = {
        name: ALL_TOOLS[name]().analyze(text)
        for name in ALL_TOOLS if name != "fingerprint"
    }
    row = {"words": n_words, "dict_coverage": coverage}
    for label, tool, key in COLUMNS:
        row[label] = reports[tool].get(key)
    return row


def parse_segment(spec: str) -> tuple[str, int, int]:
    name, _, span = spec.rpartition(":")
    start, _, end = span.partition("-")
    if not (name and start.isdigit() and end.isdigit()):
        raise argparse.ArgumentTypeError(
            f"segment must look like name:start-end, got {spec!r}"
        )
    return name, int(start), int(end)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("file", help="text file to analyze")
    parser.add_argument(
        "-s", "--segment", type=parse_segment, action="append", default=[],
        metavar="NAME:START-END",
        help="named 1-indexed inclusive line range to analyze separately "
             "(repeatable)",
    )
    args = parser.parse_args(argv)

    full = pathlib.Path(args.file).read_text()
    raw_lines = full.splitlines()

    texts = {
        "baseline:sonnet18": (EXAMPLES / "sonnet18.txt").read_text(),
        "baseline:austen_prose": (EXAMPLES / "prose.txt").read_text(),
        "full_text": full,
    }
    for name, start, end in args.segment:
        texts[f"segment:{name} (l.{start}-{end})"] = "\n".join(
            raw_lines[start - 1:end]
        )

    print(json.dumps({name: metric_row(t) for name, t in texts.items()}, indent=1))

    fingerprint = RhythmicFingerprint()
    print("\n=== fingerprint cosine similarity vs baselines ===")
    for name, text in texts.items():
        if name.startswith("baseline:") or len(phonology.words(text)) < 50:
            continue
        to_verse = fingerprint.compare(
            text, texts["baseline:sonnet18"])["cosine_similarity"]
        to_prose = fingerprint.compare(
            text, texts["baseline:austen_prose"])["cosine_similarity"]
        print(f"{name:40s} verse={to_verse:.4f} prose={to_prose:.4f}")

    segments = [n for n in texts if n.startswith("segment:")]
    if len(segments) >= 2:
        first, last = segments[0], segments[-1]
        print(f"\n=== {first} vs {last}: biggest feature gaps ===")
        comparison = fingerprint.compare(texts[first], texts[last])
        print("cosine:", comparison["cosine_similarity"])
        for d in comparison["largest_differences"]:
            print(f"  {d['feature']:24s} a={d['a']:.3f}  b={d['b']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
