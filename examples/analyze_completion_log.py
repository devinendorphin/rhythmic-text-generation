"""Rhythmic analysis of a base-model completion log (davinci-002 export format).

A completion log records an interactive writing session: repeated takes from a
seed prompt, each take a chain of sampled continuations that the human either
keeps (it gets absorbed into the next prompt) or rerolls. Sampling parameters
change between takes. That structure makes the log a natural experiment — same
prompt, varying temperature and top_p — so the toolkit can ask which knob
actually moves the rhythm of the text.

Usage:
    python examples/analyze_completion_log.py log.json
    python examples/analyze_completion_log.py log*.json --per-generation

Expected schema (extra keys ignored)::

    {"model": ..., "story": "<final text>", "generations": [
        {"ts":..., "params": {"temperature":..., "top_p":...},
         "prompt": "<story so far>", "output": "<sampled continuation>"}, ...]}

Three things it reports:

1. **Run structure.** Takes are recovered by watching for the prompt length to
   drop (a reset to the seed). Per take: sampling params, how far it got, and
   what share of continuations were kept rather than rerolled.

2. **Partial correlations.** Rhythm metrics against temperature and top_p, each
   controlling for the other — necessary because a human raising temperature
   usually clamps the nucleus at the same time, confounding the two.

3. **Shuffle controls.** For any passage, the same metrics recomputed on its own
   words in scrambled order at identical line lengths. Metrics that survive the
   shuffle reflect word *order*; metrics that don't are properties of the word
   bag (or, for ``end_rhyme_ratio`` on many-line documents, chance collision).
   Never report a rhythm result on generated text without this control.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import pathlib
import random
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from musicality import ALL_TOOLS, phonology

TOOLS = {name: cls() for name, cls in ALL_TOOLS.items() if name != "fingerprint"}

# Metrics correlated against sampling parameters. Keep these order-sensitive
# where possible; `cov` is the lexicality/coherence signal.
METRICS = [
    ("cov", None, None),
    ("stress_density", "stress", "stress_density"),
    ("metricality", "meter", "metricality"),
    ("npvi_inter_stress", "variability", "npvi_inter_stress"),
    ("periodicity_strength", "variability", "periodicity_strength"),
    ("stress_entropy_bits", "variability", "stress_entropy_bits"),
    ("mean_sonority", "sonority", "mean_sonority"),
    ("alliteration_density", "alliteration", "alliteration_density"),
    ("euphony_index", "euphony", "euphony_index"),
    ("syllables_per_word", "syllables", "syllables_per_word"),
]

MIN_WORDS = 60  # below this the metrics are too noisy to correlate


def dictionary_coverage(text: str) -> float | None:
    """Share of tokens found in CMUdict — the lexicality signal.

    Coherent English runs ~95-99%; phonotactically-plausible nonce-word salad
    drops toward 60%. Real-word salad stays near 100%, which is why coverage
    must be read alongside the rhythm metrics rather than instead of them.
    """
    words = phonology.words(text)
    if not words:
        return None
    known = sum(1 for w in words if phonology.phones_for(w) is not None)
    return known / len(words)


def repeated_ngram_rate(text: str, n: int = 4) -> float:
    """Share of n-gram tokens that are repeats — the classic degeneration tell.

    Separates a repetition loop (high) from productive degeneration into fluent
    nonsense (zero), which the rhythm metrics alone do not distinguish.
    """
    words = text.lower().split()
    if len(words) <= n:
        return 0.0
    grams = collections.Counter(
        " ".join(words[i:i + n]) for i in range(len(words) - n + 1)
    )
    return sum(c - 1 for c in grams.values() if c > 1) / len(grams)


def measure(text: str) -> dict | None:
    """All correlated metrics for one passage, or None if it is too short."""
    words = phonology.words(text)
    if len(words) < MIN_WORDS:
        return None
    reports = {name: tool.analyze(text) for name, tool in TOOLS.items()}
    row = {"words": len(words), "cov": dictionary_coverage(text)}
    for label, tool, key in METRICS:
        if tool is not None:
            row[label] = reports[tool][key]
    return row


# --- log structure ----------------------------------------------------------

def split_runs(generations: list[dict]) -> list[list[dict]]:
    """Group generations into takes, splitting where the prompt length drops.

    Within a take the prompt is the story so far, so it grows monotonically; a
    drop means the human restarted from (near) the seed.
    """
    runs: list[list[dict]] = []
    current: list[dict] = []
    previous = None
    for gen in generations:
        length = len(gen["prompt"])
        if previous is not None and length < previous:
            runs.append(current)
            current = []
        current.append(gen)
        previous = length
    if current:
        runs.append(current)
    return runs


def label_kept(run: list[dict], probe_len: int = 60) -> None:
    """Mark each generation kept/rerolled in place.

    A continuation was kept iff its opening survives into the next prompt of the
    same take. The last generation of a take is undetermined — the human may
    simply have stopped — so it is marked None rather than guessed.
    """
    for i, gen in enumerate(run):
        probe = gen["output"].strip()[:probe_len]
        gen["kept"] = bool(probe) and i + 1 < len(run) and probe in run[i + 1]["prompt"]
    if run:
        run[-1]["kept"] = None


def load_runs(paths: list[pathlib.Path]) -> list[dict]:
    """Read logs and return one record per take, with generations labelled."""
    records = []
    for path in paths:
        data = json.loads(path.read_text())
        for index, run in enumerate(split_runs(data["generations"])):
            label_kept(run)
            params = run[0]["params"]
            records.append({
                "source": path.stem,
                "index": index,
                "generations": run,
                "temperature": params["temperature"],
                "top_p": params["top_p"],
                "grown": len(run[-1]["prompt"]) - len(run[0]["prompt"]),
            })
    return records


# --- statistics -------------------------------------------------------------

def pearson(xs: list[float], ys: list[float]) -> float:
    """Pearson r; nan when either series is constant."""
    if len(xs) < 3:
        return math.nan
    mx, my = statistics.mean(xs), statistics.mean(ys)
    sx, sy = statistics.pstdev(xs), statistics.pstdev(ys)
    if sx == 0 or sy == 0:
        return math.nan
    return sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / (len(xs) * sx * sy)


def partial(r_xy: float, r_xz: float, r_yz: float) -> float:
    """Correlation of x and y with z partialled out."""
    denominator = math.sqrt((1 - r_xz ** 2) * (1 - r_yz ** 2))
    if denominator == 0:
        return math.nan
    return (r_xy - r_xz * r_yz) / denominator


# --- shuffle control --------------------------------------------------------

def relineate(text: str, line_lengths: list[int]) -> str:
    """Re-wrap text to a given profile of words-per-line."""
    words = text.split()
    lines, cursor = [], 0
    for length in line_lengths:
        if cursor >= len(words):
            break
        lines.append(" ".join(words[cursor:cursor + length]))
        cursor += length
    return "\n".join(lines)


def shuffle_control(text: str, seed: int = 0) -> str:
    """The same words and line lengths, in scrambled order.

    The null model for "is this rhythm real?": anything the passage scores above
    this is attributable to word order, not to its vocabulary or line shape.
    """
    lines = text.splitlines()
    lengths = [len(line.split()) for line in lines]
    words = text.split()
    random.Random(seed).shuffle(words)
    return relineate(" ".join(words), lengths)


def control_table(text: str, label: str) -> None:
    """Print a passage's metrics beside its shuffled and reversed controls."""
    reversed_lines = "\n".join(
        " ".join(reversed(line.split())) for line in text.splitlines()
    )
    variants = [
        (label, text),
        ("  shuffled (same words)", shuffle_control(text)),
        ("  reversed (same lines)", reversed_lines),
    ]
    print(f"{'passage':<34}{'words':>6}{'cov':>7}{'rep4':>7}"
          f"{'metr':>7}{'nPVI':>7}{'per_str':>9}{'stress':>8}")
    for name, variant in variants:
        row = measure(variant)
        if row is None:
            print(f"{name:<34}  (too short)")
            continue
        print(f"{name:<34}{row['words']:>6}{row['cov']:>7.3f}"
              f"{repeated_ngram_rate(variant):>7.3f}{row['metricality']:>7.3f}"
              f"{row['npvi_inter_stress']:>7.1f}"
              f"{row['periodicity_strength']:>9.3f}"
              f"{row['stress_density']:>8.3f}")


# --- reports ----------------------------------------------------------------

def report_runs(records: list[dict]) -> None:
    print("=== takes ===")
    print(f"{'source':<24}{'#':>3}{'temp':>6}{'top_p':>7}"
          f"{'gens':>6}{'grown':>8}{'kept%':>7}")
    for record in records:
        kept = [g["kept"] for g in record["generations"] if g["kept"] is not None]
        share = f"{100 * sum(kept) / len(kept):.0f}%" if kept else "-"
        print(f"{record['source'][:24]:<24}{record['index']:>3}"
              f"{record['temperature']:>6}{record['top_p']:>7}"
              f"{len(record['generations']):>6}{record['grown']:>8}{share:>7}")


def report_correlations(records: list[dict]) -> list[dict]:
    """Correlate per-generation metrics against temperature and top_p."""
    rows = []
    for record in records:
        for gen in record["generations"]:
            row = measure(gen["output"])
            if row is None:
                continue
            row["temperature"] = gen["params"]["temperature"]
            row["top_p"] = gen["params"]["top_p"]
            row["kept"] = gen["kept"]
            rows.append(row)

    temps = [r["temperature"] for r in rows]
    top_ps = [r["top_p"] for r in rows]
    r_tp = pearson(temps, top_ps)

    print(f"\n=== rhythm vs sampling params (n={len(rows)} generations) ===")
    print(f"temperature and top_p are themselves correlated at r={r_tp:+.2f}; "
          "the partial columns are the ones to read.")
    print(f"{'metric':<22}{'r(temp)':>9}{'r(top_p)':>10}"
          f"{'temp|p':>9}{'top_p|T':>9}")
    for label, _, _ in METRICS:
        values = [r[label] for r in rows]
        r_t, r_p = pearson(temps, values), pearson(top_ps, values)
        print(f"{label:<22}{r_t:>9.3f}{r_p:>10.3f}"
              f"{partial(r_t, r_tp, r_p):>9.3f}{partial(r_p, r_tp, r_t):>9.3f}")

    kept = [r for r in rows if r["kept"] is True]
    rerolled = [r for r in rows if r["kept"] is False]
    if len(rerolled) >= 5:
        print(f"\n=== kept (n={len(kept)}) vs rerolled (n={len(rerolled)}) ===")
        print(f"{'metric':<22}{'kept':>9}{'rerolled':>10}{'delta':>9}")
        for label in ["cov", "metricality", "npvi_inter_stress",
                      "stress_density", "words"]:
            a = statistics.mean(r[label] for r in kept)
            b = statistics.mean(r[label] for r in rerolled)
            print(f"{label:<22}{a:>9.3f}{b:>10.3f}{a - b:>+9.3f}")
    return rows


def report_by_prompt(records: list[dict], head: int = 40) -> None:
    """Group generations by seed prompt and score each cell against its shuffle.

    For prompt-grid experiments (see song_mode_probe.py): pools all output for a
    given seed, then reports the metrics beside the same words scrambled. A cell
    has only produced a rhythm effect if it beats its own shuffled baseline.
    """
    cells = collections.defaultdict(list)
    for record in records:
        for gen in record["generations"]:
            cells[gen["prompt"][:head]].append(gen["output"])

    print(f"\n=== by prompt cell (n={len(cells)}) ===")
    print(f"{'seed prompt':<42}{'gens':>5}{'words':>7}{'cov':>7}"
          f"{'metr':>7}{'d_metr':>8}{'nPVI':>7}{'d_nPVI':>8}{'per_str':>9}")
    for seed, outputs in sorted(cells.items(), key=lambda kv: -len(kv[1])):
        text = "\n".join(outputs)
        row = measure(text)
        if row is None:
            continue
        control = measure(shuffle_control(text))
        label = seed.replace("\n", "\\n")[:40]
        print(f"{label:<42}{len(outputs):>5}{row['words']:>7}{row['cov']:>7.3f}"
              f"{row['metricality']:>7.3f}"
              f"{row['metricality'] - control['metricality']:>+8.3f}"
              f"{row['npvi_inter_stress']:>7.1f}"
              f"{row['npvi_inter_stress'] - control['npvi_inter_stress']:>+8.1f}"
              f"{row['periodicity_strength']:>9.3f}")
    print("d_ columns are the cell minus its own shuffled words: the part "
          "attributable to word order.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("logs", nargs="+", type=pathlib.Path)
    parser.add_argument("--per-generation", action="store_true",
                        help="dump per-generation metrics as JSON")
    parser.add_argument("--by-prompt", action="store_true",
                        help="group by seed prompt and score each cell against "
                             "its shuffled baseline (for prompt grids)")
    args = parser.parse_args(argv)

    records = load_runs(args.logs)
    report_runs(records)
    if args.by_prompt:
        report_by_prompt(records)
    rows = report_correlations(records)

    print("\n=== shuffle control on the final text of each log ===")
    for path in args.logs:
        story = json.loads(path.read_text()).get("story", "")
        if len(story.split()) >= MIN_WORDS:
            control_table(story, path.stem[:32])
            print()

    if args.per_generation:
        print(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
