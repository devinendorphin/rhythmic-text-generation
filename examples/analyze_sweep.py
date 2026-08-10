"""Sweep analysis: what does a sampling parameter do to rhythm?

Companion to ``analyze_extract.py``. That script describes one text after the
fact; this one reads a completion log, bins generations by a swept parameter
(temperature by default), and reports the metric table per bin against
verse/prose baselines.

Usage:
    python examples/analyze_sweep.py examples/generations/*.json
    python examples/analyze_sweep.py log.json --by top_p

The reason this exists as its own script: the interesting question about a
base model is not "is this sample rhythmic" but "what happens to rhythm as
sampling degrades coherence". You need the parameter attached to the text to
ask it.

READ ``dict_cov`` FIRST, on every row. Out-of-vocabulary words are scored by
letter heuristics that floor at one syllable and impose an alternating 1-0
stress guess, so a low-coverage bin's rhythm numbers describe that fallback
rather than the text. The two synthetic controls printed at the bottom are
there to keep you honest: they are literal random noise, and they post
"perfect pulse" numbers.
"""

from __future__ import annotations

import argparse
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from musicality import ALL_TOOLS, phonology
from musicality.completion_log import CompletionLog

EXAMPLES = pathlib.Path(__file__).resolve().parent

COLUMNS = [
    ("dict_cov", None, None),
    ("syl/w", "syllables", "syllables_per_word"),
    ("stress_d", "stress", "stress_density"),
    ("metricality", "meter", "metricality"),
    ("nPVI", "variability", "npvi_inter_stress"),
    ("entropy", "variability", "stress_entropy_bits"),
    ("sonority", "sonority", "mean_sonority"),
    ("assonance", "rhyme", "assonance_density"),
    ("allit", "alliteration", "alliteration_density"),
    ("euphony", "euphony", "euphony_index"),
]


def metric_row(text: str) -> dict:
    reports = {
        name: ALL_TOOLS[name]().analyze(text)
        for name in ALL_TOOLS if name != "fingerprint"
    }
    row = {
        "words": len(phonology.words(text)),
        "dict_cov": phonology.coverage(text),
    }
    for label, tool, key in COLUMNS:
        if tool:
            row[label] = reports[tool].get(key)
    return row


def noise_controls() -> dict[str, str]:
    """Random letter strings — no language, no phonotactics, no rhythm.

    Any metric that cannot separate these from real text is measuring the
    OOV fallback. They are the floor every other row should be read against.
    """
    rng = random.Random(0)
    def salad(alphabet: str) -> str:
        return " ".join(
            "".join(rng.choice(alphabet) for _ in range(rng.randint(3, 9)))
            for _ in range(400)
        )
    return {
        "CONTROL random consonants": salad("bcdfghjklmnpqrstvwxz"),
        "CONTROL random letters": salad("abcdefghijklmnopqrstuvwxyz"),
    }


def fmt(value, width: int = 9) -> str:
    if value is None:
        return "-".rjust(width)
    if isinstance(value, float):
        return f"{value:>{width}.3f}"
    return f"{value:>{width}}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("logs", nargs="+", help="completion-log JSON export(s)")
    parser.add_argument("--by", default="temperature",
                        help="sampling parameter to bin by (default temperature)")
    parser.add_argument("--min-words", type=int, default=20,
                        help="drop generations shorter than this (default 20)")
    args = parser.parse_args(argv)

    log = CompletionLog.load_all(*args.logs).usable(args.min_words)
    print(f"{len(log)} generations · model {log.model or '?'} · "
          f"swept: {', '.join(log.swept()) or 'nothing'}")

    rows: dict[str, str] = {}
    for name, path in [("BASELINE verse (sonnet18)", "sonnet18.txt"),
                       ("BASELINE prose (austen)", "prose.txt")]:
        rows[name] = (EXAMPLES / path).read_text()
    for value, gens in log.grouped_by(args.by).items():
        rows[f"{args.by}={value}  (n={len(gens)})"] = "\n\n".join(
            g.output for g in gens
        )
    rows.update(noise_controls())

    labels = ["words"] + [c[0] for c in COLUMNS]
    print(f"\n{'':<28}" + "".join(f"{l:>9}"[:9].rjust(10) for l in labels))
    for name, text in rows.items():
        row = metric_row(text)
        print(f"{name:<28}" + "".join(fmt(row.get(l)).rjust(10) for l in labels))

    print("\ndict_cov is the load-bearing column: rows near the CONTROL floor "
          "\nare reporting the letter-heuristic fallback, not pronunciation.")

    per_generation_report(log, args.by)
    return 0


def _pearson(xs: list[float], ys: list[float]) -> float:
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5
    return num / den if den else 0.0


def per_generation_report(log, param: str) -> None:
    """Separate real rhythm effects from the OOV artifact.

    Binning by a parameter pools coherent and degenerate generations together,
    which hides the mechanism — a bin can look well-covered on average while
    containing the junk that is actually moving its rhythm numbers. Measuring
    each generation on its own lets coverage be held constant instead.
    """
    stats = []
    for gen in log:
        variability = ALL_TOOLS["variability"]().analyze(gen.output)
        if variability.get("npvi_inter_stress") is None or gen[param] is None:
            continue
        stats.append({
            "cov": phonology.coverage(gen.output),
            "npvi": variability["npvi_inter_stress"],
            "sd": ALL_TOOLS["stress"]().analyze(gen.output)["stress_density"],
            "param": gen[param],
        })
    if len(stats) < 20:
        return

    col = lambda k: [s[k] for s in stats]
    print(f"\n=== per-generation correlations (n={len(stats)}) ===")
    print(f"  r(coverage, nPVI)           = {_pearson(col('cov'), col('npvi')):+.3f}")
    print(f"  r(coverage, stress_density) = {_pearson(col('cov'), col('sd')):+.3f}")
    print(f"  r({param}, coverage)  = {_pearson(col('param'), col('cov')):+.3f}")

    # The control: hold coverage at ceiling and see what the parameter still does.
    clean = [s for s in stats if s["cov"] >= 0.99]
    if len(clean) >= 20:
        ccol = lambda k: [s[k] for s in clean]
        print(f"\n  holding coverage >= 0.99 (n={len(clean)}) — artifact removed:")
        print(f"    r({param}, nPVI)            = "
              f"{_pearson(ccol('param'), ccol('npvi')):+.3f}")
        print(f"    r({param}, stress_density)  = "
              f"{_pearson(ccol('param'), ccol('sd')):+.3f}")
        print("  Whatever collapses toward zero here was the fallback talking.")


if __name__ == "__main__":
    raise SystemExit(main())
