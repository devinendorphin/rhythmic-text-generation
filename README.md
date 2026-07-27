# rhythmic-text-generation

Tools for researching how musicality and rhythmicality are embedded in the
structure of language — and how (or whether) text generation systems
reproduce them.

The `musicality` package provides **ten quantification tools**, each with a
uniform `analyze(text) -> dict` interface, a shared phonological core built
on the CMU Pronouncing Dictionary (with heuristic fallback for
out-of-vocabulary words), and a JSON-emitting CLI.

## Installation

```bash
pip install -e ".[dev]"   # installs cmudict + pytest
python -m pytest tests/   # 18 sanity tests against canonical texts
```

## The ten tools

| # | Tool | What it quantifies |
|---|------|--------------------|
| 1 | `SyllableProfiler` | Syllable distribution — the basic pulse. Syllables/word, monosyllabic ratio, texture evenness. |
| 2 | `StressExtractor` | The stress sequence ("drum track") of a text: binary beat/offbeat per syllable, inter-stress intervals, stress density. Function words are demoted, as in real scansion. |
| 3 | `MeterDetector` | Best-fit classical foot (iamb, trochee, anapest, dactyl, amphibrach) at every phase offset, per line and overall, plus a 0–1 **metricality** score and meter name (e.g. *iambic pentameter*). |
| 4 | `RhythmVariability` | Regularity measures from prosody/music research: **nPVI** (Patel & Daniele 2003) over inter-stress intervals and word weights, stress n-gram **entropy**, and **autocorrelation** (lag-2 peak = binary meter, lag-3 = ternary). |
| 5 | `SonorityAnalyzer` | The sonority contour — language's melodic line. Mean sonority (Clements hierarchy, 1–10), smoothness of transitions, oscillation rate, vowel/sonorant/obstruent balance. |
| 6 | `RhymeAnalyzer` | Phoneme-based (not spelling-based) end-rhyme scheme (ABAB…), rhyme pairs, internal rhyme, and assonance/consonance densities in a proximity window. |
| 7 | `AlliterationAnalyzer` | Onset repetition density in a sliding window — the organizing device of Germanic alliterative verse — computed on phonemes ("known"/"night" alliterate; "cat"/"city" don't). |
| 8 | `CadenceAnalyzer` | Rhythm above the syllable: sentence- and clause-length series analyzed for periodicity via autocorrelation and DFT — the long-short "phrasing" of prose. |
| 9 | `EuphonyScorer` | A 0–100 phonaesthetic index rewarding liquids/nasals, open vowels and CV alternation, penalizing obstruent clusters and sibilant pileups — with per-component diagnostics. |
| 10 | `RhythmicFingerprint` | Aggregates the other nine into a 14-dimensional normalized feature vector; compares texts by cosine similarity and reports which features most separate them. |

## CLI

```bash
python -m musicality analyze examples/sonnet18.txt              # all ten tools
python -m musicality analyze examples/tyger.txt -t meter        # one tool
python -m musicality compare examples/sonnet18.txt examples/prose.txt
echo "some generated text" | python -m musicality analyze -
```

Output is JSON, ready for `jq`, pandas, or notebooks.

### Sample results (from `examples/`)

- Sonnet 18 → `"meter_name": "iambic pentameter"`, metricality 0.76
- Sonnet 18 vs. Austen prose → the largest fingerprint differences are
  `end_rhyme_ratio`, `npvi_inter_stress`, `metricality`, `stress_density` —
  precisely the dimensions that ought to separate verse from prose.

## Python API

```python
from musicality import MeterDetector, RhythmicFingerprint

MeterDetector().analyze(open("examples/tyger.txt").read())["meter_name"]

fp = RhythmicFingerprint()
fp.compare(human_poem, llm_poem)["largest_differences"]
```

### Analyzing model extracts

`examples/analyze_extract.py` runs the full toolkit over a generated sample —
whole and sliced into named line-range segments (e.g. distinct generation
regimes) — printing a metric table against the verse/prose baselines plus
fingerprint similarities:

```bash
python examples/analyze_extract.py sample.txt -s "word_salad:105-143" -s "coherent:286-350"
```

Applied to a Llama 3.1 405B base-model extract, this workflow showed that
degeneration into nonce-word salad is rhythmically *hyper-regular*, not
noisy — nPVI drops below metrical verse, stress density approaches 0.8, and
the invented words keep English onset statistics: phonotactics and pulse
survive the collapse of semantics.

### Analyzing interactive completion logs

`examples/analyze_completion_log.py` handles the other common shape of model
output: a log of an interactive writing session, where the same seed prompt is
run as repeated takes under varying sampling parameters and each continuation is
kept or rerolled by hand.

```bash
python examples/analyze_completion_log.py log.json          # one session
python examples/analyze_completion_log.py log*.json         # pooled
```

It recovers the take structure, labels kept vs. rerolled continuations,
correlates every rhythm metric against temperature *and* top_p with each
controlled for the other, and prints a **shuffle control** — the same words in
scrambled order at identical line lengths. Metrics that beat their own shuffled
baseline reflect word order; metrics that don't are properties of the word bag.
Don't report a rhythm result on generated text without that control.

Applied to `davinci-002` creative sessions, it found a passage where the model
fell into song mode and scored as verse (metricality 0.72 vs. Sonnet 18's 0.76,
periodicity 5× its own shuffled baseline) while being semantically incoherent
with 99.9% dictionary coverage and *zero* 4-gram repetition — and found that
top_p, not temperature, is the parameter that moves the text. See
[`findings/davinci-002-creative-tests.md`](findings/davinci-002-creative-tests.md).

## Research uses

- **Corpus contrasts**: fingerprint human poetry vs. LLM poetry vs. prose and
  see which rhythmic dimensions models reproduce or flatten.
- **Prompt sensitivity**: does asking a model for "musical" prose actually
  move metricality, euphony, or nPVI — or just vocabulary?
- **Training-signal questions**: nPVI was originally used to show composers'
  music echoes their native language's speech rhythm; the same instrument
  applied to generated text can probe whether models internalize
  stress-timing.

## Caveats

All measures are text-based proxies for phenomena that are ultimately
acoustic. Stress comes from citation-form lexical entries (no sentence-level
prosody model); nPVI here uses syllable/interval counts, not durations, so
compare values within this toolkit rather than against acoustic studies.
Out-of-vocabulary words fall back to letter heuristics, and each tool
reports `dictionary_backed` so you can track coverage.
