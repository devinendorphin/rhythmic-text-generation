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
