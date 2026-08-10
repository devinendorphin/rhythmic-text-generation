# Findings

## 2026-08-10 — The hyper-regularity of degeneration is mostly an artifact

**Corpus:** 350 `davinci-002` completions in `examples/generations/`, two sessions
(Genesis/BBHMM translation, 44 gens; OpenAI-board limericks and counterfactual
interviews, 306 gens). Sampled across a temperature sweep of **1.0 → 2.0** at
top_p 0.85/0.9/1.0, 150 tokens each. ~229k characters.

Reproduce with:

```bash
python examples/analyze_sweep.py examples/generations/*.json
```

### The prior claim

A Llama 3.1 405B base extract had suggested that degeneration into nonce-word
salad is rhythmically *hyper-regular* rather than noisy — nPVI below metrical
verse, stress density ~0.8 — and the reading was that "phonotactics and pulse
survive the collapse of semantics."

The davinci-002 sweep reproduces those raw numbers. Pooled by temperature,
nPVI falls 50.5 → 35.0 (below sonnet18's 37.5) and stress density climbs
0.44 → 0.54 as temperature rises. On its face, a replication.

### The disconfirming test

Two synthetic controls, both literal random noise with no phonotactics
whatsoever, are now printed by the sweep script and pinned in
`tests/test_coverage_confound.py`:

| text | dict coverage | stress density | nPVI | entropy |
|---|---|---|---|---|
| sonnet18 (metrical verse) | 0.965 | 0.525 | 37.5 | 2.80 |
| austen prose | 0.992 | 0.380 | 56.2 | 2.72 |
| **random consonants** | 0.005 | **1.000** | **0.0** | **0.00** |
| **random letters** | 0.018 | 0.780 | 30.0 | 2.09 |

Random consonant strings score a *perfect* pulse — more regular than
Shakespeare, more regular than anything. Random letters land almost exactly on
the numbers that the original finding called hyper-regular.

The mechanism is in `phonology`: out-of-vocabulary words fall back to
`_heuristic_syllables`, which floors at one syllable and hands out an
alternating 1-0 stress guess. A vowelless token like `ptrfw` is therefore
scored as one stressed monosyllable, and a text of such tokens scans as
`1 1 1 1 1…` — stress density 1.0, inter-stress intervals all identical,
nPVI 0. The rhythm is manufactured by the fallback.

### What the 350 generations show

Pooling by temperature hides this, because each bin mixes coherent and
degenerate generations. Measured per generation (n=349):

```
r(coverage, stress_density) = -0.813
r(coverage, nPVI)           = +0.638
r(temperature, coverage)    = -0.271
```

Stress density is almost entirely a function of OOV rate, and the ladder runs
monotonically into the noise control:

| coverage bin | n | nPVI | stress density |
|---|---|---|---|
| [0.99, 1.00] | 208 | 51.9 | 0.422 |
| [0.95, 0.99) | 85 | 48.6 | 0.451 |
| [0.90, 0.95) | 23 | 46.0 | 0.496 |
| [0.50, 0.80) | 20 | 33.3 | 0.647 |
| [0.00, 0.50) | 6 | 26.3 | 0.773 |
| *random consonants* | — | *0.0* | *1.000* |

Holding coverage at ceiling (≥0.99, n=209) removes the artifact:

```
r(temperature, stress_density) = -0.012   # gone entirely
r(temperature, nPVI)           = -0.255   # survives
```

### Corrected finding

1. **The stress-density half of the claim is artifact.** It vanishes when
   coverage is controlled. "Pulse survives the collapse of semantics" was the
   fallback keeping time, not the model.
2. **A weaker nPVI effect is real.** Among fully dictionary-backed generations,
   higher temperature does produce modestly more even inter-stress intervals
   (53.6 at T=1.0 → 46.5 at T=1.9, r = -0.26). It never crosses below the
   metrical-verse baseline, so "more regular than verse" does not hold.
3. **Phonotactics claims need a phonotactic measure.** davinci-002 degenerates
   in at least two visibly different ways — vowelless consonant strings at
   T=1.9 (`ptrfw rpdl wklsfpne`, coverage 0.00) and plausible pseudo-names at
   T=2.0 (`Nahstroner`, `Crirakk`, coverage 0.32). Only the second is
   phonotactically English-like, and no metric in this toolkit currently tells
   them apart — both route through the same fallback. That is the open gap.

### Change made

`phonology.coverage(text)` now reports per-text dictionary coverage, and every
tool returns it as `dictionary_coverage` alongside the existing
`dictionary_backed` flag. The old field answers "is CMUdict installed" — a
property of the install, `True` even for pure noise — which is not the question
that qualifies a measurement. Read `dictionary_coverage` before reading any
rhythm number off a model-generated text.
