# rhythmic-text-generation

**Register: working tool.** Real code with real tests — normal software care applies.

**Thesis:** the `musicality` package quantifies how musicality and rhythmicality are
embedded in the structure of language, and whether text-generation systems reproduce them.
Ten tools with a uniform `analyze(text) -> dict` interface over a CMU-Pronouncing-Dictionary
phonological core.

## Repo-specific discipline

- **Run the tests:** `pip install -e ".[dev]" && python -m pytest tests/` (42 tests
  against canonical texts). A new tool needs tests against a text whose answer is known.
- **Keep the interface uniform.** Every tool takes text, returns a dict, emits JSON from the
  CLI. That uniformity is what makes `RhythmicFingerprint` possible.
- **Phonemes, not spelling.** Rhyme and alliteration are computed on pronunciation —
  "known"/"night" alliterate, "cat"/"city" don't. Any shortcut back to orthography is a bug.
- **Report `dictionary_coverage`, not just `dictionary_backed`.** The latter only says
  CMUdict is installed — it reads `True` for pure noise. Coverage of *the measured text* is
  what qualifies a number: OOV words fall back to letter heuristics that floor at one
  syllable with an alternating 1-0 stress guess, so low-coverage text reports the fallback's
  rhythm, not its own. Random consonant strings score a perfect pulse. Check it first.
- **Don't overclaim against acoustics.** These are text-based proxies for acoustic
  phenomena. nPVI here uses counts, not durations — compare within this toolkit, not against
  acoustic studies.

> Standing finding, **substantially corrected 2026-08-10** — read `FINDINGS.md` before
> citing it. The old claim ("degeneration into nonce-word salad is rhythmically
> *hyper-regular*; phonotactics and pulse survive the collapse of semantics") was measured
> through the OOV fallback. Random consonant strings — no language at all — post
> `stress_density 1.0, nPVI 0.0, entropy 0.0`, a *perfect* pulse. Across 350 davinci-002
> generations, `r(coverage, stress_density) = -0.81`: the "pulse" tracks how many words
> CMUdict missed. Hold coverage at ceiling and the stress-density effect vanishes
> (`r = -0.01`). What survives is smaller and real: nPVI still falls with temperature among
> fully in-vocabulary text (`r = -0.26`), never dropping below the metrical-verse baseline.

## The harness

The canonical working agreements, the atlas of all repos, and the shared glossary live in
**`devinendorphin/claude-at-claude`**. Pull it in when you need the full map:

```
add_repo devinendorphin/claude-at-claude
```

This container is ephemeral, so anything that matters gets committed *this turn*. Be a
collaborator rather than a cheerleader, and run a disconfirming test on primed claims.
Endorphin works from a phone and often dictates while walking — expect speech-to-text
artifacts, and mark guessed corrections `[?original→guess]`.

> Sibling: `cobralingus` is the generative half of this interest in language as material.
