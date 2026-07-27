# rhythmic-text-generation

**Register: working tool.** Real code with real tests — normal software care applies.

**Thesis:** the `musicality` package quantifies how musicality and rhythmicality are
embedded in the structure of language, and whether text-generation systems reproduce them.
Ten tools with a uniform `analyze(text) -> dict` interface over a CMU-Pronouncing-Dictionary
phonological core.

## Repo-specific discipline

- **Run the tests:** `pip install -e ".[dev]" && python -m pytest tests/` (18 sanity tests
  against canonical texts). A new tool needs tests against a text whose answer is known.
- **Keep the interface uniform.** Every tool takes text, returns a dict, emits JSON from the
  CLI. That uniformity is what makes `RhythmicFingerprint` possible.
- **Phonemes, not spelling.** Rhyme and alliteration are computed on pronunciation —
  "known"/"night" alliterate, "cat"/"city" don't. Any shortcut back to orthography is a bug.
- **Report `dictionary_backed`.** Out-of-vocabulary words fall back to letter heuristics;
  coverage must stay visible.
- **Don't overclaim against acoustics.** These are text-based proxies for acoustic
  phenomena. nPVI here uses counts, not durations — compare within this toolkit, not against
  acoustic studies.

> Standing finding worth not losing: applied to a Llama 3.1 405B base extract, degeneration
> into nonce-word salad is rhythmically *hyper-regular*, not noisy. Phonotactics and pulse
> survive the collapse of semantics.

## The harness

The canonical working agreements, the atlas of all 20 repos, and the shared glossary live in
**`devinendorphin/claude-at-claude`**. Pull it in when you need the full map:

```
add_repo devinendorphin/claude-at-claude
```

This container is ephemeral, so anything that matters gets committed *this turn*. Be a
collaborator rather than a cheerleader, and run a disconfirming test on primed claims.
Endorphin works from a phone and often dictates while walking — expect speech-to-text
artifacts, and mark guessed corrections `[?original→guess]`.

> Sibling: `cobralingus` is the generative half of this interest in language as material.
