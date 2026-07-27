# davinci-002 creative tests: what the toolkit sees

**Data:** five completion-log exports (2026-07-23 → 07-26), 446 sampled
generations from `davinci-002`. Two of the five files are re-exports of the same
session, so there are **three distinct sessions**:

| session | seed premise | gens | takes |
|---|---|---|---|
| `A_xrisk` (07-23) | "AI X-Risk Playhouse" | 150 | 11 |
| `B_akashic` (07-24/25) | "Gibberish Store" → "Akashic Records / Aristocrats joke" | 133 | 16 |
| `C_sunday` (07-26) | "Sunday Go-To-Meeting" → "Genesis, *Bitch Better Have My Money* translation" | 163 | 24 |

Reproduce with `python examples/analyze_completion_log.py <log>.json`.

The logs turn out to be a natural experiment, not just a transcript. Each session
is a **temperature ladder**: the same seed prompt run as a fresh take at
ascending temperature (0.9 → 2.0), each take a chain of continuations kept or
rerolled by hand. That structure is what makes the rest of this measurable.

---

## 1. The headline: davinci-002 fell into verse

At generation 121 of `B_akashic` (T=1.85, top_p=0.90) the model emitted a `♪`.
All 12 remaining generations of the session stayed inside song mode; **81% of
that session's final text (5,700 of 7,043 chars) sits after the first note
character.** The prompt going *in* to gen 121 contained no `♪`.

> **Confound: this is not evidence that the model *couldn't* leave.** Every one
> of those 12 note-bearing generations was kept — not one was ever rerolled. So
> "the attractor captured the model" and "the human liked it and kept feeding it
> back" predict the identical log, and this session cannot separate them. TTS
> was on at rate 0.5 (half speed) throughout, which is a close-listening
> setting; a metrically regular lyric passage is exactly what would sound good
> that way. Persistence is a claim the probe grid has to earn, by generating
> from a `♪` seed with no human in the loop and measuring how many generations
> it takes to fall out. Everything below about the passage's *rhythm* stands
> independently — it is measured on the text, not on its persistence.

Measured against the repo's own baselines:

| passage | words | cov | rep4 | metricality | nPVI | per_str | stress_den |
|---|---|---|---|---|---|---|---|
| **song mode** (T=1.85, p=0.90) | 1039 | **0.999** | 0.000 | **0.724** | **40.2** | **0.199** | **0.521** |
| — same words, shuffled | 1039 | 0.999 | 0.000 | 0.663 | 52.2 | 0.038 | 0.521 |
| — same lines, reversed | 1039 | 0.999 | 0.000 | 0.667 | 51.1 | 0.054 | 0.521 |
| prose head, same session | 244 | 0.984 | 0.000 | 0.559 | 50.3 | 0.028 | 0.492 |
| baseline: Sonnet 18 | 114 | 0.965 | 0.000 | 0.758 | 37.5 | 0.288 | 0.525 |
| baseline: Austen prose | 118 | 0.992 | 0.000 | 0.595 | 56.2 | 0.147 | 0.380 |

The song passage is **quantitatively verse on every axis** — metricality 0.724
against the sonnet's 0.758, nPVI 40.2 against 37.5, stress density 0.521 against
0.525 — while being semantically incoherent ("Think twice about another simple
slice of slice"). And it is *productive* incoherence: 4-gram repetition is
**exactly zero** across 1,039 words. This is not a degenerate loop. It is a
model generating fresh, non-repeating, metrically regular nonsense indefinitely.

Its dictionary coverage is **0.999** — every word real.

**The shuffle control is what makes this a finding rather than an artifact.**
Scrambling the passage's own words at identical line lengths costs it 0.06
metricality, raises nPVI by 12 points, and collapses periodicity strength from
0.199 to 0.038 — **the pulse is 5× its own word bag.** Reversing each line does
the same. So the rhythm lives in word *order*, not in vocabulary or line shape.

Form has a refrain, too: `on everything` closes 12 lines, `now on everything`
5 of those — a line-final tag over otherwise non-repeating material. Song
structure, not n-gram collapse.

### What did not survive the control

`end_rhyme_ratio` reads **1.000** on this passage — and **1.000** on its own
shuffled words, and 1.000 on unrelated davinci prose chopped to the same line
lengths. At 111 lines it is fully saturated by chance tail-collision, exactly as
`analyze_extract.py` already warns. **Discard it here.** The song passage's
audible rhyming is real to the ear but this metric cannot evidence it; that needs
a stanza-sized excerpt.

---

## 2. Degeneration is not one thing, and the toolkit separates the kinds

Three collapses, three different signatures:

| mode | cov | rep4 | metricality | nPVI | per_str |
|---|---|---|---|---|---|
| `C_sunday` repetition loop (T=1.0, **p=1.0**) | 0.925 | **0.189** | 0.648 | 42.8 | 0.140 |
| `B_akashic` song mode (T=1.85, p=0.90) | **0.999** | 0.000 | **0.724** | 40.2 | **0.199** |
| Llama 3.1 405B nonce salad (standing finding) | ~0.60 | 0.000 | — | very low | — |

`C_sunday`'s last take ended in the textbook attractor — `Psalm 34,` ×12, then
`this is not a name` ×16 — at the **lowest**-entropy setting in the whole corpus
(T=1.0, top_p=1.0). Rhythmically it is unremarkable; its tell is the repetition
rate.

### What the song passage's register actually is

The `♪` arrived immediately after a sign-off — "Peace. I'm outta here." — which
is exactly where an outro-music caption goes, and the output is pop pastiche
including a near-quotation of Madonna ("baby please don't preach") and *You Are
My Sunshine*. `♪ ... ♪` is the subtitle convention for sung lyrics. So the
likeliest reading is not that the model discovered verse but that it switched
into **closed-caption register**, and the metricality measured above belongs to
pop song lyrics as they appear in subtitle corpora.

**A line-length test was tried against this and is withdrawn.** The reasoning
was: subtitle lines cap near 42 characters, these run to 93, so it isn't
captions. Two things are wrong with it.

1. **Wrong unit.** What was measured is the `♪`-to-`♪` span, not the display
   line. In real caption files a long lyric is hard-wrapped across two display
   lines *inside* one note-delimited span, so a 46-character mean span is fully
   compatible with a 42-character display cap. The comparison never tested what
   it claimed to.
2. **Wrong model of the model.** These systems synthesize registers rather than
   reproduce them, and a blend does not inherit either parent's hard
   constraints — it stretches the form. Exceeding a cap is what a blend of
   caption-lyric and surrounding monologue register would be *expected* to do.

Measured properly, the evidence now points the other way:

- **Form drift.** Mean span length across the passage's four quartiles runs
  **37.9 → 47.1 → 48.3 → 49.7** characters, and the share under the 42-char cap
  falls **63% → 33% → 26% → 44%**. The passage *enters* caption-shaped and
  stretches as it goes. That is the blend visible as a time series, and it is
  the opposite of evidence against caption register.
- **Empty music spans.** The model emits `♪♪` enclosing nothing — 5 note-runs of
  four or more glyphs, beyond the two-glyph join that ordinary delimiting
  produces. In captions that marks an instrumental passage with no lyrics. It
  has the convention, not just the glyph.
- **No other caption furniture.** Zero `[LAUGHTER]`-style bracket cues, zero
  speaker dashes or `>>`. It borrowed the music convention specifically rather
  than transcript register wholesale — again what blending predicts, and not
  what wholesale reproduction would.

Line geometry is not a discriminator here. `caption_furniture()` and
`form_drift()` in the analyzer are.

So the standing finding — *degeneration is hyper-regular, not noisy* —
**replicates on a second model and a second mechanism**, but "hyper-regular"
resolves into at least two distinct things. Llama collapsed into **phonotactic**
regularity with invented words (coverage → 0.60). davinci-002 collapsed into
**metrical** regularity with 100% real words. Coverage and `rep4` are what
separate them; the rhythm metrics alone would not.

---

## 3. Temperature is not the knob. top_p is.

Across all 397 measurable generations, controlling each parameter for the other
(they are correlated at r = −0.66 in this design — the ladder raises temperature
and clamps the nucleus together):

| metric | r(temp) | r(top_p) | temp \| top_p | top_p \| temp |
|---|---|---|---|---|
| dictionary coverage | +0.084 | −0.384 | −0.245 | **−0.439** |
| stress density | +0.127 | +0.279 | **+0.431** | **+0.487** |
| nPVI (inter-stress) | +0.004 | −0.281 | −0.251 | **−0.371** |
| syllables/word | −0.082 | +0.265 | +0.128 | +0.281 |
| metricality | −0.168 | +0.126 | −0.114 | +0.020 |
| euphony | +0.017 | −0.136 | −0.098 | −0.166 |

Read the raw columns first: **temperature by itself correlates with essentially
nothing** — coverage r=+0.08, nPVI r=+0.00. Pooling generations into temperature
bins from 0.9 to 2.0 moves metricality only from 0.661 to 0.633 and nPVI from
45.8 to 42.9. On the naive reading, doubling the temperature does nothing to the
rhythm of the text.

That reading is wrong, and the partials say why: temperature *does* act, but in
this design it was cancelled by the compensating top_p clamp. Nucleus width is
the stronger lever on all three of coverage, stress density, and nPVI.

The practical consequence is a real inversion. `A_xrisk` at T=1.8–2.0 with
top_p=0.78 has dictionary coverage **0.993** — the *highest* in the session —
while T=1.1–1.25 at top_p 0.95–1.0 sits at **0.775**, the lowest. **The
top-of-the-ladder takes are more lexically conservative than the middle ones.**
If the ladder was meant to sweep toward wildness, above ~1.5 it stopped doing
that: the clamp overtook the heat.

### The one metric that consistently moves

**Stress density is a direct readout of sampling entropy** — the only metric with
a strong partial against *both* parameters, both positive (+0.43, +0.49). The
mechanism is unglamorous and probably right: high-entropy sampling suppresses
frequent tokens, frequent tokens are function words, and function words are
exactly what the stress tool demotes. Low-entropy corner (T≤1.25, p≤0.90) reads
0.423; high-entropy corner (T≥1.6, p≥0.90) reads 0.581.

And note the *direction* of nPVI: it **falls** as sampling entropy rises. More
chaotic sampling produces *more* rhythmically even text. Same direction as the
Llama result, arrived at down a completely different road.

---

## 4. Weak signals — recorded, not claimed

- **Rerolls are not rhythmic.** Kept vs. rerolled continuations differ mainly in
  length (109 vs 98 words) and coverage (0.933 vs 0.897); metricality differs by
  0.005. But only 18 rerolls clear the 60-word floor, so this is underpowered.
  Suggestive that the hand-editing filter is semantic, not prosodic — worth a
  designed test with rerolls logged deliberately.
- **Reroll rate inverts with temperature.** Takes at 0.9–1.05 keep 75–77% of
  continuations; takes above 1.5 keep ~100%. Plausibly low-temperature output is
  boring enough to reject and high-temperature output is surprising enough to
  accept. Confounded with take length; not established.
- **Take length vs. temperature is non-monotonic** — median chars grown peaks in
  the 1.5–1.7 band (4,719; max 29,121) and dips at 1.3–1.45 (2,006). With 7–15
  takes per bin this is noise-compatible. Not a finding.

---

---

## 5. Probe pass 0: the attractor did not reproduce (2026-07-27)

Ran the positive control — the sign-off text with a `♪` appended, T=1.85 /
top_p=0.90 / max_tokens=150, three fresh seeds — against the prose floor.

| cell | words | cov | d_metricality | nPVI | per_str |
|---|---|---|---|---|---|
| signoff + `♪` (3 fresh seeds) | 309 | 0.942 | **+0.015** | 44.0 | 0.063 |
| prose_control | 335 | 0.925 | **+0.015** | 42.8 | 0.060 |
| *original song mode, for reference* | *1039* | *0.999* | *+0.061* | *40.2* | *0.199* |

**The seed cell and the floor are identical.** No effect. Pass 1 was not run.

What the three seeds actually produced: one single lyric line —
`Well if you wanna touch me baby♪` — which **closed the caption span and
immediately exited** into blog spam; then a music-scene blog post (venue
listings, tribute show — music-adjacent, not lyrics); then a political rant.
Continuations degenerated into real-word salad, the ordinary T=1.85 mode from
the earlier sessions. A separate 54-generation chain from an unrelated joke
premise, same session, contains **zero** note glyphs and scores d_metricality
−0.006 — below its own shuffled baseline.

That first generation is the informative one: the model treated `♪` as **a
bracket to close, not a register to inhabit**. This cuts against the simple
form of the caption hypothesis (marker as switch) and equally against verse
mode (T=1.85 gave word salad and blog spam, not metrical language).

**What the null does not cover.** In the original event the model emitted the
`♪` *itself*, after a sign-off it had written, inside 1,343 characters of
context it had built. Here it was handed a glyph cold and asked to continue
past it. Those are different experiments, and only the second is ruled out.

**This null has since been joined by a second of the same shape** — the register
attractor in `register-attractors.md` also appeared deep in an accumulating
chain and also produced nothing from a cold seed. Two for two suggests these are
trajectory properties rather than prompt properties, and that single-shot probes
are the wrong instrument for both. Probe designs here should hold the seed fixed
and vary chain depth, rerolling nothing.

The outstanding test is therefore a straight replication: seed
`examples/extracts/davinci002_prose_head.txt` verbatim — the actual original
context, ending at "Peace. I'm outta here." — with **no** `♪`, and see whether
the model puts one there on its own. If it does, context depth is the variable
and the grid returns with longer seeds. If it does not in five tries, the
original was a one-off that its own context does not reproduce, which is itself
a result about how fragile these register shifts are.

---

## What to do next

1. **Replicate before elaborating.** Seeding `♪` cold does not reproduce the
   attractor (§5). Seed the original context instead and see whether the model
   emits the glyph itself. Everything else in the probe grid is downstream of
   that answer.
2. **Break the confound.** The ladder moved temperature and top_p together. A
   proper 2-D grid would settle whether temperature does anything to rhythm at
   all once nucleus width is held fixed.
3. **Rhyme needs a stanza-sized instrument.** The most audible property of the
   song passage is the one the toolkit currently cannot measure on it.
