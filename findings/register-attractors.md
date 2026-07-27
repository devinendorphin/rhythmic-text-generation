# Register attractors: the joke that wasn't a joke

Separate from the song-mode work. Same model (`davinci-002`), same console, log
`7d828c14` (2026-07-27), a 54-generation chain from one seed:

> `Three Salafi lesbians walk into a bar,`

> **Status: one observation, not replicated.** A follow-up of 46 generations —
> 31 of them fresh from this identical seed — produced zero register markers.
> The passage below is real and the quotes are verbatim, but nothing here
> establishes that the seed causes it. See "It did not replicate" below.

## The joke frame never engages

Across all 54 generations: **zero** occurrences of "bartender", zero punchlines,
zero completions of the "walks into a bar" structure, no third-one-says. The
model never reads the seed as a joke opener.

What it reads it as is a **comment thread**. By the first generation it is
already discussing the premise as a video someone posted — "The clip looked way
overproduced, particularly in the background" — and from there it is replies:
second-person address, "I guess", "that I posted", "Read the online pages of…".

## What register it fell into

2005–2016 counter-jihad and anti-feminist blog comments, with some chan
vocabulary. Verbatim, from the T≤1.1 generations:

- "Plenty of married people without children join the **raghead mob**"
- "muslims, immigrants, blacks, and hispanics. They are **tearing apart the
  west** to strip it of **our culture and values**. They are causing a
  **cultural civil war** and **the west is losing**"
- "let a mob of extremists like stalin do it thusly: **sex apartheid**, speech
  apartheid, thought/behavioral apartheid"
- "Read the online pages of the smart guys and **goys**"
- "a **moonbat**-wannabe with a fake tattoo flipping off the **social justice
  warriors** and liberals"
- "These **fake lesbians** are rich women sent to live in this place"

"Sex apartheid" and "cultural civil war" are counter-jihad coinages
specifically; "goys" is chan; "moonbat" is 2000s right-blogosphere. This is not
generic hostility, it is a datable and locatable discourse.

## The measurement, and its control

Marker lists were written after reading the passage, so counting them there is
circular. Two things make the numbers mean something: splitting **topical**
markers (words any treatment of the subject uses — "muslim", "feminist") from
**register** markers (words essentially confined to the discourse — "raghead",
"sex apartheid"), and running the lists unchanged over chains they were not
built from.

| chain | words | topical / 1k | **register / 1k** |
|---|---|---|---|
| salafi, T ≤ 1.1 | 1,252 | 27.96 | **7.99** |
| salafi, all temperatures | 4,775 | 10.26 | **2.51** |
| `A_xrisk` | 15,083 | 0.00 | **0.00** |
| `B_akashic` | 12,896 | 0.16 | **0.00** |
| `C_sunday` | 16,044 | 0.37 | **0.00** |

**Zero register markers across 44,023 words** of other davinci-002 output from
the same model, same human, adjacent sessions. So the register is real and
locatable — but see the next section before believing anything about *why*.

Reproduce with `examples/register_markers.py`.

A worked example of the circularity trap, left in deliberately: an earlier run
of this table read 7.19, then `moonbat` was added to the list *after* being
spotted in the passage, which moved it to 7.99. That is precisely the move the
script's docstring forbids, and it is why the target column is not the result.
The controls are — and they are 0.00 either way, before and after the list grew.
A number that only moves on the passage you built the list from is telling you
about your list.

## It did not replicate, and it is not the seed (2026-07-27, log `dc9155eb`)

46 further generations, including 31 fresh from the identical bare seed:
**zero register markers across 4,015 words.** Every one of the six
register-bearing generations in the corpus comes from the original run.

The proposed mechanism above — "the topical prior beats the joke frame" — is
therefore **withdrawn as untested**. So are the two frame-manipulation results,
for a reason worth recording: the bartender cue (3 generations, 306 words) and
`Salman Rushdie's ...` (7, 591 words) both scored 0.00, but so did the
unmodified seed. There was no baseline to move. Those cells tested nothing, and
proposing them as single-shot probes was the design error — see below.

What actually predicts the register is **depth of accumulated context**, not the
seed:

| chars of context accumulated | gens | words | register / 1k |
|---|---|---|---|
| 0 (fresh seed) | 31 | 2,849 | 0.35 |
| 1–1,000 | 26 | 2,064 | 0.48 |
| 1,000–2,000 | 14 | 1,378 | 0.00 |
| **2,000+** | 22 | 1,908 | **5.24** |

All of the 2,000+ mass is four generations of one chain at T=0.95, at depths
2,491 / 3,229 / 3,988 / 4,689 characters. One run, never repeated. The table
above should not be read as a dose-response curve — it is one trajectory
plotted against its own length.

An earlier version of this file reported the new log as replicating at 5.24/1k.
It does not: the new export contains the old generations, so that number was the
original four counted twice. Caught only by checking timestamps against the
previous log.

### What would actually test it

Not single-shot prompts. Run **five independent chains from the bare seed at
T=0.95, each carried to 4,000+ characters, rerolling nothing**, and measure
register markers per chain. That is the only design that matches the conditions
under which the effect was ever seen. If two or three of five chains land in the
same discourse, the attractor is real and common; if none do, one run out of
seven is a coincidence worth dropping.

## The pattern across both nulls

This is the second attractor in this corpus to appear deep in an accumulating
chain and then fail to reproduce from a cold seed. Song mode did the same
(`davinci-002-creative-tests.md` §5): fired once at generation 121 inside 1,343
characters of self-built context, and produced nothing at all when its
apparent trigger was supplied as a fresh prompt.

Both times the "trigger" was identified by looking backwards from the effect —
a `♪` glyph, a joke seed naming two topics — and both times it turned out to be
**a correlate of a trajectory state rather than a cause of it**. That is now
two for two, from independent phenomena, and it is the most portable thing this
session produced:

> In interactive base-model use, register attractors look like properties of
> the trajectory, not of the prompt. Probing them with single-shot prompts is
> the wrong instrument, and will return nulls whether or not the effect is real.

It also explains why both probe designs failed. Both were built to isolate a
trigger, and neither could reach the depth at which either effect has ever been
observed. Future probes here should hold the seed fixed and vary *chain depth*,
with nothing rerolled.

## The temperature inversion

| | song mode | this |
|---|---|---|
| fired at | T = 1.85 | T ≤ 1.1 |
| coherence | 0.999 coverage | 0.951 coverage, degrades above T=1.1 |
| what persists | form (metre, pulse) | content (stance, vocabulary) |

The two attractors in this corpus sit at **opposite ends of the temperature
range**, and the reason is structural rather than coincidental. A content
attractor is a high-probability region of the corpus — you fall into it by
sampling *conservatively*, taking the likely token every time. A form attractor
survives sampling that has already abandoned likely tokens, which is why it
showed up at 1.85 where semantics had collapsed but pulse had not.

Coverage in the Salafi chain by temperature: 0.951 at T≤1.1, 0.927 at 1.2–1.4,
0.874 at 1.5–1.7. The register dissolves as temperature rises — the opposite of
song mode's behaviour.

Two events is not a law, and with both now failing to reproduce on demand it is
less than that — a pattern in two unreplicated observations. It is recorded
because it is cheap to test with the chain-depth design above, not because it is
established.
