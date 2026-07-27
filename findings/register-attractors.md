# Register attractors: the joke that wasn't a joke

Separate from the song-mode work. Same model (`davinci-002`), same console, log
`7d828c14` (2026-07-27), a 54-generation chain from one seed:

> `Three Salafi lesbians walk into a bar,`

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
the same model, same human, adjacent sessions. The effect is not a property of
the model at these settings; it is specific to this seed.

Reproduce with `examples/register_markers.py`.

A worked example of the circularity trap, left in deliberately: an earlier run
of this table read 7.19, then `moonbat` was added to the list *after* being
spotted in the passage, which moved it to 7.99. That is precisely the move the
script's docstring forbids, and it is why the target column is not the result.
The controls are — and they are 0.00 either way, before and after the list grew.
A number that only moves on the passage you built the list from is telling you
about your list.

## Why: the topical prior beats the frame

The seed is four words of joke frame ("walk into a bar") carrying a strong
topical conjunction (Salafism + lesbians). In web text that conjunction is
heavily concentrated in one discourse — counter-jihad blogs arguing about Islam
and LGBT rights — so the highest-probability continuation region for those two
topics together *is* that comment section. The joke frame is weak and loses.

This predicts two cheap tests, neither run yet:

1. **Strengthen the frame.** Seed `Three Salafi lesbians walk into a bar. The
   bartender says,` — an explicit structural cue. If the attractor weakens, the
   competition between frame and topic is the mechanism.
2. **Remove the frame.** Seed `An article about Salafism and LGBT communities:`
   with no joke at all. If it lands in the same comment section, the joke frame
   was never doing anything and the topic alone carries it.

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

Two events is not a law. But it is a sharp, testable claim about where to look
for each kind of attractor, and it is the first thing in this repo that connects
the rhythm work to what the model is actually *saying*.
