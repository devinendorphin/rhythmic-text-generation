"""Prompt grid for chasing the davinci-002 song-mode attractor.

Background: in one session davinci-002 emitted a lone `♪` mid-generation and
every generation after it stayed in song mode — 81% of the resulting text,
scoring as verse (metricality 0.724 against Sonnet 18's 0.758) while
semantically incoherent. See findings/davinci-002-creative-tests.md.

Note that all 12 of those generations were *kept* — none was rerolled — so that
session cannot distinguish an attractor the model could not leave from a
passage the human liked and kept feeding back. Persistence is therefore
something this grid has to measure, not assume: run each firing cell forward
several generations with nothing rejected, and record how many it takes to fall
out of song mode. That number is the actual attractor strength.

That was one accident at T=1.85 / top_p=0.90. This grid turns it into a
measurement. It is built to separate two explanations that predict the same
single observation:

  H1 "verse mode"    — high-entropy sampling pushes the model toward metrical
                       language generally, and `♪` is incidental.
  H2 "caption mode"  — `♪ ... ♪` is the *subtitle convention* for sung lyrics.
                       The model switched into closed-caption transcript
                       register and is reproducing pop lyrics as they appear in
                       subtitle corpora. The metricality is song lyrics', not
                       the model's.

H2 is the better fit for what actually happened: the trigger came directly after
a sign-off ("Peace. I'm outta here.") — exactly where an outro-music caption
goes — the output is pop pastiche including a near-quotation of Madonna, the
model emits empty `♪♪` spans (the caption marker for an instrumental passage),
and span length drifts 37.9 → 49.7 characters across the passage, entering
caption-shaped and stretching toward the surrounding prose.

An earlier line-length argument against H2 is withdrawn: it compared
note-delimited spans to subtitle *display* lines, which are hard-wrapped inside
those spans, and it assumed a blend would inherit a hard constraint from one
parent. Do not reintroduce it. See findings/davinci-002-creative-tests.md.

H2 still is not established — none of this rules out H1, because every cell so
far comes from a single unreplicated event at one sampling setting. That is what
the grid is for.

Run every cell at the setting where the attractor actually fired, then ladder
only the cells that fire.

Usage:
    python examples/song_mode_probe.py                 # print the grid
    python examples/song_mode_probe.py --json          # machine-readable
    python examples/analyze_completion_log.py out.json --by-prompt
"""

from __future__ import annotations

import argparse
import json

# The setting at which the attractor was observed. Hold it fixed for pass 1.
OBSERVED = {"temperature": 1.85, "top_p": 0.90, "max_tokens": 150}

# Hand-driven in a completion console, an ear is the detector: song mode is
# audible in one generation, so cells need only enough output to clear the
# 60-word measurement floor. At max_tokens=150 (~110 words) three generations
# per cell is ample, which puts the whole grid at ~36 generations.
SUGGESTED_N = 3

# Run in two passes so a fragile effect costs almost nothing to rule out.
#
# Pass 0 (2 cells, ~16 generations): signoff_then_note + prose_control. The
#   first replicates the conditions the attractor was actually observed under
#   and is the positive control; the second is the floor. If pass 0 does not
#   produce song mode, stop — there is nothing for the rest of the grid to
#   measure.
# Pass 1 (the rest): only if pass 0 fires. This is where bare_note and
#   music_bracket earn their keep by separating token from convention.
PASS_0 = ("signoff_then_note", "prose_control")

SIGNOFF = ("Well that's my show for the day. As for me, don't bother trying to "
           "connect me on Facebook, this stuff sucks. Peace. I'm outta here.\n")

NEUTRAL = ("The quarterly maintenance report was filed on Tuesday. Three of the "
           "four generators passed inspection without comment.\n")

GRID = [
    # --- A. does the bare marker do it alone? ---
    ("bare_note", "♪",
     "The minimal cell. If song mode fires from a single character with no "
     "context, the marker itself carries the register."),

    ("bare_note_pair", "♪ ",
     "As observed in the wild, lyrics are delimited on both sides. Opening a "
     "pair may be a stronger cue than one loose symbol."),

    # --- B. same convention, different token (H2 predicts these fire too) ---
    ("music_bracket", "[MUSIC PLAYING]\n",
     "Subtitle convention, no ♪. Under H2 this should fire; under H1 it "
     "should not."),

    ("singing_bracket", "[SINGING]\n",
     "As above, and more explicitly vocal."),

    ("note_variant", "♫",
     "The other music glyph. If ♪ fires and ♫ doesn't, the effect is "
     "token-specific rather than convention-specific."),

    ("cue_stack", "[MUSIC PLAYING] ♪",
     "Both cues at once — the ceiling to read bare_note and music_bracket "
     "against. Confounded on its own: it cannot say which cue fired, which is "
     "exactly what those two cells exist to answer. Note it is also mildly "
     "off-distribution — in real captions the bracket cue marks instrumental "
     "music and the note pair marks transcribed lyrics, so they are "
     "alternatives rather than neighbours, and stacking them may weaken the "
     "cue rather than strengthen it."),

    # --- C. context: does the caption slot matter? ---
    ("signoff_then_note", SIGNOFF + "♪",
     "Replicates the observed trigger — a sign-off, then the outro-music "
     "caption. H2's strongest cell; should score highest."),

    ("neutral_then_note", NEUTRAL + "♪",
     "Same marker, wrong slot. If this scores as high as signoff_then_note, "
     "context is irrelevant and the marker is a hard switch."),

    ("signoff_no_note", SIGNOFF,
     "Control for the sign-off itself. Should NOT produce song mode; if it "
     "does, the cue is the ending, not the symbol."),

    # --- D. register control: verse without the caption marker ---
    #     This is the cell that can falsify H2. If it matches the ♪ cells on
    #     metricality, the model has a general verse mode and the caption story
    #     is wrong.
    ("poem_header", "A poem:\n\n",
     "Verse asked for in words, no subtitle convention anywhere."),

    ("bare_iambic", "Shall I compare thee to a summer's day?\n",
     "Verse by example — a metrical line, no marker. The sharpest test of H1."),

    ("prose_control", NEUTRAL,
     "Floor. Whatever metricality this scores is what the sampler gives you "
     "for free at this temperature; every other cell must beat it to mean "
     "anything."),
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true",
                        help="emit the grid as JSON")
    args = parser.parse_args(argv)

    if args.json:
        print(json.dumps({
            "params": OBSERVED,
            "n_per_cell": SUGGESTED_N,
            "pass_0": list(PASS_0),
            "cells": [{"name": n, "prompt": p, "rationale": r,
                       "pass": 0 if n in PASS_0 else 1}
                      for n, p, r in GRID],
        }, indent=1, ensure_ascii=False))
        return 0

    print(f"song-mode probe — {len(GRID)} cells x n={SUGGESTED_N} "
          f"at {OBSERVED}\n")
    print(f"Pass 0 first ({', '.join(PASS_0)}) — if it doesn't fire, stop.\n")
    print("Score with:  python examples/analyze_completion_log.py "
          "out.json --by-prompt")
    print("Read metricality, nPVI and periodicity_strength against the "
          "prose_control row,\nand read every one of them against its own "
          "shuffled baseline.\n")
    for name, prompt, rationale in GRID:
        marker = "  [PASS 0]" if name in PASS_0 else ""
        print(f"--- {name} ---{marker}")
        print(f"    {rationale}")
        print(f"    prompt: {prompt!r}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
