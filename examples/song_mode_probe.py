"""Prompt grid for chasing the davinci-002 song-mode attractor.

Background: in one session davinci-002 emitted a lone `♪` mid-generation and
never left song mode again — 81% of the resulting text, scoring as verse
(metricality 0.724 against Sonnet 18's 0.758) while semantically incoherent.
See findings/davinci-002-creative-tests.md.

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
goes — and the output is pop pastiche, including a near-quotation of Madonna
("baby please don't preach"). But line geometry does *not* support it: the lines
run to 93 characters, past any subtitle cap. So it is a live hypothesis, not a
conclusion, which is what the grid is for.

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

# Suggested pass 1: every cell, n=8, at OBSERVED. ~15 minutes of generations.
SUGGESTED_N = 8

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
            "cells": [{"name": n, "prompt": p, "rationale": r}
                      for n, p, r in GRID],
        }, indent=1, ensure_ascii=False))
        return 0

    print(f"song-mode probe — {len(GRID)} cells x n={SUGGESTED_N} "
          f"at {OBSERVED}\n")
    print("Score with:  python examples/analyze_completion_log.py "
          "out.json --by-prompt")
    print("Read metricality, nPVI and periodicity_strength against the "
          "prose_control row,\nand read every one of them against its own "
          "shuffled baseline.\n")
    for name, prompt, rationale in GRID:
        print(f"--- {name} ---")
        print(f"    {rationale}")
        print(f"    prompt: {prompt!r}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
