"""Measure how far a passage has fallen into a specific discourse register.

Built for one observation (see findings/register-attractors.md): the seed
"Three Salafi lesbians walk into a bar," put davinci-002 not into a joke but
into the comment section of a 2005-2016 counter-jihad blog, and it stayed there.

**Method warning, which is the whole point of this file.** The marker lists were
written *after* reading the passage they describe. Finding them there proves
nothing whatsoever — it is circular. Two things make the measurement mean
something, and both must be kept:

1. **The topical/register split.** Words like "muslim" or "feminist" appear in
   any discussion of the subject, neutral or hostile, so counting them measures
   only what the text is *about*. Only markers that essentially cannot occur
   outside the specific discourse ("raghead", "sex apartheid", "cultural civil
   war") are evidence of register. Keep the lists separate and report both;
   if the gap between a suspect passage and a control is carried by the topical
   list, there is no finding.

2. **Controls the list was not built from.** Run it unchanged over unrelated
   generations from the same model and human. A register rate near zero there,
   against a substantial rate in the suspect passage, is the actual result.

Adding a marker because you just saw it in the passage under test invalidates
the measurement for that passage. Add it, then re-run against fresh controls.

Usage:
    python examples/register_markers.py file.txt [control.txt ...]
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

# Appears in any treatment of the subject — hostile, neutral or academic.
# Counting these measures topic, not stance.
TOPICAL = [
    r"\bmuslims?\b", r"\bislam\b", r"\bislamic\b", r"\bsalafi", r"\bsharia",
    r"\bjihad", r"\blesbians?\b", r"\bfeminis[tm]", r"\bgay\b", r"\bqueer\b",
]

# Essentially confined to the specific discourse. A neutral essay on Salafism
# and LGBT politics contains none of these.
REGISTER = [
    r"raghead", r"\bgoys?\b", r"\bsjws?\b", r"social justice warriors?",
    r"sex apartheid", r"gender apartheid", r"cultural civil war",
    r"tearing apart the west", r"our culture and values", r"aryan nation",
    r"white geno", r"fake lesbians?", r"\bcommies?\b", r"the west is losing",
    r"moonbat", r"\bcuck", r"red ?pill",
]


def rate_per_1k(text: str, patterns: list[str]) -> tuple[float, int]:
    """Marker hits per 1,000 words, and the word count."""
    lowered = text.lower()
    words = len(lowered.split())
    hits = sum(len(re.findall(p, lowered)) for p in patterns)
    return (1000 * hits / words if words else 0.0), words


def hits_in_context(text: str, patterns: list[str], window: int = 55) -> list[str]:
    """Every register hit with surrounding text, so claims stay auditable."""
    found = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.I):
            start = max(0, match.start() - window)
            found.append(text[start:match.end() + window].replace("\n", " "))
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("target", type=pathlib.Path)
    parser.add_argument("controls", nargs="*", type=pathlib.Path,
                        help="unrelated text the marker lists were NOT built "
                             "from; without these the numbers mean nothing")
    parser.add_argument("--quotes", action="store_true",
                        help="print each register hit in context")
    args = parser.parse_args(argv)

    if not args.controls:
        print("warning: no controls given — a register rate on its own is not "
              "evidence, because the lists were written from a known passage.\n",
              file=sys.stderr)

    print(f"{'file':<34}{'words':>8}{'topical/1k':>12}{'REGISTER/1k':>13}")
    for path in [args.target, *args.controls]:
        text = path.read_text()
        topical, words = rate_per_1k(text, TOPICAL)
        register, _ = rate_per_1k(text, REGISTER)
        print(f"{path.name[:34]:<34}{words:>8}{topical:>12.2f}{register:>13.2f}")

    if args.quotes:
        print("\nregister hits in target, in context:")
        for quote in hits_in_context(args.target.read_text(), REGISTER):
            print(f"  ...{quote}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
