"""Shared phonological core.

Everything in the toolkit rests on this module: tokenization, phoneme lookup
(CMU Pronouncing Dictionary when available, letter-based heuristics otherwise),
syllable counting, lexical stress patterns, and the sonority hierarchy.

Phonemes use ARPABET (the CMUdict alphabet). Vowel phones carry a stress digit:
1 = primary stress, 2 = secondary stress, 0 = unstressed.
"""

from __future__ import annotations

import re
from functools import lru_cache

try:
    import cmudict as _cmudict

    _CMU = _cmudict.dict()
except ImportError:  # pragma: no cover - exercised only without cmudict
    _CMU = {}

WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)*")
SENTENCE_RE = re.compile(r"[.!?]+(?:\s|$)")

# Monosyllabic closed-class words: conventionally unstressed in metrical
# analysis ("promotion" aside), even though CMUdict marks their vowel as
# stressed because every citation-form monosyllable carries stress.
FUNCTION_WORDS = frozenset(
    """a an the and but or nor for so yet as if of at by in on to up
    is am are was be been do did does has had have will would shall should
    can could may might must it its he him his she her we us our they them
    their you your i me my this that than then when while who whom whose
    with from into onto out off not no o'er 'tis""".split()
)

ARPABET_VOWELS = frozenset(
    "AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW".split()
)

# Sonority hierarchy (after Clements 1990 / Parker 2008), scaled 1-10.
# Higher = more sonorous = more open, more "singable".
_SONORITY_CLASSES = [
    (1, "P T K"),            # voiceless stops
    (2, "B D G"),            # voiced stops
    (2, "CH"),               # voiceless affricate
    (3, "JH"),               # voiced affricate
    (3, "F TH S SH HH"),     # voiceless fricatives
    (4, "V DH Z ZH"),        # voiced fricatives
    (5, "M N NG"),           # nasals
    (6, "L R"),              # liquids
    (7, "W Y"),              # glides
    (8, "IY IH UH UW ER"),   # high vowels
    (9, "EH EY OW AO AH"),   # mid vowels
    (10, "AE AA AW AY OY"),  # low vowels / wide diphthongs
]
SONORITY = {
    phone: level for level, phones in _SONORITY_CLASSES for phone in phones.split()
}

_HEURISTIC_VOWEL_GROUPS = re.compile(r"[aeiouy]+", re.IGNORECASE)


def words(text: str) -> list[str]:
    """Lowercased word tokens, apostrophes kept ("don't" stays one token)."""
    return [w.lower() for w in WORD_RE.findall(text)]


def sentences(text: str) -> list[str]:
    """Split on terminal punctuation; drops empty fragments."""
    parts = SENTENCE_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def lines(text: str) -> list[str]:
    """Non-empty lines — the natural unit for verse."""
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def strip_stress(phone: str) -> str:
    """'AH0' -> 'AH'."""
    return phone.rstrip("012")


def is_vowel(phone: str) -> bool:
    return strip_stress(phone) in ARPABET_VOWELS


@lru_cache(maxsize=65536)
def phones_for(word: str) -> tuple[str, ...] | None:
    """First CMUdict pronunciation, or None if the word is out-of-vocabulary."""
    entries = _CMU.get(word.lower())
    if entries:
        return tuple(entries[0])
    return None


def _heuristic_syllables(word: str) -> int:
    """Vowel-group count with a silent-e correction. Used only for OOV words."""
    w = word.lower()
    groups = _HEURISTIC_VOWEL_GROUPS.findall(w)
    count = len(groups)
    if count > 1 and w.endswith("e") and not w.endswith(("le", "ee", "ye")):
        count -= 1
    return max(count, 1)


def syllable_count(word: str) -> int:
    phones = phones_for(word)
    if phones is not None:
        return sum(1 for p in phones if is_vowel(p))
    return _heuristic_syllables(word)


def stress_digits(word: str) -> list[int]:
    """Per-syllable CMUdict stress digits: 1 primary, 2 secondary, 0 none.

    Out-of-vocabulary words get an alternating 1-0-1-0 guess, which follows
    English's trochaic bias and keeps sequences usable rather than silently
    dropping unknown words.
    """
    phones = phones_for(word)
    if phones is not None:
        return [int(p[-1]) for p in phones if p[-1].isdigit()]
    n = _heuristic_syllables(word)
    return [1 if i % 2 == 0 else 0 for i in range(n)]


def metrical_stress(word: str) -> list[int]:
    """Binary stress for metrical scansion: 1 = beat, 0 = offbeat.

    Secondary stress counts as a beat. Monosyllabic function words are
    demoted to offbeats, matching how they behave in actual meter.
    """
    digits = stress_digits(word)
    if len(digits) == 1 and word.lower() in FUNCTION_WORDS:
        return [0]
    return [1 if d in (1, 2) else 0 for d in digits]


def text_stress_sequence(text: str) -> list[int]:
    """Concatenated binary stress sequence for a whole text."""
    seq: list[int] = []
    for w in words(text):
        seq.extend(metrical_stress(w))
    return seq


def sonority_of(phone: str) -> int:
    return SONORITY.get(strip_stress(phone), 0)


def has_dictionary() -> bool:
    """True when CMUdict loaded; tools report degraded accuracy otherwise."""
    return bool(_CMU)


def coverage(text: str) -> float | None:
    """Share of tokens in *this text* found in CMUdict. None if no words.

    ``has_dictionary()`` answers "is CMUdict installed" — a property of the
    install, not of the text. This answers the question that actually
    qualifies a metric: how much of what we just measured was real
    pronunciation, and how much was letter-heuristic guesswork.

    The distinction matters most exactly where the toolkit gets used on
    model output. Every OOV token is scored by ``_heuristic_syllables``,
    which floors at one syllable and hands out an alternating 1-0 stress
    guess. A text of vowelless junk ("ptrfw rpdl hdgnf") therefore scans as
    an unbroken run of stressed monosyllables: stress density 1.0, nPVI 0,
    entropy 0 — the numbers of a perfect pulse, produced entirely by the
    fallback. Read any rhythm metric on a low-coverage text as a statement
    about the fallback until proven otherwise.
    """
    ws = words(text)
    if not ws:
        return None
    return round(sum(1 for w in ws if phones_for(w) is not None) / len(ws), 4)
