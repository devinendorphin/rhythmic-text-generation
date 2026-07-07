"""Tool 1 — SyllableProfiler.

The syllable is the basic rhythmic pulse of speech. This tool measures how
that pulse is distributed: syllables per word, the monosyllable/polysyllable
balance, and the evenness of the syllabic texture. Texts built from short,
even words (nursery rhymes, chants) profile very differently from
Latinate academic prose, and that difference is a rhythmic signature.
"""

from __future__ import annotations

from collections import Counter
from statistics import mean, pstdev

from . import phonology


class SyllableProfiler:
    name = "syllables"

    def analyze(self, text: str) -> dict:
        ws = phonology.words(text)
        if not ws:
            return {"error": "no words found"}

        counts = [phonology.syllable_count(w) for w in ws]
        total = sum(counts)
        distribution = Counter(counts)

        return {
            "word_count": len(ws),
            "syllable_count": total,
            "syllables_per_word": round(mean(counts), 4),
            "syllables_per_word_stdev": round(pstdev(counts), 4),
            # Fraction of one-syllable words: high values read as punchy,
            # beat-like text; English verse tends to sit above prose here.
            "monosyllabic_ratio": round(distribution[1] / len(ws), 4),
            "polysyllabic_ratio": round(
                sum(v for k, v in distribution.items() if k >= 3) / len(ws), 4
            ),
            "max_word_syllables": max(counts),
            "distribution": {str(k): v for k, v in sorted(distribution.items())},
            "dictionary_backed": phonology.has_dictionary(),
        }
