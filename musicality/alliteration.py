"""Tool 7 — AlliterationAnalyzer.

Alliteration — repeated word onsets in close succession — is the oldest
rhythmic device in Germanic verse (Beowulf is organized by it, not rhyme).
It functions like an accent pattern in percussion: recurring timbre marking
the beat. This tool measures onset repetition within a sliding word window,
on phonemes rather than letters, so "known"/"night" alliterate ("cat"/"city"
do not).
"""

from __future__ import annotations

from collections import Counter

from . import phonology

WINDOW = 4  # words; alliteration is only perceived at close range


class AlliterationAnalyzer:
    name = "alliteration"

    def analyze(self, text: str) -> dict:
        ws = [w for w in phonology.words(text) if w not in phonology.FUNCTION_WORDS]
        if len(ws) < 2:
            return {"error": "not enough content words"}

        onsets = [(w, self._onset(w)) for w in ws]
        events: list[dict] = []
        hits = 0
        for i, (word, onset) in enumerate(onsets):
            if onset is None:
                continue
            window = onsets[max(0, i - WINDOW):i]
            partners = [w for w, o in window if o == onset]
            if partners:
                hits += 1
                events.append({"words": partners + [word], "onset": onset})

        onset_counts = Counter(o for _, o in onsets if o is not None)

        return {
            "content_word_count": len(ws),
            # Fraction of content words echoing an onset within the window.
            # English baseline by chance is nonzero; sustained values well
            # above it mark deliberate sound-patterning.
            "alliteration_density": round(hits / len(ws), 4),
            "event_count": hits,
            "events": events[:40],
            "top_onsets": dict(onset_counts.most_common(8)),
            "window_words": WINDOW,
            "dictionary_backed": phonology.has_dictionary(),
        }

    @staticmethod
    def _onset(word: str) -> str | None:
        """First phoneme of the word; None for vowel-initial words.

        Vowel-initial words traditionally alliterate with any vowel, but
        counting them inflates density on ordinary prose, so they're skipped.
        """
        phones = phonology.phones_for(word)
        if phones is None:
            first = word[0]
            return None if first in "aeiou" else first.upper()
        if phonology.is_vowel(phones[0]):
            return None
        return phonology.strip_stress(phones[0])
