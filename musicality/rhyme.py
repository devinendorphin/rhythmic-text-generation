"""Tool 6 — RhymeAnalyzer.

Rhyme is pitch-free harmony: recurring sound patterns that bind lines
together the way chord repetition binds a song. This tool detects

* end-rhyme and the rhyme scheme (ABAB, AABB, ...),
* internal rhyme (rhyming words inside the same line),
* assonance (repeated vowel sounds in close proximity), and
* consonance (repeated consonant sounds in close proximity),

all computed on phonemes, not spelling, so "enough" rhymes with "stuff".
"""

from __future__ import annotations

from collections import Counter
from string import ascii_uppercase

from . import phonology

WINDOW = 12  # phoneme window for assonance/consonance proximity


class RhymeAnalyzer:
    name = "rhyme"

    def analyze(self, text: str) -> dict:
        lns = phonology.lines(text)
        if not lns:
            return {"error": "no lines found"}

        scheme, rhyme_pairs = self._end_rhyme_scheme(lns)
        internal = self._internal_rhymes(lns)
        assonance, consonance = self._echo_densities(text)

        rhymed_lines = sum(
            1 for letter in scheme if letter != "-" and scheme.count(letter) > 1
        )

        return {
            "rhyme_scheme": "".join(scheme),
            "end_rhyme_ratio": round(rhymed_lines / len(lns), 4),
            "rhyme_pairs": rhyme_pairs,
            "internal_rhymes": internal,
            "internal_rhyme_count": len(internal),
            # Echo densities: repeated-sound events per phoneme within the
            # proximity window. Poetry and song lyrics run measurably hotter
            # than plain prose.
            "assonance_density": assonance,
            "consonance_density": consonance,
            "dictionary_backed": phonology.has_dictionary(),
            # Coverage of THIS text: low values mean the numbers
            # above describe the OOV fallback, not pronunciation.
            "dictionary_coverage": phonology.coverage(text),
        }

    @staticmethod
    def rhyme_part(word: str) -> tuple[str, ...] | None:
        """Phones from the last primary-stressed vowel onward.

        Two words sharing this tail are a perfect rhyme (moon/June: UW1 N).
        Out-of-vocabulary words fall back to their longest in-dictionary
        suffix ("untrimmed" -> "trimmed"), which is sound because the rhyme
        part depends only on the word's tail.
        """
        phones = phonology.phones_for(word)
        if phones is None:
            for i in range(1, len(word) - 2):
                phones = phonology.phones_for(word[i:])
                if phones is not None:
                    break
        if phones is None:
            return None
        last_primary = None
        for i, p in enumerate(phones):
            if p.endswith("1"):
                last_primary = i
        if last_primary is None:  # no primary stress: fall back to last vowel
            for i, p in enumerate(phones):
                if phonology.is_vowel(p):
                    last_primary = i
        if last_primary is None:
            return None
        return tuple(phonology.strip_stress(p) for p in phones[last_primary:])

    def _end_rhyme_scheme(self, lns: list[str]) -> tuple[list[str], list[dict]]:
        finals = []
        for ln in lns:
            ws = phonology.words(ln)
            finals.append(ws[-1] if ws else None)

        parts = [self.rhyme_part(w) if w else None for w in finals]
        scheme: list[str] = []
        letter_for: dict[tuple[str, ...], str] = {}
        pairs: list[dict] = []
        next_letter = 0
        for i, part in enumerate(parts):
            if part is None:
                scheme.append("-")
                continue
            if part in letter_for:
                scheme.append(letter_for[part])
                first = parts.index(part)
                pairs.append({
                    "words": [finals[first], finals[i]],
                    "lines": [first + 1, i + 1],
                })
            else:
                letter = ascii_uppercase[next_letter % 26]
                letter_for[part] = letter
                scheme.append(letter)
                next_letter += 1
        return scheme, pairs

    def _internal_rhymes(self, lns: list[str]) -> list[dict]:
        found = []
        for idx, ln in enumerate(lns, start=1):
            ws = [w for w in phonology.words(ln) if phonology.syllable_count(w) >= 1]
            seen: dict[tuple[str, ...], str] = {}
            for w in ws:
                part = self.rhyme_part(w)
                if part is None or len(part) < 2:  # vowel+coda minimum
                    continue
                if part in seen and seen[part] != w:
                    found.append({"line": idx, "words": [seen[part], w]})
                seen[part] = w
        return found

    @staticmethod
    def _echo_densities(text: str) -> tuple[float, float]:
        vowels: list[str] = []
        consonants: list[str] = []
        stream: list[tuple[str, bool]] = []  # (phone, is_vowel), in order
        for w in phonology.words(text):
            phones = phonology.phones_for(w)
            if phones is None:
                continue
            for p in phones:
                bare = phonology.strip_stress(p)
                stream.append((bare, phonology.is_vowel(p)))

        if len(stream) < 2:
            return 0.0, 0.0

        def density(target_vowel: bool) -> float:
            echoes = 0
            for i, (phone, isv) in enumerate(stream):
                if isv != target_vowel:
                    continue
                window = stream[max(0, i - WINDOW):i]
                if any(p == phone and v == target_vowel for p, v in window):
                    echoes += 1
            pool = sum(1 for _, v in stream if v == target_vowel)
            return round(echoes / pool, 4) if pool else 0.0

        return density(True), density(False)
