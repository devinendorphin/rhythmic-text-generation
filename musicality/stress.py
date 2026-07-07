"""Tool 2 — StressExtractor.

Turns text into its stress sequence — the "drum track" underlying the words.
English is a stress-timed language: perceived rhythm comes from the placement
of stressed syllables, so this sequence is the raw material most other tools
in the kit consume. Exposed per word, per line, and as one flat sequence.
"""

from __future__ import annotations

from statistics import mean, pstdev

from . import phonology


class StressExtractor:
    name = "stress"

    def analyze(self, text: str) -> dict:
        ws = phonology.words(text)
        if not ws:
            return {"error": "no words found"}

        per_word = [
            {"word": w, "stress": "".join(map(str, phonology.metrical_stress(w)))}
            for w in ws
        ]
        sequence = [int(c) for item in per_word for c in item["stress"]]

        per_line = []
        for ln in phonology.lines(text):
            seq = phonology.text_stress_sequence(ln)
            per_line.append({"line": ln, "stress": "".join(map(str, seq))})

        stressed = sum(sequence)
        intervals = self._inter_stress_intervals(sequence)

        return {
            "sequence": "".join(map(str, sequence)),
            "syllable_count": len(sequence),
            # ~0.5 suggests strict binary meter; conversational prose runs lower.
            "stress_density": round(stressed / len(sequence), 4) if sequence else 0.0,
            "inter_stress_intervals": intervals,
            "mean_inter_stress_interval": round(mean(intervals), 4) if intervals else None,
            "interval_stdev": round(pstdev(intervals), 4) if len(intervals) > 1 else None,
            "per_line": per_line,
            "per_word": per_word,
            "dictionary_backed": phonology.has_dictionary(),
        }

    @staticmethod
    def _inter_stress_intervals(sequence: list[int]) -> list[int]:
        """Syllable distances between consecutive beats — the tempo map.

        In a perfectly isochronous stress-timed text these are all equal.
        """
        positions = [i for i, s in enumerate(sequence) if s == 1]
        return [b - a for a, b in zip(positions, positions[1:])]
