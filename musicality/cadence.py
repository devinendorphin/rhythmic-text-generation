"""Tool 8 — CadenceAnalyzer.

Rhythm above the syllable: the rise and fall of sentence and clause lengths.
This is the level writers mean by "cadence" — long sentence, long sentence,
short punch. The tool treats the sequence of sentence lengths (in syllables)
as a signal and looks for periodicity in it with autocorrelation and a
discrete Fourier transform, plus burstiness of clause lengths between
punctuation marks. Strong low-frequency structure here is the prose
equivalent of phrasing in music.
"""

from __future__ import annotations

import cmath
import re
from statistics import mean, pstdev

from . import phonology

CLAUSE_RE = re.compile(r"[,;:.!?—]+")


class CadenceAnalyzer:
    name = "cadence"

    def analyze(self, text: str) -> dict:
        sents = phonology.sentences(text)
        if not sents:
            return {"error": "no sentences found"}

        lengths = [
            sum(phonology.syllable_count(w) for w in phonology.words(s))
            for s in sents
        ]
        lengths = [n for n in lengths if n > 0]
        if not lengths:
            return {"error": "no measurable sentences"}

        clause_lengths = [
            len(phonology.words(c))
            for c in CLAUSE_RE.split(text)
            if phonology.words(c)
        ]

        result = {
            "sentence_count": len(lengths),
            "sentence_syllables": lengths,
            "mean_sentence_syllables": round(mean(lengths), 4),
            # Coefficient of variation: 0 = metronomic uniformity; high
            # values = dramatic long/short contrast. Both extremes are
            # rhythmic choices; the middle is unmarked prose.
            "sentence_length_cv": round(pstdev(lengths) / mean(lengths), 4)
            if len(lengths) > 1 and mean(lengths) > 0 else 0.0,
            "clause_count": len(clause_lengths),
            "mean_clause_words": round(mean(clause_lengths), 4)
            if clause_lengths else None,
            "clause_length_cv": round(pstdev(clause_lengths) / mean(clause_lengths), 4)
            if len(clause_lengths) > 1 and mean(clause_lengths) > 0 else None,
        }

        if len(lengths) >= 4:
            result["autocorrelation_lag1"] = round(self._autocorr(lengths, 1), 4)
            period, strength = self._dominant_period(lengths)
            # e.g. period 2.0 = alternating long/short sentences.
            result["dominant_sentence_period"] = period
            result["periodicity_strength"] = strength
        return result

    @staticmethod
    def _autocorr(series: list[int], lag: int) -> float:
        n = len(series)
        mu = mean(series)
        var = sum((x - mu) ** 2 for x in series)
        if var == 0:
            return 0.0
        return sum(
            (series[i] - mu) * (series[i + lag] - mu) for i in range(n - lag)
        ) / var

    @staticmethod
    def _dominant_period(series: list[int]) -> tuple[float | None, float | None]:
        """DFT of the mean-removed length series; returns (period, relative power)."""
        n = len(series)
        mu = mean(series)
        centered = [x - mu for x in series]
        total_power = sum(x * x for x in centered)
        if total_power == 0:
            return None, None
        best_k, best_power = None, 0.0
        for k in range(1, n // 2 + 1):
            coeff = sum(
                centered[t] * cmath.exp(-2j * cmath.pi * k * t / n)
                for t in range(n)
            )
            power = abs(coeff) ** 2
            if power > best_power:
                best_k, best_power = k, power
        if best_k is None:
            return None, None
        return round(n / best_k, 2), round(best_power / (total_power * n / 2), 4)
