"""Tool 4 — RhythmVariability.

Quantifies how *regular* the rhythm is, using measures borrowed from speech
prosody and music research:

* nPVI (normalized Pairwise Variability Index) — the contrast between
  neighbouring durations. Patel & Daniele (2003) used it to show a
  composer's music echoes the rhythm of their native language. Here it is
  applied to text-derived duration proxies (inter-stress intervals and
  syllable weights), so values are comparable within this toolkit rather
  than with acoustic studies.
* Shannon entropy of the stress sequence over n-grams — how predictable the
  next beat is. A metronome has low entropy; noise has high entropy.
  Musical language sits in between.
* Autocorrelation of the stress sequence — periodicity detection. A strong
  peak at lag 2 is the fingerprint of binary meter (iambs/trochees), at
  lag 3 of ternary meter (anapests/dactyls, waltz-time language).
"""

from __future__ import annotations

from collections import Counter
from math import log2
from statistics import mean

from . import phonology


class RhythmVariability:
    name = "variability"

    MAX_LAG = 8
    NGRAM = 3

    def analyze(self, text: str) -> dict:
        seq = phonology.text_stress_sequence(text)
        if len(seq) < 4:
            return {"error": "text too short for variability analysis"}

        positions = [i for i, s in enumerate(seq) if s == 1]
        intervals = [b - a for a, b in zip(positions, positions[1:])]

        weights = [
            len(phones := phonology.phones_for(w) or ()) or phonology.syllable_count(w) * 2
            for w in phonology.words(text)
        ]

        autocorr = {
            lag: round(self._autocorrelation(seq, lag), 4)
            for lag in range(1, min(self.MAX_LAG, len(seq) - 1) + 1)
        }
        dominant_lag = max(autocorr, key=autocorr.get) if autocorr else None

        return {
            "npvi_inter_stress": self._npvi(intervals),
            "npvi_word_weight": self._npvi(weights),
            "stress_entropy_bits": self._ngram_entropy(seq, self.NGRAM),
            "max_entropy_bits": round(log2(2 ** self.NGRAM), 4),
            "autocorrelation": autocorr,
            # Lag 2 -> binary meter, lag 3 -> ternary; the strength of the
            # peak is how "locked in" the pulse is.
            "dominant_period": dominant_lag,
            "periodicity_strength": autocorr.get(dominant_lag) if dominant_lag else None,
            "mean_inter_stress_interval": round(mean(intervals), 4) if intervals else None,
            "dictionary_backed": phonology.has_dictionary(),
            # Coverage of THIS text: low values mean the numbers
            # above describe the OOV fallback, not pronunciation.
            "dictionary_coverage": phonology.coverage(text),
        }

    @staticmethod
    def _npvi(values: list[int]) -> float | None:
        """100 * mean( |d_k - d_{k+1}| / mean(d_k, d_{k+1}) ) over neighbours.

        0 = perfectly even; ~30-40 typical of syllable-timed languages'
        speech, ~60+ of stress-timed ones (acoustic studies; ours is a
        text-based proxy).
        """
        pairs = [
            (a, b) for a, b in zip(values, values[1:]) if a + b > 0
        ]
        if not pairs:
            return None
        return round(
            100 * mean(abs(a - b) / ((a + b) / 2) for a, b in pairs), 4
        )

    @staticmethod
    def _ngram_entropy(seq: list[int], n: int) -> float:
        grams = Counter(tuple(seq[i:i + n]) for i in range(len(seq) - n + 1))
        total = sum(grams.values())
        return round(
            -sum((c / total) * log2(c / total) for c in grams.values()), 4
        )

    @staticmethod
    def _autocorrelation(seq: list[int], lag: int) -> float:
        n = len(seq)
        mu = sum(seq) / n
        var = sum((s - mu) ** 2 for s in seq)
        if var == 0:
            return 0.0
        cov = sum((seq[i] - mu) * (seq[i + lag] - mu) for i in range(n - lag))
        return cov / var
