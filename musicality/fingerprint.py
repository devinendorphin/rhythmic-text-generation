"""Tool 10 — RhythmicFingerprint.

Condenses the other nine tools into one fixed-length feature vector — a
rhythmic fingerprint — and compares texts by cosine similarity. This is the
tool for the research questions behind this repo: does LLM output carry the
rhythmic signature of its training text? Does a model asked for "musical"
prose actually move the measurable needles, and which ones? Fingerprint two
corpora and the distance is your answer.
"""

from __future__ import annotations

from math import sqrt

from .alliteration import AlliterationAnalyzer
from .cadence import CadenceAnalyzer
from .euphony import EuphonyScorer
from .meter import MeterDetector
from .rhyme import RhymeAnalyzer
from .sonority import SonorityAnalyzer
from .stress import StressExtractor
from .syllables import SyllableProfiler
from .variability import RhythmVariability

# (feature, source tool, key, normalizer) — normalizers map raw values to a
# roughly 0-1 range so no single feature dominates the cosine.
_FEATURES = [
    ("syllables_per_word", "syllables", "syllables_per_word", lambda v: v / 3),
    ("monosyllabic_ratio", "syllables", "monosyllabic_ratio", lambda v: v),
    ("stress_density", "stress", "stress_density", lambda v: v),
    ("metricality", "meter", "metricality", lambda v: v),
    ("npvi_inter_stress", "variability", "npvi_inter_stress", lambda v: v / 100),
    ("stress_entropy", "variability", "stress_entropy_bits", lambda v: v / 3),
    ("periodicity_strength", "variability", "periodicity_strength", lambda v: max(v, 0.0)),
    ("mean_sonority", "sonority", "mean_sonority", lambda v: v / 10),
    ("sonority_smoothness", "sonority", "smoothness", lambda v: v / 6),
    ("end_rhyme_ratio", "rhyme", "end_rhyme_ratio", lambda v: v),
    ("assonance_density", "rhyme", "assonance_density", lambda v: v),
    ("alliteration_density", "alliteration", "alliteration_density", lambda v: v),
    ("sentence_length_cv", "cadence", "sentence_length_cv", lambda v: min(v, 1.0)),
    ("euphony", "euphony", "euphony_index", lambda v: v / 100),
]


class RhythmicFingerprint:
    name = "fingerprint"

    def __init__(self) -> None:
        self._tools = {
            t.name: t for t in (
                SyllableProfiler(), StressExtractor(), MeterDetector(),
                RhythmVariability(), SonorityAnalyzer(), RhymeAnalyzer(),
                AlliterationAnalyzer(), CadenceAnalyzer(), EuphonyScorer(),
            )
        }

    def analyze(self, text: str) -> dict:
        vector = self.vector(text)
        return {
            "features": {k: round(v, 4) for k, v in vector.items()},
            "dimensions": len(vector),
        }

    def vector(self, text: str) -> dict[str, float]:
        reports = {name: tool.analyze(text) for name, tool in self._tools.items()}
        vector: dict[str, float] = {}
        for feature, tool, key, norm in _FEATURES:
            raw = reports[tool].get(key)
            vector[feature] = float(norm(raw)) if isinstance(raw, (int, float)) else 0.0
        return vector

    def compare(self, text_a: str, text_b: str) -> dict:
        va, vb = self.vector(text_a), self.vector(text_b)
        keys = list(va)
        a = [va[k] for k in keys]
        b = [vb[k] for k in keys]

        dot = sum(x * y for x, y in zip(a, b))
        na, nb = sqrt(sum(x * x for x in a)), sqrt(sum(y * y for y in b))
        cosine = dot / (na * nb) if na and nb else 0.0

        deltas = sorted(
            ({"feature": k, "a": round(va[k], 4), "b": round(vb[k], 4),
              "delta": round(vb[k] - va[k], 4)} for k in keys),
            key=lambda d: abs(d["delta"]), reverse=True,
        )
        return {
            "cosine_similarity": round(cosine, 4),
            "euclidean_distance": round(
                sqrt(sum((x - y) ** 2 for x, y in zip(a, b))), 4
            ),
            # The features that most separate the two texts — usually the
            # interesting part of the comparison.
            "largest_differences": deltas[:5],
            "vector_a": {k: round(v, 4) for k, v in va.items()},
            "vector_b": {k: round(v, 4) for k, v in vb.items()},
        }
