"""Tool 9 — EuphonyScorer.

Phonaesthetics: some sound combinations are systematically judged pleasant
("cellar door"), others harsh. The score rewards the ingredients the
phonaesthetic literature keeps finding — liquids and nasals, open vowels,
smooth consonant-vowel alternation — and penalizes dense obstruent clusters
and sibilant pileups. Output is a 0-100 euphony index plus its components,
so a text can be diagnosed, not just ranked.
"""

from __future__ import annotations

from statistics import mean

from . import phonology

LIQUIDS_NASALS = frozenset("L R M N NG".split())
SIBILANTS = frozenset("S Z SH ZH CH JH".split())
HARSH_STOPS = frozenset("K G T D P B".split())


class EuphonyScorer:
    name = "euphony"

    # Component weights, tuned so canonical "beautiful" test phrases land
    # high and tongue-twisters / legalese land low. Sum of weights = 1.
    WEIGHTS = {
        "vowel_openness": 0.20,
        "liquid_nasal_ratio": 0.25,
        "cluster_penalty": 0.25,
        "sibilant_penalty": 0.10,
        "alternation": 0.20,
    }

    def analyze(self, text: str) -> dict:
        phones: list[str] = []
        for w in phonology.words(text):
            p = phonology.phones_for(w)
            if p:
                phones.extend(phonology.strip_stress(x) for x in p)
        if len(phones) < 5:
            return {"error": "not enough dictionary-covered phonemes"}

        components = {
            "vowel_openness": self._vowel_openness(phones),
            "liquid_nasal_ratio": self._liquid_nasal(phones),
            "cluster_penalty": 1.0 - self._cluster_load(phones),
            "sibilant_penalty": 1.0 - self._sibilant_load(phones),
            "alternation": self._cv_alternation(phones),
        }
        score = sum(self.WEIGHTS[k] * v for k, v in components.items())

        return {
            "euphony_index": round(100 * score, 2),
            "components": {k: round(v, 4) for k, v in components.items()},
            "phoneme_count": len(phones),
            "dictionary_backed": phonology.has_dictionary(),
            # Coverage of THIS text: low values mean the numbers
            # above describe the OOV fallback, not pronunciation.
            "dictionary_coverage": phonology.coverage(text),
        }

    @staticmethod
    def _vowel_openness(phones: list[str]) -> float:
        """Mean sonority of the vowels, rescaled to 0-1 over the 8-10 band."""
        sonorities = [
            phonology.sonority_of(p) for p in phones
            if p in phonology.ARPABET_VOWELS
        ]
        if not sonorities:
            return 0.0
        return (mean(sonorities) - 8) / 2

    @staticmethod
    def _liquid_nasal(phones: list[str]) -> float:
        """Liquids/nasals as a share of consonants, saturating at 50%."""
        consonants = [p for p in phones if p not in phonology.ARPABET_VOWELS]
        if not consonants:
            return 1.0
        ratio = sum(1 for p in consonants if p in LIQUIDS_NASALS) / len(consonants)
        return min(ratio / 0.5, 1.0)

    @staticmethod
    def _cluster_load(phones: list[str]) -> float:
        """Obstruent-cluster harshness: adjacent obstruent pairs per phoneme."""
        harsh_pairs = sum(
            1 for a, b in zip(phones, phones[1:])
            if a in HARSH_STOPS | SIBILANTS and b in HARSH_STOPS | SIBILANTS
        )
        return min(harsh_pairs / (len(phones) * 0.15), 1.0)

    @staticmethod
    def _sibilant_load(phones: list[str]) -> float:
        """Hissiness beyond the English baseline (~10% of phones)."""
        ratio = sum(1 for p in phones if p in SIBILANTS) / len(phones)
        return min(max(ratio - 0.10, 0.0) / 0.15, 1.0)

    @staticmethod
    def _cv_alternation(phones: list[str]) -> float:
        """How often consonants and vowels alternate — the la-la-la factor."""
        flips = sum(
            1 for a, b in zip(phones, phones[1:])
            if (a in phonology.ARPABET_VOWELS) != (b in phonology.ARPABET_VOWELS)
        )
        return flips / (len(phones) - 1)
