"""Tool 5 — SonorityAnalyzer.

Maps text onto the sonority hierarchy (stops < fricatives < nasals < liquids
< glides < vowels) and studies the resulting contour — the closest thing
language has to a melodic line. Speech naturally rises and falls in sonority
once per syllable; texts differ in how smooth, how deep, and how wave-like
those oscillations are. High mean sonority with gentle transitions is the
acoustic profile of "mellifluous" language; dense low-sonority clusters
("strengths", "crisp") make text percussive instead.
"""

from __future__ import annotations

from statistics import mean, pstdev

from . import phonology


class SonorityAnalyzer:
    name = "sonority"

    def analyze(self, text: str) -> dict:
        contour: list[int] = []
        for w in phonology.words(text):
            phones = phonology.phones_for(w)
            if phones is None:
                continue
            contour.extend(phonology.sonority_of(p) for p in phones)
        if len(contour) < 3:
            return {"error": "not enough dictionary-covered phonemes"}

        steps = [b - a for a, b in zip(contour, contour[1:])]
        direction_changes = sum(
            1 for a, b in zip(steps, steps[1:]) if a * b < 0
        )

        return {
            "phoneme_count": len(contour),
            # 1-10 scale; vowel-heavy open texts trend high.
            "mean_sonority": round(mean(contour), 4),
            "sonority_stdev": round(pstdev(contour), 4),
            # Mean absolute step between adjacent phonemes: small steps =
            # smooth, legato transitions; large = jagged, staccato.
            "smoothness": round(mean(abs(s) for s in steps), 4),
            # Peaks-and-valleys per phoneme. Speech oscillates by design;
            # values near the syllable rate indicate clean CV alternation.
            "oscillation_rate": round(direction_changes / len(contour), 4),
            "vowel_ratio": round(
                sum(1 for s in contour if s >= 8) / len(contour), 4
            ),
            "sonorant_consonant_ratio": round(
                sum(1 for s in contour if 5 <= s <= 7) / len(contour), 4
            ),
            "obstruent_ratio": round(
                sum(1 for s in contour if 0 < s <= 4) / len(contour), 4
            ),
            "contour_preview": contour[:60],
            "dictionary_backed": phonology.has_dictionary(),
            # Coverage of THIS text: low values mean the numbers
            # above describe the OOV fallback, not pronunciation.
            "dictionary_coverage": phonology.coverage(text),
        }
