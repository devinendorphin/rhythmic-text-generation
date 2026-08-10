"""Tool 3 — MeterDetector.

Scores a text's stress sequence against the classical metrical feet (iamb,
trochee, anapest, dactyl, amphibrach) and reports the best-fit meter plus a
"metricality" score: how much of the text actually lands on the template.
High metricality with a clear winner means the text has a poetic meter;
prose typically scores in a muddled middle with no dominant foot.
"""

from __future__ import annotations

from statistics import mean

from . import phonology

FEET = {
    "iamb": (0, 1),
    "trochee": (1, 0),
    "anapest": (0, 0, 1),
    "dactyl": (1, 0, 0),
    "amphibrach": (0, 1, 0),
}

METER_NAMES = {
    1: "monometer", 2: "dimeter", 3: "trimeter", 4: "tetrameter",
    5: "pentameter", 6: "hexameter", 7: "heptameter", 8: "octameter",
}


class MeterDetector:
    name = "meter"

    def analyze(self, text: str) -> dict:
        lns = phonology.lines(text)
        if not lns:
            return {"error": "no lines found"}

        line_results = []
        for ln in lns:
            seq = phonology.text_stress_sequence(ln)
            if seq:
                line_results.append(self._scan_line(ln, seq))
        if not line_results:
            return {"error": "no scannable lines"}

        foot_scores = {
            foot: round(mean(r["scores"][foot] for r in line_results), 4)
            for foot in FEET
        }
        best_foot = max(foot_scores, key=foot_scores.get)
        metricality = foot_scores[best_foot]

        # Modal line length in best-foot units gives the classical meter name.
        foot_len = len(FEET[best_foot])
        feet_counts = [round(len(r["stress"]) / foot_len) for r in line_results]
        modal_feet = max(set(feet_counts), key=feet_counts.count) if feet_counts else 0

        return {
            "best_foot": best_foot,
            "meter_name": f"{best_foot}ic {METER_NAMES.get(modal_feet, f'{modal_feet}-foot')}",
            # 1.0 = every syllable matches the template; >= ~0.8 reads as
            # clearly metrical, <= ~0.6 as prose-like.
            "metricality": metricality,
            "foot_scores": foot_scores,
            "lines": line_results,
            "dictionary_backed": phonology.has_dictionary(),
            # Coverage of THIS text: low values mean the numbers
            # above describe the OOV fallback, not pronunciation.
            "dictionary_coverage": phonology.coverage(text),
        }

    def _scan_line(self, line: str, seq: list[int]) -> dict:
        scores = {foot: self._match(seq, template) for foot, template in FEET.items()}
        best = max(scores, key=scores.get)
        return {
            "line": line,
            "stress": "".join(map(str, seq)),
            "syllables": len(seq),
            "best_foot": best,
            "scores": {k: round(v, 4) for k, v in scores.items()},
        }

    @staticmethod
    def _match(seq: list[int], template: tuple[int, ...]) -> float:
        """Fraction of syllables agreeing with the repeated foot template.

        The template is tried at every phase offset (a headless iambic line
        scans as a trochaic phase shift) and the best alignment wins.
        """
        n = len(seq)
        if n == 0:
            return 0.0
        best = 0
        for offset in range(len(template)):
            hits = sum(
                1 for i, s in enumerate(seq)
                if s == template[(i + offset) % len(template)]
            )
            best = max(best, hits)
        return best / n
