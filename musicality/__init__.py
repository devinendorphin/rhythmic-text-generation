"""musicality — a toolkit for quantifying rhythm and musicality embedded in language.

Ten tools, each a class with a uniform ``analyze(text) -> dict`` interface:

 1. SyllableProfiler      — syllable structure and distribution
 2. StressExtractor       — lexical stress sequences (the text's "drum track")
 3. MeterDetector         — best-fit metrical foot and metricality score
 4. RhythmVariability     — nPVI, inter-stress intervals, rhythm entropy
 5. SonorityAnalyzer      — sonority contours (the text's "melodic line")
 6. RhymeAnalyzer         — end rhyme, internal rhyme, assonance, consonance
 7. AlliterationAnalyzer  — onset repetition density
 8. CadenceAnalyzer       — sentence/phrase-level periodicity (prose cadence)
 9. EuphonyScorer         — phonaesthetic smoothness vs. harshness
10. RhythmicFingerprint   — aggregate feature vector + text-to-text comparison
"""

from .syllables import SyllableProfiler
from .stress import StressExtractor
from .meter import MeterDetector
from .variability import RhythmVariability
from .sonority import SonorityAnalyzer
from .rhyme import RhymeAnalyzer
from .alliteration import AlliterationAnalyzer
from .cadence import CadenceAnalyzer
from .euphony import EuphonyScorer
from .fingerprint import RhythmicFingerprint

ALL_TOOLS = {
    "syllables": SyllableProfiler,
    "stress": StressExtractor,
    "meter": MeterDetector,
    "variability": RhythmVariability,
    "sonority": SonorityAnalyzer,
    "rhyme": RhymeAnalyzer,
    "alliteration": AlliterationAnalyzer,
    "cadence": CadenceAnalyzer,
    "euphony": EuphonyScorer,
    "fingerprint": RhythmicFingerprint,
}

__all__ = [cls.__name__ for cls in ALL_TOOLS.values()] + ["ALL_TOOLS"]
__version__ = "0.1.0"
