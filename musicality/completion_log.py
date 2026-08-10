"""Ingest for completion-log exports — batches of generations with parameters.

The generative side of this work (see the sibling ``cobralingus``) exports
sessions as JSON: a model name, the story-so-far, and a list of generations
each carrying its sampling parameters. That shape is more informative than a
flat extract, because the parameters are an *independent variable*. A log with
a temperature sweep in it lets you ask what sampling does to rhythm, rather
than only describing one sample after the fact.

    log = CompletionLog.load("session.json")
    for temp, gens in log.grouped_by("temperature").items():
        ...

Nothing here computes rhythm — it turns a log into text you can hand to the
ten tools, and keeps each generation's parameters attached on the way.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field
from typing import Any, Iterator

# Sampling parameters worth grouping by. Others (n, stop, streamed) are
# bookkeeping rather than knobs that could plausibly move phonology.
SWEEPABLE = ("temperature", "top_p", "frequency_penalty", "presence_penalty",
             "max_tokens")


@dataclass(frozen=True)
class Generation:
    """One completion, with the parameters that produced it."""

    output: str
    prompt: str = ""
    model: str = ""
    ts: str = ""
    params: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        """Read a sampling parameter, e.g. ``gen["temperature"]``."""
        return self.params.get(key)

    @property
    def words(self) -> int:
        from . import phonology
        return len(phonology.words(self.output))


@dataclass
class CompletionLog:
    """A parsed export: many generations sharing a model and a session."""

    generations: list[Generation]
    model: str = ""
    exported: str = ""
    story: str = ""
    source: str = ""

    @classmethod
    def load(cls, path: str | pathlib.Path) -> "CompletionLog":
        path = pathlib.Path(path)
        return cls.from_dict(json.loads(path.read_text()), source=path.name)

    @classmethod
    def from_dict(cls, data: dict, source: str = "") -> "CompletionLog":
        if not isinstance(data, dict) or "generations" not in data:
            raise ValueError(
                "not a completion log: expected a dict with a 'generations' key"
            )
        model = data.get("model", "")
        gens = [
            Generation(
                output=g.get("output", ""),
                prompt=g.get("prompt", ""),
                # Per-generation model wins; logs may mix models in one session.
                model=g.get("model", model),
                ts=g.get("ts", ""),
                params=g.get("params") or {},
            )
            for g in data["generations"]
        ]
        return cls(
            generations=gens,
            model=model,
            exported=data.get("exported", ""),
            story=data.get("story", ""),
            source=source,
        )

    @classmethod
    def load_all(cls, *paths: str | pathlib.Path) -> "CompletionLog":
        """Merge several exports into one log (keeps every generation)."""
        logs = [cls.load(p) for p in paths]
        models = {lg.model for lg in logs if lg.model}
        return cls(
            generations=[g for lg in logs for g in lg.generations],
            model=models.pop() if len(models) == 1 else "+".join(sorted(models)),
            exported=max((lg.exported for lg in logs), default=""),
            source=", ".join(lg.source for lg in logs),
        )

    def __len__(self) -> int:
        return len(self.generations)

    def __iter__(self) -> Iterator[Generation]:
        return iter(self.generations)

    def usable(self, min_words: int = 20) -> "CompletionLog":
        """Drop generations too short to carry a stable rhythm measurement.

        Metrics on a handful of words are noise — nPVI needs a run of beats.
        Empty outputs (a refused or truncated call) land here too.
        """
        return CompletionLog(
            generations=[g for g in self.generations if g.words >= min_words],
            model=self.model, exported=self.exported, story=self.story,
            source=self.source,
        )

    def grouped_by(self, param: str) -> dict[Any, list[Generation]]:
        """Bucket generations by a sampling parameter, ordered by its value."""
        buckets: dict[Any, list[Generation]] = {}
        for g in self.generations:
            buckets.setdefault(g[param], []).append(g)
        return dict(
            sorted(buckets.items(), key=lambda kv: (kv[0] is None, kv[0]))
        )

    def swept(self) -> list[str]:
        """Parameters that actually vary here — the log's independent variables."""
        return [
            p for p in SWEEPABLE
            if len({repr(g[p]) for g in self.generations}) > 1
        ]

    def text(self, joiner: str = "\n\n") -> str:
        """All outputs as one document, for whole-corpus analysis."""
        return joiner.join(g.output for g in self.generations)
