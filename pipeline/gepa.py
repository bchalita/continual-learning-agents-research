"""
GEPA-style multi-document prompt optimization with Pareto frontier.

Implements the core ideas from GEPA (ICLR 2026):
1. Prompt pool — maintains a growing population of candidate prompts
2. Pareto frontier — keeps non-dominated prompts (no other prompt beats
   it on ALL documents)
3. Win-frequency selection — prompts that "win" (score best) on more
   documents are more likely to be selected as parents for mutation
4. Reflective mutation — propose new prompts by reflecting on the
   worst-performing document for the selected parent

Key difference from Erwin's single-doc loop:
- Evaluates every prompt on ALL documents, not just one
- Maintains a diverse pool instead of linear prompt replacement
- Selection pressure via Pareto dominance, not greedy best-so-far
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PromptCandidate:
    """A candidate prompt with its per-document evaluation scores."""

    prompt_id: str
    text: str
    parent_id: str | None
    iteration: int
    scores: dict[str, float] = field(default_factory=dict)         # doc_id → score
    subscores: dict[str, dict] = field(default_factory=dict)       # doc_id → {structure, numbers, text}
    eval_results: dict[str, dict] = field(default_factory=dict)    # doc_id → full eval_result

    @property
    def mean_score(self) -> float:
        if not self.scores:
            return 0.0
        return sum(self.scores.values()) / len(self.scores)

    @property
    def min_score(self) -> float:
        if not self.scores:
            return 0.0
        return min(self.scores.values())

    def worst_doc(self) -> str | None:
        """Return the doc_id where this prompt scores lowest."""
        if not self.scores:
            return None
        return min(self.scores, key=self.scores.get)

    def bottom_docs(self, n: int = 2) -> list[str]:
        """Return the n worst-scoring doc_ids, sorted worst-first."""
        if not self.scores:
            return []
        sorted_docs = sorted(self.scores, key=self.scores.get)
        return sorted_docs[:n]

    def dominates(self, other: PromptCandidate) -> bool:
        """True if self >= other on ALL docs and self > other on at least one.

        Both must have been evaluated on the same set of documents.
        """
        if not self.scores or not other.scores:
            return False
        common_docs = set(self.scores) & set(other.scores)
        if not common_docs:
            return False
        at_least_one_better = False
        for doc_id in common_docs:
            if self.scores[doc_id] < other.scores[doc_id]:
                return False
            if self.scores[doc_id] > other.scores[doc_id]:
                at_least_one_better = True
        return at_least_one_better

    def to_dict(self) -> dict:
        return {
            "prompt_id": self.prompt_id,
            "parent_id": self.parent_id,
            "iteration": self.iteration,
            "mean_score": self.mean_score,
            "min_score": self.min_score,
            "scores": self.scores,
            "subscores": self.subscores,
            "text": self.text,
        }


class PromptPool:
    """Population of prompt candidates with Pareto frontier tracking."""

    def __init__(self) -> None:
        self.candidates: list[PromptCandidate] = []
        self._next_id = 0

    def _gen_id(self) -> str:
        pid = f"p{self._next_id:03d}"
        self._next_id += 1
        return pid

    def add(
        self,
        text: str,
        parent_id: str | None = None,
        iteration: int = 0,
    ) -> PromptCandidate:
        """Add a new (unscored) prompt to the pool."""
        candidate = PromptCandidate(
            prompt_id=self._gen_id(),
            text=text,
            parent_id=parent_id,
            iteration=iteration,
        )
        self.candidates.append(candidate)
        return candidate

    @property
    def pareto_front(self) -> list[PromptCandidate]:
        """Return the set of non-dominated candidates.

        A candidate is non-dominated if no other candidate dominates it
        (i.e., no other is >= on all docs and > on at least one).
        """
        scored = [c for c in self.candidates if c.scores]
        if not scored:
            return []
        front = []
        for c in scored:
            dominated = any(other.dominates(c) for other in scored if other is not c)
            if not dominated:
                front.append(c)
        return front

    def win_frequencies(self) -> dict[str, int]:
        """Count how many documents each prompt "wins" (scores highest on).

        For each document, the prompt with the highest score gets +1.
        Ties split the win.
        """
        scored = [c for c in self.candidates if c.scores]
        if not scored:
            return {}

        # Gather all doc_ids
        all_docs = set()
        for c in scored:
            all_docs.update(c.scores.keys())

        wins: dict[str, int] = {c.prompt_id: 0 for c in scored}
        for doc_id in all_docs:
            # Find the max score for this doc
            candidates_with_doc = [(c, c.scores[doc_id]) for c in scored if doc_id in c.scores]
            if not candidates_with_doc:
                continue
            max_score = max(s for _, s in candidates_with_doc)
            winners = [c for c, s in candidates_with_doc if s == max_score]
            for w in winners:
                wins[w.prompt_id] += 1

        return wins

    def select_parent(self) -> PromptCandidate:
        """Select a parent prompt for mutation via win-frequency weighted sampling.

        Prompts that win on more documents are more likely to be selected.
        Adds a small epsilon so even zero-win prompts have some chance
        (exploration).
        """
        front = self.pareto_front
        if not front:
            # No scored prompts yet — return the most recent
            return self.candidates[-1]

        wins = self.win_frequencies()
        epsilon = 0.5  # Exploration bonus
        weights = [wins.get(c.prompt_id, 0) + epsilon for c in front]

        return random.choices(front, weights=weights, k=1)[0]

    @property
    def best_prompt(self) -> PromptCandidate | None:
        """Return the prompt with the highest mean score."""
        scored = [c for c in self.candidates if c.scores]
        if not scored:
            return None
        return max(scored, key=lambda c: c.mean_score)

    def summary(self) -> dict[str, Any]:
        """Summary statistics for logging."""
        front = self.pareto_front
        wins = self.win_frequencies()
        best = self.best_prompt
        return {
            "pool_size": len(self.candidates),
            "scored": len([c for c in self.candidates if c.scores]),
            "pareto_front_size": len(front),
            "pareto_front_ids": [c.prompt_id for c in front],
            "win_frequencies": wins,
            "best_prompt_id": best.prompt_id if best else None,
            "best_mean_score": best.mean_score if best else None,
        }

    def to_json(self) -> str:
        """Serialize pool to JSON for saving."""
        return json.dumps(
            {
                "candidates": [c.to_dict() for c in self.candidates],
                "summary": self.summary(),
            },
            indent=2,
        )
