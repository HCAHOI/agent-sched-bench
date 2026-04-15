"""BrowseComp benchmark plugin."""

from __future__ import annotations

from typing import Any, ClassVar

from agents.benchmarks._research import (
    ResearchBenchmark,
    _optional_urls,
    _require_text,
)


class BrowseCompBenchmark(ResearchBenchmark):
    """Host-mode plugin for browsing-comprehension QA tasks."""

    slug: ClassVar[str] = "browsecomp"

    def normalize_task(self, raw: dict[str, Any]) -> dict[str, Any]:
        extras = self.config.extras
        id_field = str(extras.get("id_field", "id"))
        question_field = str(extras.get("question_field", "question"))
        answer_field = str(extras.get("answer_field", "answer"))
        urls_field = extras.get("source_urls_field", "source_urls")

        instance_id = _require_text(raw, id_field, label="instance_id")
        problem_statement = _require_text(
            raw,
            question_field,
            label="problem_statement",
        )
        reference_answer = _require_text(raw, answer_field, label="reference_answer")
        return {
            "instance_id": instance_id,
            "problem_statement": problem_statement,
            "reference_answer": reference_answer,
            "source_urls": _optional_urls(
                raw,
                str(urls_field) if urls_field else None,
            ),
            "repo": None,
            "image_name": None,
            "docker_image": None,
        }

