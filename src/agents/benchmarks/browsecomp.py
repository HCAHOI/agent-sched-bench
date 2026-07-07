"""BrowseComp benchmark plugin.

The official BrowseComp CSV stores encrypted questions and reference answers.
This plugin pins the raw CSV bytes by SHA-256, decrypts rows only after source
validation, and keeps reference answers out of agent-visible fields.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import io
from pathlib import Path
from typing import Any, ClassVar, Sequence

import httpx

from agents.benchmarks.base import Benchmark


class BrowseCompBenchmark(Benchmark):
    """Benchmark plugin for OpenAI BrowseComp."""

    slug: ClassVar[str] = "browsecomp"
    SUPPORTED_SCAFFOLDS: ClassVar[set[str]] = {"deep-research"}

    _REQUIRED_EXTRAS: ClassVar[tuple[str, ...]] = (
        "task_source_kind",
        "csv_url",
        "csv_sha256",
        "expected_schema",
        "expected_row_count",
        "id_field",
        "question_field",
        "answer_field",
        "topic_field",
        "canary_field",
        "scorer_template",
        "scorer_provider",
        "scorer_model",
        "web_search_provider",
        "web_fetch_provider",
        "max_search_calls",
        "max_fetch_calls",
    )

    @property
    def execution_environment(self) -> str:
        return "host"

    def validate_config(self) -> None:
        extras = self.config.extras
        missing = [key for key in self._REQUIRED_EXTRAS if key not in extras]
        if missing:
            raise ValueError(f"browsecomp config missing extras: {', '.join(missing)}")
        if self.config.slug != self.slug:
            raise ValueError(f"browsecomp config slug must be {self.slug!r}")
        expected_schema = extras["expected_schema"]
        if not isinstance(expected_schema, Sequence) or isinstance(
            expected_schema, (str, bytes)
        ):
            raise ValueError("browsecomp extras.expected_schema must be a sequence")
        if [str(item) for item in expected_schema] != [
            "problem",
            "answer",
            "problem_topic",
            "canary",
        ]:
            raise ValueError("browsecomp extras.expected_schema does not match official CSV")
        if int(extras["expected_row_count"]) <= 0:
            raise ValueError("browsecomp extras.expected_row_count must be positive")
        if int(extras["max_search_calls"]) <= 0:
            raise ValueError("browsecomp extras.max_search_calls must be positive")
        if int(extras["max_fetch_calls"]) <= 0:
            raise ValueError("browsecomp extras.max_fetch_calls must be positive")
        if float(extras.get("scorer_temperature", 0.0)) != 0.0:
            raise ValueError("browsecomp scorer_temperature must be 0.0")

    def load_tasks(self) -> list[dict[str, Any]]:
        """Load, validate, decrypt, and normalize the official CSV rows.

        BrowseComp deliberately does not call ``load_tasks_from_local_json``:
        arbitrary normalized JSON would bypass the pinned encrypted CSV source.
        """
        raw_bytes = self._fetch_csv_bytes()
        rows = self._parse_validated_rows(raw_bytes)
        return [self.normalize_task(row) for row in rows]

    def normalize_task(self, raw: dict[str, Any]) -> dict[str, Any]:
        extras = self.config.extras
        id_field = str(extras["id_field"])
        question_field = str(extras["question_field"])
        answer_field = str(extras["answer_field"])
        topic_field = str(extras["topic_field"])
        canary_field = str(extras["canary_field"])

        for field in (id_field, question_field, answer_field, topic_field, canary_field):
            if field not in raw:
                raise ValueError(f"browsecomp row missing required field {field!r}")

        row_index = str(raw[id_field])
        canary = str(raw[canary_field])
        encrypted = bool(extras.get("encrypted", True))
        problem = str(raw[question_field])
        answer = str(raw[answer_field])
        if encrypted:
            problem = decrypt(problem, canary)
            answer = decrypt(answer, canary)

        return {
            "instance_id": f"browsecomp-{row_index}",
            "problem_statement": problem,
            "reference_answer": answer,
            "problem_topic": str(raw[topic_field]),
            "task_source_kind": str(extras["task_source_kind"]),
            "task_source_id": row_index,
            "task_source_path": str(raw.get("task_source_path") or extras["csv_url"]),
            "task_source_sha256": str(extras["csv_sha256"]),
            "task_source_schema": list(extras["expected_schema"]),
            "task_source_row_count": int(extras["expected_row_count"]),
            "repo": None,
            "image_name": None,
            "docker_image": None,
        }

    def runtime_mode_for(self, scaffold: str) -> str:
        self.validate_scaffold_support(scaffold)
        return "host_controller"

    def validate_scaffold_support(self, scaffold: str) -> None:
        if scaffold not in self.SUPPORTED_SCAFFOLDS:
            raise NotImplementedError(
                f"BrowseComp supports scaffold='deep-research' only, got {scaffold!r}"
            )

    def image_name_for(self, task: dict[str, Any]) -> str | None:
        return None

    def build_runner(
        self,
        *,
        scaffold: str,
        provider: Any,
        workspace_base: Path,
        max_iterations: int,
        context_window_tokens: int,
        model: str,
        **kwargs: Any,
    ) -> Any:
        self.validate_scaffold_support(scaffold)
        from agents.deep_research.runner import DeepResearchRunner

        return DeepResearchRunner(
            provider=provider,
            workspace_base=workspace_base,
            benchmark_slug=self.config.slug,
            benchmark_extras=self.config.extras,
            max_iterations=max_iterations,
            context_window_tokens=context_window_tokens,
            model=model,
            provider_name=kwargs.get("provider_name"),
            env_key=kwargs.get("env_key"),
            api_base=kwargs.get("api_base") or getattr(provider, "api_base", None) or "",
            api_key=kwargs.get("api_key") or getattr(provider, "api_key", None) or "",
            generation_config=kwargs.get("generation_config") or {},
        )

    def _fetch_csv_bytes(self) -> bytes:
        url = str(self.config.extras["csv_url"])
        with httpx.Client(timeout=60.0, follow_redirects=True) as client:
            response = client.get(url)
            response.raise_for_status()
            return response.content

    def _parse_validated_rows(self, raw_bytes: bytes) -> list[dict[str, Any]]:
        actual_sha = hashlib.sha256(raw_bytes).hexdigest()
        expected_sha = str(self.config.extras["csv_sha256"])
        if actual_sha != expected_sha:
            raise ValueError(
                "browsecomp CSV SHA-256 mismatch: "
                f"expected {expected_sha}, got {actual_sha}"
            )

        text = raw_bytes.decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(text))
        expected_schema = [str(item) for item in self.config.extras["expected_schema"]]
        if reader.fieldnames != expected_schema:
            raise ValueError(
                "browsecomp CSV schema mismatch: "
                f"expected {expected_schema}, got {reader.fieldnames}"
            )
        rows: list[dict[str, Any]] = []
        for row_index, row in enumerate(reader):
            item = dict(row)
            item[str(self.config.extras["id_field"])] = row_index
            item["task_source_path"] = str(self.config.extras["csv_url"])
            rows.append(item)
        expected_count = int(self.config.extras["expected_row_count"])
        if len(rows) != expected_count:
            raise ValueError(
                "browsecomp CSV row-count mismatch: "
                f"expected {expected_count}, got {len(rows)}"
            )
        return rows


def derive_key(password: str, length: int) -> bytes:
    """Derive the repeated SHA-256 XOR key used by BrowseComp."""
    digest = hashlib.sha256(password.encode()).digest()
    return digest * (length // len(digest)) + digest[: length % len(digest)]


def decrypt(ciphertext_b64: str, password: str) -> str:
    """Decrypt a base64-encoded BrowseComp field using the row canary."""
    encrypted = base64.b64decode(ciphertext_b64)
    key = derive_key(password, len(encrypted))
    decrypted = bytes(value ^ key_byte for value, key_byte in zip(encrypted, key))
    return decrypted.decode("utf-8")
