"""PMAgent: turns a user requirement into a validated Spec via structured output."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from agents.config import Settings
from agents.model import make_model
from agents.prompts import PM_CONSOLIDATE_PROMPT, PM_SYSTEM_PROMPT
from orchestration.consolidation import MergePlan, build_merged, deterministic_merge, validate_merge
from orchestration.project_state import MAX_TASKS, Consolidation, Spec, Task


class PMError(Exception):
    """The PM could not produce a valid spec (model error, or two invalid outputs)."""


class PMAgent:
    def __init__(self, settings: Settings | None = None, *, model: BaseChatModel | None = None) -> None:
        self.settings = settings or Settings.from_env()
        self.model = model

    def _structured(self, schema: type = Spec) -> Any:
        if self.model is None:
            self.model = make_model(self.settings)
        return self.model.with_structured_output(schema, include_raw=True)

    def draft_spec(self, requirement: str) -> Spec:
        """Ask for a Spec; on invalid output retry once with the validation error appended."""
        try:
            structured = self._structured()
        except Exception as e:  # missing provider package, unsupported structured output, bad model id
            raise PMError(f"PM model setup failed: {type(e).__name__}: {e}") from e
        messages: list[Any] = [SystemMessage(PM_SYSTEM_PROMPT), HumanMessage(requirement)]
        error = ""
        for attempt in range(2):
            try:
                out = structured.invoke(messages)
            except Exception as e:  # provider/network failures end the run, no retry
                raise PMError(f"PM model call failed: {e}") from e
            if out.get("parsing_error") is None and out.get("parsed") is not None:
                return out["parsed"]
            error = str(out.get("parsing_error") or "the model returned no structured spec")
            if attempt == 0:
                messages.append(HumanMessage(
                    f"Your spec was invalid:\n{error}\n\nReturn a corrected spec that fixes these problems."))
        raise PMError(f"PM produced an invalid spec twice: {error}")

    def consolidate(self, spec: Spec, max_tasks: int = MAX_TASKS) -> tuple[list[Task], Consolidation]:
        """Merge the spec's tasks to at most max_tasks. Never raises: any failure uses the deterministic split."""
        try:
            out = self._structured(MergePlan).invoke([
                SystemMessage(PM_CONSOLIDATE_PROMPT.format(max_tasks=max_tasks)),
                HumanMessage(spec.model_dump_json(indent=2)),
            ])
            plan = out.get("parsed")
            if out.get("parsing_error") is None and plan is not None and not validate_merge(spec, plan, max_tasks):
                return build_merged(spec, plan.groups, "llm")
        except Exception:
            pass
        return deterministic_merge(spec, max_tasks)
