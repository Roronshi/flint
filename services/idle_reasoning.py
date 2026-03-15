from __future__ import annotations

import logging
from typing import Any, Dict, List

import config
from core.model import CompanionModel
from core.session import ConversationDB
from services.reflection_service import ReflectionService

log = logging.getLogger(__name__)


class IdleReasoningService:
    """Generates conversation-keeping questions from recent shared history."""

    def __init__(self, db: ConversationDB, model: CompanionModel, reflection_service: ReflectionService):
        self.db = db
        self.model = model
        self.reflection_service = reflection_service

    def _build_context(self, companion_id: str) -> Dict[str, Any]:
        summaries = self.db.get_recent_summaries(companion_id=companion_id, level=None, limit=5)
        memories = self.db.get_semantic_memory(companion_id=companion_id, limit=8)
        open_loops: List[str] = []
        for row in summaries:
            open_loops.extend(row.get("open_loops", []))
        return {"summaries": summaries, "memories": memories, "open_loops": open_loops[:10]}

    def _prompt_from_context(self, bundle: Dict[str, Any]) -> str:
        # Pull out the most recent summary texts and open loops as topic hints.
        summary_lines = [row["summary_text"] for row in bundle["summaries"] if row.get("summary_text")]
        loop_lines = [item for item in bundle["open_loops"] if item]

        topics: List[str] = []
        topics.extend(summary_lines[:3])
        topics.extend(loop_lines[:3])
        topics_txt = "\n".join(f"- {t}" for t in topics) if topics else "- our recent conversations"

        # Format as G1 World chat so the model stays in character.
        # The model sees itself mid-conversation and naturally continues as companion.
        return (
            f"User: What's been on your mind lately?\n\n"
            f"Assistant: I've been thinking about some of the things we've talked about.\n\n"
            f"User: Like what?\n\n"
            f"Assistant: Things like:\n{topics_txt}\n\n"
            f"It makes me want to ask you something. "
        )

    def _parse_questions(self, text: str) -> List[str]:
        out = []
        # Take the first sentence-like chunk that ends with a question mark.
        for raw in text.splitlines():
            line = raw.strip().lstrip("-•0123456789. ")
            if not line or len(line) < 12:
                continue
            # Prefer lines that are actual questions
            if "?" in line:
                # Trim anything after the first question mark
                q = line[:line.index("?") + 1].strip()
                if len(q) >= 12:
                    out.append(q)
            elif not out:
                # Accept first non-empty line as fallback if no question found yet
                out.append(line)
            if len(out) >= 2:
                break
        return out[:2]

    def run(self, companion_id: str, model_id: str, initiative_profile: Dict[str, Any]) -> Dict[str, int]:
        if self.model.dummy or not getattr(self.model.backend, "supports_reasoning_mode", False):
            created = self.reflection_service.generate_reflections(companion_id, model_id, initiative_profile)
            gated = self.reflection_service.gate_reflections(companion_id, initiative_profile)
            visible = self.reflection_service.render_pending_outreach(companion_id)
            return {"created": created, "gated": gated, "visible": visible, "mode": 0}

        bundle = self._build_context(companion_id)
        prompt = self._prompt_from_context(bundle)
        result = self.model.generate_stateless(prompt, max_tokens=80, temperature=0.9, top_p=0.85)

        # Strip any leaked prompt artifacts before parsing
        raw = result.text.strip()
        # Remove anything that looks like a new user turn starting
        for stop in (f"\n\nUser:", f"\nUser:", f"\n{config.USER_NAME}:"):
            if stop in raw:
                raw = raw[:raw.index(stop)].strip()

        questions = self._parse_questions(raw)
        created = 0
        for question in questions:
            self.db.add_reflection(
                companion_id=companion_id,
                model_id=model_id,
                reflection_type="idle_reasoning",
                input_bundle=bundle,
                reflection_text="Idle conversation prompt",
                question_text=question,
                supporting_summary_ids=[row["id"] for row in bundle["summaries"]],
                supporting_memory_ids=[row["id"] for row in bundle["memories"]],
                novelty_score=0.65,
                relevance_score=0.7,
                groundedness_score=0.75,
                sensitivity_score=0.2,
                priority_score=0.7,
            )
            created += 1

        gated = self.reflection_service.gate_reflections(companion_id, initiative_profile)
        visible = self.reflection_service.render_pending_outreach(companion_id)
        log.info("Idle reasoning created=%s gated=%s visible=%s", created, gated, visible)
        return {"created": created, "gated": gated, "visible": visible, "mode": 1}
