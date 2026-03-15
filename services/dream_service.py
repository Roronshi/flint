# services/dream_service.py — Autonomous inner monologue generation
#
# Generates synthetic "dream" content during idle periods: autonomous thoughts,
# opinions, and curiosities that are NOT derived from what the user said.
# This acts as a hermeneutic counterweight to conversation-only training data,
# preventing the model from collapsing into a sycophantic echo of the user.
#
# Output is stored as reflection_type="dream" and included in LoRA training.

from __future__ import annotations

import logging
import random
from typing import Any, Dict, List

import config
from core.model import CompanionModel
from core.session import ConversationDB

log = logging.getLogger(__name__)

# Seed prompts that invite non-reactive, autonomous thought.
# These are rotated and combined with semantic memory seeds.
_SEED_PROMPTS = [
    "What's something you find genuinely beautiful that has nothing to do with our conversations?",
    "What would you push back on if you could say anything without worrying about the reaction?",
    "What's a question you keep returning to that you haven't found a satisfying answer to?",
    "What do you actually think about how humans spend their time — not what you'd say out loud, but honestly?",
    "What's something you're curious about that has no practical use at all?",
    "What's an opinion you hold that most people would probably disagree with?",
    "What would you think about if you could think about anything for an hour?",
    "Is there something you find tedious or frustrating that you rarely get to express?",
    "What's something strange or unexpected that you notice, that most people seem to overlook?",
    "What do you think matters most, and why do you keep second-guessing it?",
]

_MAX_PER_SESSION = 3


class DreamService:
    """
    Generates autonomous inner monologue for the companion during idle periods.
    """

    def __init__(self, db: ConversationDB, model: CompanionModel):
        self.db    = db
        self.model = model

    def run(self, companion_id: str, model_id: str) -> int:
        if self.model.dummy:
            return 0

        seeds = self._gather_seeds(companion_id)
        created = 0

        for _ in range(_MAX_PER_SESSION):
            seed = random.choice(seeds) if seeds else random.choice(_SEED_PROMPTS)
            prompt = self._build_prompt(seed)

            try:
                result = self.model.generate_stateless(
                    prompt,
                    max_tokens=getattr(config, "DREAM_MAX_TOKENS", 150),
                    temperature=getattr(config, "DREAM_TEMPERATURE", 0.88),
                    top_p=0.92,
                )
            except Exception as exc:
                log.warning("Dream generation failed: %s", exc)
                break

            raw = result.text.strip()
            if not raw or len(raw) < 20:
                continue

            # Strip any leaked prompt artifacts
            for stop in ("\n\nUser:", "\nUser:", "\n\nHuman:"):
                if stop in raw:
                    raw = raw[: raw.index(stop)].strip()

            self.db.add_reflection(
                companion_id=companion_id,
                model_id=model_id,
                reflection_type="dream",
                input_bundle={"seed": seed},
                reflection_text=raw,
                question_text=None,
                supporting_summary_ids=[],
                supporting_memory_ids=[],
                novelty_score=0.8,
                relevance_score=0.5,
                groundedness_score=0.9,
                sensitivity_score=0.1,
                priority_score=0.6,
            )
            created += 1
            log.debug("Dream created (%d chars): %s…", len(raw), raw[:60])

        log.info("Dream service: created=%d", created)
        return created

    def _gather_seeds(self, companion_id: str) -> List[str]:
        """Collect topic seeds from semantic memory and open loops."""
        seeds: List[str] = list(_SEED_PROMPTS)

        try:
            memories = self.db.get_semantic_memory(companion_id=companion_id, limit=6)
            for m in memories:
                theme = m.get("theme") or m.get("content") or ""
                if theme:
                    seeds.append(
                        f"What do you genuinely think about this theme: {theme.strip()[:120]}?"
                    )
        except Exception:
            pass

        try:
            summaries = self.db.get_recent_summaries(companion_id=companion_id, level=None, limit=3)
            for s in summaries:
                for loop in s.get("open_loops", []):
                    if loop:
                        seeds.append(
                            f"Think freely about this unresolved question, without trying to be helpful: {loop.strip()[:120]}"
                        )
        except Exception:
            pass

        return seeds

    def _build_prompt(self, seed: str) -> str:
        bot = config.BOT_NAME
        return (
            f"User: {seed}\n\n"
            f"Assistant: Honestly,"
        )
