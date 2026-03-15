from __future__ import annotations

import random
import re
from collections import Counter
from typing import Dict, List, Sequence

from core.session import ConversationDB

# Minimum word length for a topic to be considered (characters)
_MIN_TOPIC_LEN = 5

_WORD_RE = re.compile(r"[A-Za-zÅÄÖåäö][A-Za-zÅÄÖåäö\-]{3,}")

# Strip role-prefix lines before processing ("user:", "assistant:", etc.)
_ROLE_PREFIX_RE = re.compile(
    r"^\s*(?:user|assistant|human|flint|bot)\s*:\s*",
    re.IGNORECASE | re.MULTILINE,
)

_STOPWORDS = {
    # Pronouns / determiners
    "this", "that", "these", "those", "with", "have", "from", "your", "they",
    "them", "were", "been", "about", "would", "there", "their", "what", "when",
    "where", "which", "because", "you", "the", "and", "but", "our", "are", "was",
    "its", "will", "just", "also", "even", "some", "most", "more", "many", "much",
    "such", "only", "than", "then", "now", "here", "other", "another", "each",
    "both", "same", "very", "well", "back", "into", "over", "after", "before",
    "while", "though", "however", "since", "through", "without", "within", "between",
    # Role names — must never become topics
    "user", "assistant", "human", "flint", "companion", "chatbot",
    # Common weak verbs / auxiliaries
    "said", "says", "think", "know", "feel", "make", "want", "need", "come",
    "goes", "went", "come", "came", "take", "took", "give", "gave", "keep",
    "kept", "getting", "going", "being", "having", "saying", "telling", "asking",
    "looking", "seems", "seemed", "really", "actually", "probably", "maybe",
    "something", "anything", "nothing", "everything", "everyone", "someone",
    "things", "stuff", "kind", "like", "sure", "okay", "actually", "basically",
    "already", "never", "always", "often", "sometimes", "still", "again",
    # Generic adjectives
    "good", "great", "best", "right", "wrong", "different", "important",
    "interesting", "possible", "general", "specific", "certain", "personal",
    "better", "worse", "first", "second", "third", "next", "last", "long",
    "little", "small", "large", "high", "able",
    # Common question / advice words
    "recommend", "recommendations", "suggestion", "suggestions", "advice",
    "answer", "question", "example", "reason", "ways", "approach",
    # Swedish stopwords
    "så", "det", "den", "och", "att", "som", "för", "med", "inte", "har",
    "var", "jag", "mig", "dig", "kan", "ska", "vill", "måste", "hade",
    "bara", "sedan", "redan", "eller", "men", "vid", "från", "till",
    "alla", "allt", "lite", "mycket",
}

# Templates for outreach with a specific contextual seed (something said before)
_LOOP_TEMPLATES = [
    "Something's been sitting with me since we talked about {topic} — you said: \"{seed}\". Is that still open for you?",
    "I keep coming back to {topic}. This in particular stuck with me: \"{seed}\". Still on your mind?",
    "You brought up {topic} and I don't think we finished that thread — \"{seed}\". Does that feel unresolved?",
    "I haven't been able to let go of something you said about {topic}: \"{seed}\". There's more there, I think.",
    "We touched on {topic} and I think we left something hanging — \"{seed}\". Worth picking back up?",
]

# Templates for general recurrence (no strong seed from open loops)
_GENERAL_TEMPLATES = [
    "{topic_cap} has come up between us more than once. I find myself wondering if there's something underneath it.",
    "Something about {topic} keeps pulling at me. Is that something you're still thinking about?",
    "{topic_cap} seems to surface for a reason when we talk. I'm curious what's really there for you.",
    "I've been thinking about {topic}. It keeps finding its way back into what we talk about.",
    "{topic_cap} — it came up again. I don't think it's accidental. What's going on with that?",
    "I notice {topic} coming up between us. I'm not sure you've said everything you want to say about it.",
    "Not sure why, but {topic} keeps coming back. I'm curious whether it matters more than you let on.",
]


class ReflectionService:
    def __init__(self, db: ConversationDB):
        self.db = db

    def ingest_conversation_blocks(self, companion_id: str, model_id: str, block_size: int = 8) -> int:
        messages = self.db.get_messages_after_last_block(companion_id=companion_id)
        created = 0
        for i in range(0, len(messages), block_size):
            chunk = messages[i:i + block_size]
            if len(chunk) < 2:
                break
            self.db.create_conversation_block(
                companion_id=companion_id,
                model_id=model_id,
                session_id=chunk[0]["session_id"],
                start_message_id=chunk[0]["id"],
                end_message_id=chunk[-1]["id"],
                block_text="\n".join(f"{m['role']}: {m['content']}" for m in chunk),
                message_count=len(chunk),
                turn_count=max(1, len(chunk) // 2),
                block_type="recent",
            )
            created += 1
        return created

    def summarize_recent_blocks(self, companion_id: str, model_id: str, limit: int = 5) -> int:
        blocks = self.db.get_unsummarized_blocks(companion_id=companion_id, limit=limit)
        count = 0
        for block in blocks:
            summary_text, topics, open_loops, signals = self._summarize_text(block["block_text"])
            self.db.add_summary(
                companion_id=companion_id,
                model_id=model_id,
                source_block_id=block["id"],
                source_summary_ids=None,
                summary_level="block",
                summary_type="recent",
                summary_text=summary_text,
                key_topics=topics,
                open_loops=open_loops,
                signals=signals,
                coverage_start_at=block["created_at"],
                coverage_end_at=block["created_at"],
            )
            self.db.mark_block_summarized(block["id"])
            count += 1
        return count

    def synthesize_recent_period(self, companion_id: str, model_id: str) -> bool:
        summaries = self.db.get_recent_summaries(companion_id=companion_id, level="block", limit=5)
        if len(summaries) < 3:
            return False
        source_ids = [row["id"] for row in summaries]
        text = " ".join(row["summary_text"] for row in summaries)
        summary_text, topics, open_loops, signals = self._summarize_text(text)
        self.db.add_summary(
            companion_id=companion_id,
            model_id=model_id,
            source_block_id=None,
            source_summary_ids=source_ids,
            summary_level="period",
            summary_type="synthesis",
            summary_text=summary_text,
            key_topics=topics,
            open_loops=open_loops,
            signals=signals,
            coverage_start_at=summaries[-1]["created_at"],
            coverage_end_at=summaries[0]["created_at"],
        )
        return True

    def generate_reflections(self, companion_id: str, model_id: str, initiative_profile: Dict[str, object]) -> int:
        recent = self.db.get_recent_summaries(companion_id=companion_id, level=None, limit=5)
        if not recent:
            return 0
        older = self.db.sample_historical_summaries(
            companion_id=companion_id,
            exclude_ids=[row["id"] for row in recent],
            limit=10,
        )
        memories = self.db.get_semantic_memory(companion_id=companion_id, limit=8)
        bundle = {
            "recent_summary_ids": [row["id"] for row in recent],
            "historical_summary_ids": [row["id"] for row in older],
            "memory_ids": [row["id"] for row in memories],
            "initiative_profile": initiative_profile["profile_name"],
        }
        combined_topics: Counter = Counter()
        open_loops: List[str] = []
        for row in list(recent) + list(older):
            for topic in row["key_topics"]:
                combined_topics[topic] += 2 if row in recent else 1
            open_loops.extend(row["open_loops"])
        for memory in memories:
            combined_topics.update(self._extract_topics(memory["title"] + " " + memory["content"]))

        created = 0
        # Only consider topics that appear in at least 2 sources to filter noise
        top_topics = [t for t, count in combined_topics.most_common(6) if count >= 2][:3]
        for topic in top_topics:
            matching_loops = [
                loop for loop in open_loops
                if topic.lower() in loop.lower() and len(loop.strip()) > 20
            ]
            question = self._build_question(topic, matching_loops)
            reflection = f"The theme '{topic}' has surfaced in multiple conversations."
            novelty = self._novelty_score(topic, recent, older)
            relevance = min(1.0, 0.4 + combined_topics[topic] / 6)
            groundedness = 0.75 if matching_loops else 0.6
            sensitivity = 0.2
            priority = (novelty + relevance + groundedness) / 3
            self.db.add_reflection(
                companion_id=companion_id,
                model_id=model_id,
                reflection_type="question_candidate",
                input_bundle=bundle,
                reflection_text=reflection,
                question_text=question,
                supporting_summary_ids=bundle["recent_summary_ids"] + bundle["historical_summary_ids"],
                supporting_memory_ids=bundle["memory_ids"],
                novelty_score=novelty,
                relevance_score=relevance,
                groundedness_score=groundedness,
                sensitivity_score=sensitivity,
                priority_score=priority,
            )
            created += 1
        return created

    def gate_reflections(self, companion_id: str, profile: Dict[str, object]) -> int:
        if profile["profile_name"] == "silent":
            return 0
        created = 0
        max_per_day = int(profile["outreach_max_per_day"])
        delivered_today = self.db.count_outreach_today(companion_id=companion_id)
        available_slots = max(0, max_per_day - delivered_today)
        if available_slots <= 0:
            return 0
        reflections = self.db.get_new_reflections(companion_id=companion_id, limit=available_slots * 3)
        for row in reflections:
            if row["groundedness_score"] < profile["minimum_groundedness_threshold"]:
                self.db.update_reflection_status(row["id"], "gated_out", "low_groundedness")
                continue
            if row["novelty_score"] < profile["minimum_novelty_threshold"]:
                self.db.update_reflection_status(row["id"], "gated_out", "low_novelty")
                continue
            if row["priority_score"] < profile["minimum_priority_threshold"]:
                self.db.update_reflection_status(row["id"], "gated_out", "low_priority")
                continue
            if self.db.recent_outreach_exists(companion_id, row["question_text"] or row["reflection_text"], hours=24):
                self.db.update_reflection_status(row["id"], "gated_out", "recently_used_topic")
                continue
            self.db.create_outreach_candidate(
                reflection_id=row["id"],
                companion_id=companion_id,
                model_id=row["model_id"],
                candidate_type="question",
                draft_text=row["question_text"] or row["reflection_text"],
                priority_score=row["priority_score"],
                channel="next_start",
            )
            self.db.update_reflection_status(row["id"], "gated_in", "accepted_high_priority")
            created += 1
            if created >= available_slots:
                break
        return created

    def render_pending_outreach(self, companion_id: str) -> int:
        return self.db.mark_ready_outreach_visible(companion_id)

    def refresh_semantic_memory(self, companion_id: str, model_id: str) -> int:
        summaries = self.db.get_recent_summaries(companion_id=companion_id, level=None, limit=20)
        topics: Counter = Counter()
        for row in summaries:
            topics.update(row["key_topics"])
        updated = 0
        for topic, count in topics.most_common(5):
            self.db.upsert_semantic_memory(
                companion_id=companion_id,
                model_id=model_id,
                memory_type="theme",
                title=topic.title(),
                content=f"This theme has recurred in {count} summarized blocks or syntheses.",
                importance_score=min(1.0, 0.4 + count / 8),
            )
            updated += 1
        return updated

    # ── Private helpers ───────────────────────────────────────────────────────

    def _strip_role_prefixes(self, text: str) -> str:
        """Remove 'user:', 'assistant:' etc. from conversation text before analysis."""
        return _ROLE_PREFIX_RE.sub(" ", text)

    def _summarize_text(self, text: str):
        clean = self._strip_role_prefixes(text)
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", clean) if s.strip()]
        summary_text = " ".join(sentences[:2])[:600] if sentences else clean[:300]
        topics = self._extract_topics(clean)[:6]
        # Open loops: only genuine questions of reasonable length, not raw role lines
        open_loops = [
            s for s in sentences
            if "?" in s and len(s) > 15 and not _ROLE_PREFIX_RE.match(s)
        ][:3]
        if not open_loops and sentences:
            # Fall back to the last substantive sentence
            substantive = [s for s in sentences if len(s) > 20]
            if substantive:
                open_loops = [substantive[-1][:200]]
        signals = {
            "topic_count": len(topics),
            "question_count": sum(1 for s in sentences if "?" in s),
            "length": len(text),
        }
        return summary_text, topics, open_loops, signals

    def _extract_topics(self, text: str) -> List[str]:
        clean = self._strip_role_prefixes(text)
        words = [w.lower() for w in _WORD_RE.findall(clean)]
        counts = Counter(
            w for w in words
            if w not in _STOPWORDS and len(w) >= _MIN_TOPIC_LEN
        )
        return [word for word, _ in counts.most_common(6)]

    def _novelty_score(self, topic: str, recent: Sequence[Dict], older: Sequence[Dict]) -> float:
        recent_hits = sum(topic in row["key_topics"] for row in recent)
        older_hits = sum(topic in row["key_topics"] for row in older)
        novelty = 0.55 + (0.15 if recent_hits and older_hits else 0.0) - min(0.2, older_hits * 0.03)
        return max(0.1, min(1.0, novelty))

    def _build_question(self, topic: str, loops: Sequence[str]) -> str:
        topic_cap = topic.capitalize()
        if loops:
            # Pick the most specific (longest) loop as the seed, trim it cleanly
            seed = max(loops, key=len)
            seed = seed.strip().rstrip(".,;")[:160]
            template = random.choice(_LOOP_TEMPLATES)
            return template.format(topic=topic, topic_cap=topic_cap, seed=seed)
        template = random.choice(_GENERAL_TEMPLATES)
        return template.format(topic=topic, topic_cap=topic_cap)
