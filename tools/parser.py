# tools/parser.py — Conversation import for ChatGPT and Claude exports
#
# Usage:
#   python tools/parser.py --source chatgpt --file ~/Downloads/conversations.json
#   python tools/parser.py --source claude  --file ~/Downloads/conversations.json
#   python tools/parser.py --source claude  --file data/our_conversation.json --dry-run

import json
import logging
import argparse
from datetime import datetime, timezone
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from core.session import ConversationDB

log = logging.getLogger(__name__)


# ── ChatGPT ────────────────────────────────────────────────────────────────────

def parse_chatgpt(filepath: str) -> list:
    """
    Parse ChatGPT conversations.json (OpenAI data export format).
    Returns a list of normalized session dicts.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return _parse_chatgpt_from_data(data)


def _parse_chatgpt_from_data(data) -> list:
    if not isinstance(data, list):
        log.warning("Unexpected ChatGPT export structure — expected a list.")
        return []

    sessions = []
    for convo in data:
        mapping = convo.get("mapping", {})
        if not mapping:
            continue
        messages = _walk_chatgpt_tree(mapping, convo.get("current_node"))
        if len(messages) >= 2:
            # Use the export's own conversation ID for stable deduplication.
            export_id = convo.get("id") or convo.get("conversation_id") or ""
            sessions.append({
                "id":         export_id,
                "source":     "chatgpt",
                "started_at": messages[0]["timestamp"],
                "messages":   messages,
            })

    log.info("ChatGPT: %d conversations parsed.", len(sessions))
    return sessions


def _walk_chatgpt_tree(mapping: dict, current_node: str) -> list:
    """
    Reconstruct the linear conversation along the active branch by walking
    from current_node UP through parent pointers to the root, then reversing.
    This is the correct traversal for ChatGPT's tree-structured message mapping.
    """
    if not current_node:
        return []

    # Walk from leaf to root via parent pointers.
    path = []
    node_id = current_node
    visited = set()
    while node_id and node_id not in visited and node_id in mapping:
        visited.add(node_id)
        path.append(node_id)
        node_id = mapping[node_id].get("parent")

    path.reverse()  # root → leaf order

    messages = []
    for node_id in path:
        node = mapping[node_id]
        msg  = node.get("message")
        if not msg or not msg.get("content"):
            continue
        parts   = msg["content"].get("parts", [])
        content = " ".join(p for p in parts if isinstance(p, str)).strip()
        role    = msg.get("author", {}).get("role", "")
        if not content or role not in ("user", "assistant"):
            continue
        ct = msg.get("create_time")
        ts = (
            datetime.fromtimestamp(ct, tz=timezone.utc).isoformat()
            if ct
            else datetime.now(tz=timezone.utc).isoformat()
        )
        messages.append({
            "role":      "user" if role == "user" else "assistant",
            "content":   content,
            "timestamp": ts,
        })

    return messages


# ── Claude ─────────────────────────────────────────────────────────────────────

def parse_claude(filepath: str) -> list:
    """
    Parse Claude.ai conversations.json export.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return _parse_claude_from_data(data)


def _parse_claude_from_data(data) -> list:
    if isinstance(data, list):
        conversations = data
    elif isinstance(data, dict) and "conversations" in data:
        conversations = data["conversations"]
    else:
        conversations = [data]

    sessions = []
    for convo in conversations:
        messages     = []
        raw_messages = convo.get("messages", convo.get("chat_messages", []))

        for msg in raw_messages:
            role    = msg.get("role", msg.get("sender", ""))
            content = msg.get("content", "")

            if isinstance(content, list):
                content = " ".join(
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                )
            content = content.strip()

            if content and role in ("human", "user", "assistant"):
                messages.append({
                    "role":      "user" if role in ("human", "user") else "assistant",
                    "content":   content,
                    "timestamp": msg.get("created_at", datetime.now().isoformat()),
                })

        if len(messages) >= 2:
            export_id = convo.get("uuid") or convo.get("id") or ""
            sessions.append({
                "id":         export_id,
                "source":     "claude",
                "started_at": messages[0]["timestamp"],
                "messages":   messages,
            })

    log.info("Claude: %d conversations parsed.", len(sessions))
    return sessions


# ── Database import ────────────────────────────────────────────────────────────

def import_to_db(
    sessions: list,
    db: ConversationDB,
    companion_id: str | None = None,
    model_id: str | None = None,
) -> tuple[int, int]:
    """
    Import normalized sessions into SQLite using ConversationDB methods so
    every session is properly bound to the companion and visible to LoRA
    training, reflection, and search.

    Idempotent: uses the export's own session ID stored in import_batch_id
    to skip sessions that have already been imported for this companion.

    Returns (imported_count, skipped_count).
    """
    companion_id = companion_id or db.get_or_create_default_companion()

    if model_id is None:
        with db._conn() as conn:
            row = conn.execute(
                "SELECT model_id FROM model_installations WHERE is_default = 1 LIMIT 1"
            ).fetchone()
            model_id = row["model_id"] if row else None

    imported = 0
    skipped  = 0

    for i, session in enumerate(sessions):
        if i > 0 and i % 50 == 0:
            print(f"  Progress: {i}/{len(sessions)} sessions "
                  f"({imported} imported, {skipped} skipped)...")

        export_id = session.get("id") or ""

        # Idempotency: skip if this export session was already imported.
        if export_id:
            with db._conn() as conn:
                if conn.execute(
                    "SELECT id FROM sessions WHERE import_batch_id = ? AND companion_id = ?",
                    (export_id, companion_id),
                ).fetchone():
                    skipped += 1
                    continue

        try:
            session_id = db.new_session(companion_id=companion_id, model_id=model_id)

            # Backfill the export metadata that new_session() doesn't set.
            with db._conn() as conn:
                conn.execute(
                    "UPDATE sessions SET source = ?, import_batch_id = ?, started_at = ? WHERE id = ?",
                    ("import", export_id or None, session["started_at"], session_id),
                )

            for msg in session["messages"]:
                role = msg["role"]
                message_id = db.add_message(
                    session_id=session_id,
                    role=role,
                    content=msg["content"],
                    companion_id=companion_id,
                    model_id=model_id,
                    timestamp=msg.get("timestamp"),
                )
                # Mark as imported so the UI can distinguish these messages.
                with db._conn() as conn:
                    conn.execute(
                        "UPDATE messages SET imported = 1 WHERE id = ?",
                        (message_id,),
                    )

            db.end_session(session_id, lora_version="imported")
            imported += 1

        except Exception as exc:
            log.warning("Skipping session %s: %s", export_id or i, exc)
            skipped += 1

    log.info("Import complete: %d imported, %d skipped.", imported, skipped)
    return imported, skipped


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Import conversation history into Flint"
    )
    parser.add_argument(
        "--source",
        choices=["chatgpt", "claude"],
        required=True,
        help="Format of the export file",
    )
    parser.add_argument(
        "--file",
        required=True,
        help="Path to the export JSON file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and preview without writing to the database",
    )
    args = parser.parse_args()

    filepath = Path(args.file).expanduser().resolve()
    if not filepath.exists():
        print(f"File not found: {filepath}")
        sys.exit(1)

    print(f"Parsing {args.source} export: {filepath.name}")

    if args.source == "chatgpt":
        sessions = parse_chatgpt(str(filepath))
    else:
        sessions = parse_claude(str(filepath))

    if not sessions:
        print("No conversations found. Check the file format.")
        sys.exit(0)

    total_messages = sum(len(s["messages"]) for s in sessions)
    print(f"\nParsed: {len(sessions)} conversations, {total_messages} messages")
    print(f"\nSample (first 2 turns):")
    for msg in sessions[0]["messages"][:2]:
        preview = msg["content"][:120].replace("\n", " ")
        print(f"  [{msg['role']}] {preview}...")

    if args.dry_run:
        print("\nDry run — nothing saved.")
        return

    print(f"\nImporting into {config.CONVERSATIONS_DB}...")
    db = ConversationDB()
    imported, skipped = import_to_db(sessions, db)
    print(f"\nDone. {imported} sessions imported, {skipped} skipped.")
    if imported > 0:
        print("Run '/lora now' in the chat to train immediately,")
        print("or they will be included in the next nightly run.")


if __name__ == "__main__":
    main()
