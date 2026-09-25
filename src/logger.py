import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

LOG_PATH = Path("logs/queries.jsonl")


def log_query(question: str, result: dict, seconds: float) -> str:
    """Append one query's result to the log file. Returns a log_id for later feedback."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    log_id = str(uuid.uuid4())
    entry = {
        "log_id": log_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "search_query": result.get("search_query"),
        "answer": result["answer"],
        "sources": result.get("sources", []),
        "closest_distance": result.get("closest_distance"),
        "seconds": round(seconds, 2),
        "feedback": None,
    }

    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    return log_id


def log_feedback(log_id: str, feedback: str) -> bool:
    """Update the feedback field ('up' or 'down') for a given log_id. Returns True if found."""
    if not LOG_PATH.exists():
        return False

    lines = LOG_PATH.read_text(encoding="utf-8").splitlines()
    found = False
    updated_lines = []
    for line in lines:
        entry = json.loads(line)
        if entry["log_id"] == log_id:
            entry["feedback"] = feedback
            found = True
        updated_lines.append(json.dumps(entry))

    if found:
        LOG_PATH.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")
    return found