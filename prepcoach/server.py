"""PrepCoach: a voice-friendly DSA placement coach exposed as an MCP server.

Transport: Streamable HTTP (endpoint: /mcp). Run:  python -m prepcoach.server
"""
import json
import os
import random

from mcp.server.mcpserver import MCPServer

from . import ai, interview, store
from .checker import is_correct

_QUESTIONS_PATH = os.path.join(os.path.dirname(__file__), "questions.json")
with open(_QUESTIONS_PATH, encoding="utf-8") as f:
    QUESTIONS = {q["id"]: q for q in json.load(f)}
TOPICS = sorted({q["topic"] for q in QUESTIONS.values()})

mcp = MCPServer(
    "PrepCoach",
    instructions=(
        "PrepCoach is a voice DSA interview coach. Ask the student a question with "
        "get_question, check their spoken answer with check_answer, offer get_hint if "
        "they are stuck, and use get_weak_topics to decide what to practice next. "
        "Keep spoken replies short."
    ),
)


def _public(q: dict) -> dict:
    return {
        "question_id": q["id"],
        "topic": q["topic"],
        "difficulty": q["difficulty"],
        "question": q["question"],
    }


def _weakest_topic(user_id: str | None = None) -> str | None:
    stats = store.topic_stats(user_id)
    if not stats:
        return None
    return min(stats, key=lambda t: (stats[t]["accuracy"], stats[t]["attempts"]))


@mcp.tool()
def list_topics() -> dict:
    """List the DSA topics available for practice."""
    return {"topics": TOPICS}


@mcp.tool()
def get_question(topic: str | None = None, difficulty: str | None = None, user_id: str | None = None) -> dict:
    """Get a practice question.

    If no topic is given, PrepCoach picks the student's weakest topic
    (or a random one if they have no history yet), preferring unseen questions.
    Pass user_id to keep each student's history separate.
    """
    chosen_topic = topic.lower().strip() if topic else None
    reason = "you asked for this topic"
    if chosen_topic and chosen_topic not in TOPICS:
        return {"error": f"Unknown topic '{topic}'. Available: {', '.join(TOPICS)}"}
    if not chosen_topic:
        weakest = _weakest_topic(user_id)
        chosen_topic = weakest or random.choice(TOPICS)
        reason = "this is your weakest topic" if weakest else "a fresh start"

    pool = [q for q in QUESTIONS.values() if q["topic"] == chosen_topic]
    if difficulty:
        filtered = [q for q in pool if q["difficulty"] == difficulty.lower().strip()]
        pool = filtered or pool
    seen = store.seen_question_ids(user_id)
    unseen = [q for q in pool if q["id"] not in seen]
    q = random.choice(unseen or pool)
    return {**_public(q), "why_this_topic": reason}


@mcp.tool()
def get_hint(question_id: str) -> dict:
    """Get a short hint for a question without revealing the answer."""
    q = QUESTIONS.get(question_id)
    if not q:
        return {"error": f"Unknown question_id '{question_id}'."}
    hint, source = ai.make_hint(q)
    return {"question_id": question_id, "hint": hint, "hint_source": source}


@mcp.tool()
def check_answer(question_id: str, answer: str, user_id: str | None = None) -> dict:
    """Check the student's spoken answer, record the attempt, and explain if wrong."""
    q = QUESTIONS.get(question_id)
    if not q:
        return {"error": f"Unknown question_id '{question_id}'."}
    correct = is_correct(answer, q["answers"])
    store.record_attempt(q["id"], q["topic"], correct, user_id)
    if correct:
        return {"correct": True, "feedback": "Correct! " + q["explanation"]}
    explanation, source = ai.make_explanation(q, answer)
    return {"correct": False, "feedback": explanation, "explanation_source": source}


@mcp.tool()
def get_weak_topics(user_id: str | None = None) -> dict:
    """Rank topics from weakest to strongest based on the student's history."""
    stats = store.topic_stats(user_id)
    ranked = sorted(stats.items(), key=lambda kv: (kv[1]["accuracy"], kv[1]["attempts"]))
    untried = [t for t in TOPICS if t not in stats]
    return {
        "weakest_first": [{"topic": t, **s} for t, s in ranked],
        "not_tried_yet": untried,
    }


@mcp.tool()
def get_progress(user_id: str | None = None) -> dict:
    """Overall progress summary across all topics."""
    stats = store.topic_stats(user_id)
    attempts = sum(s["attempts"] for s in stats.values())
    correct = sum(s["correct"] for s in stats.values())
    return {
        "total_attempts": attempts,
        "total_correct": correct,
        "overall_accuracy": round(correct / attempts, 2) if attempts else None,
        "by_topic": stats,
    }


@mcp.tool()
def start_interview(topic: str | None = None, num_questions: int = 2, user_id: str | None = None) -> dict:
    """Start a spoken mock interview.

    The interviewer asks open-ended problems ("walk me through how you'd solve..."),
    listens, and asks follow-up questions about anything missed. With no topic it
    starts from the student's weakest topic. Pass user_id to keep each student's
    history separate. Read `interviewer_says` aloud, then pass the student's reply
    to answer_interview.
    """
    return interview.start(topic, num_questions, user_id)


@mcp.tool()
def answer_interview(session_id: str, answer: str) -> dict:
    """Give the student's spoken answer to the interviewer.

    Returns `interviewer_says` (feedback plus a follow-up or the next question).
    Read it aloud and keep passing replies until `finished` is true.
    """
    return interview.answer(session_id, answer)


@mcp.tool()
def get_interview_report(session_id: str) -> dict:
    """Scores, covered and missed points per question, and what to study next."""
    return interview.report(session_id)


@mcp.tool()
def llm_status(check: bool = False) -> dict:
    """Report which AI (Amazon Bedrock or Gemini) is configured; check=True makes one live test call."""
    return ai.status(check)


@mcp.tool()
def reset_progress(user_id: str | None = None) -> dict:
    """Clear one student's saved attempts and interviews (useful before recording the demo)."""
    store.reset(store.norm_user(user_id))
    return {"status": "progress cleared"}


# --- Voice simulator (browser UI) -------------------------------------------------
# The page does speech in/out and drives the tools above. Each tool call goes through
# a real MCP client session against this server (in-memory transport), so the simulator
# exercises exactly the tools an assistant like Alexa+ would call over Streamable HTTP.
from mcp.client import Client  # noqa: E402
from starlette.requests import Request  # noqa: E402
from starlette.responses import HTMLResponse, JSONResponse  # noqa: E402

_WEB_DIR = os.path.join(os.path.dirname(__file__), "web")
_SIM_TOOLS = {
    "start_interview", "answer_interview", "get_interview_report",
    "get_weak_topics", "get_progress", "reset_progress", "llm_status",
}


@mcp.custom_route("/", methods=["GET"])
async def simulator_page(request: Request) -> HTMLResponse:
    with open(os.path.join(_WEB_DIR, "index.html"), encoding="utf-8") as f:
        return HTMLResponse(f.read())


@mcp.custom_route("/api/status", methods=["GET"])
async def simulator_status(request: Request) -> JSONResponse:
    return JSONResponse(ai.status())  # configuration only: no keys, no network call


@mcp.custom_route("/api/tool", methods=["POST"])
async def simulator_tool(request: Request) -> JSONResponse:
    try:
        body = await request.json()
        name, args = body.get("name"), body.get("args") or {}
        if name not in _SIM_TOOLS or not isinstance(args, dict):
            return JSONResponse({"error": f"Tool '{name}' is not available in the simulator."}, status_code=400)
        async with Client(mcp) as client:
            result = await client.call_tool(name, args)
        text = result.content[0].text if result.content else "{}"
        return JSONResponse(json.loads(text))
    except Exception as exc:  # surface errors to the page instead of a blank 500
        return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=500)


def main() -> None:
    mcp.run(
        transport="streamable-http",
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8000")),
        stateless_http=True,
        json_response=True,
    )


if __name__ == "__main__":
    main()
