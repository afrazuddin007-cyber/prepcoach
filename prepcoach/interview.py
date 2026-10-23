"""Mock-interview mode: an interviewer who asks open-ended questions and follows up.

Each problem has a rubric of key points. When Bedrock is available it judges which points
the candidate actually covered (so natural spoken answers count) and writes the follow-up
question. Otherwise a keyword rubric is used so everything still runs offline.
Control flow and scoring are deterministic either way.
"""
import json
import os
import random
import re
import uuid

from . import ai, store
from .checker import is_correct

_PATH = os.path.join(os.path.dirname(__file__), "interview_problems.json")
with open(_PATH, encoding="utf-8") as f:
    PROBLEMS = {p["id"]: p for p in json.load(f)}
TOPICS = sorted({p["topic"] for p in PROBLEMS.values()})

MAX_TURNS = 3      # first answer plus up to two follow-ups per question
PASS_SCORE = 6     # score out of 10 that counts as "correct" for weak-topic tracking


_EXPLAIN = re.compile(r"\b(explain|teach|tell me|show me|give up|skip|you explain|answer please|what is the answer)\b")
_IDK = re.compile(r"\b(don'?t know|do not know|no idea|not sure|unsure|sorry|hint|help|stuck|clue)\b")


_IDK_EXACT = {"no", "nope", "nah", "nothing", "pass", "idk", "dunno", "i dunno", "nothing really"}


def _intent(text: str) -> str | None:
    """Detect a short 'I don't know' or 'please explain' reply. Long answers are never matched."""
    t = text.lower().strip()
    if len(t.split()) > 12:
        return None
    if _EXPLAIN.search(t):
        return "explain"
    if _IDK.search(t) or re.sub(r"[^a-z' ]", "", t).strip() in _IDK_EXACT:
        return "idk"  # a bare "No." counts, but "no extra space needed..." is a real answer
    return None


def _join(items: list[str]) -> str:
    return "; ".join(items)


def _pick_problems(topic: str | None, n: int, user_id: str | None = None) -> tuple[list[dict] | None, str | None]:
    seen = store.seen_question_ids(user_id)

    def ordered(topic_name: str) -> list[dict]:
        pool = [p for p in PROBLEMS.values() if p["topic"] == topic_name]
        random.shuffle(pool)
        return sorted(pool, key=lambda p: p["id"] in seen)  # unseen first

    if topic:
        t = topic.lower().strip()
        if t not in TOPICS:
            return None, f"Unknown topic '{topic}'. Available: {', '.join(TOPICS)}"
        return ordered(t)[:n], None

    stats = store.topic_stats(user_id)
    weakest = min(stats, key=lambda t: (stats[t]["accuracy"], stats[t]["attempts"])) if stats else None
    first = weakest if weakest in TOPICS else random.choice(TOPICS)
    rest = [t for t in TOPICS if t != first]
    random.shuffle(rest)
    per_topic = [ordered(t) for t in [first] + rest]
    # One problem from each topic first, then second problems, and so on.
    picked = []
    for round_idx in range(2):
        for pool in per_topic:
            if round_idx < len(pool):
                picked.append(pool[round_idx])
    return picked[:n], None


def start(topic: str | None = None, num_questions: int = 2, user_id: str | None = None) -> dict:
    n = max(1, min(int(num_questions or 2), 5))
    problems, error = _pick_problems(topic, n, user_id)
    if error:
        return {"error": error}
    first = problems[0]
    state = {
        "id": uuid.uuid4().hex[:12],
        "user_id": store.norm_user(user_id),
        "status": "active",
        "problem_ids": [p["id"] for p in problems],
        "idx": 0,
        "turn": 0,
        "covered": [],
        "asked": [],
        "nudged": [],
        "transcript": [{"role": "interviewer", "text": first["prompt"]}],
        "results": [],
    }
    store.save_interview(state["id"], state)
    count = len(problems)
    return {
        "session_id": state["id"],
        "total_questions": count,
        "topics": sorted({p["topic"] for p in problems}),
        "question_title": first["title"],
        "interviewer_says": (
            f"Hi, I'm your interviewer today. We'll go through {count} question{'s' if count > 1 else ''}. "
            "Talk me through your thinking like it's a real interview. "
            f"First question: {first['prompt']}"
        ),
    }


def _grade(problem: dict, covered_prev: list[str], transcript: list[dict]) -> tuple[set, str, str, str]:
    """Return (covered ids, feedback, follow_up, source)."""
    valid = {kp["id"] for kp in problem["key_points"]}
    llm = ai.grade_interview_turn(problem, covered_prev, transcript)
    if llm:
        covered = set(covered_prev) | (set(llm["covered_point_ids"]) & valid)
        return covered, llm["feedback"], llm["follow_up"], llm.get("source", "bedrock")

    candidate_text = " ".join(t["text"] for t in transcript if t["role"] == "candidate")
    covered = set(covered_prev) | {
        kp["id"] for kp in problem["key_points"] if is_correct(candidate_text, kp["keywords"])
    }
    new = [kp["description"] for kp in problem["key_points"] if kp["id"] in covered - set(covered_prev)]
    if new:
        feedback = f"Good, you brought up: {_join(new[:2])}."
    elif not covered:
        first_turn = sum(1 for t in transcript if t["role"] == "candidate") == 1
        feedback = "I'm not hearing a clear approach yet." if first_turn else "No problem, let's break it down."
    else:
        feedback = "Not quite what I was after, let's try a different angle."
    return covered, feedback, "", "fallback"


def _report(state: dict) -> dict:
    results = state["results"]
    if not results:
        return {"questions_completed": 0, "average_score": None, "results": [], "verdict": "No questions completed yet."}
    avg = round(sum(r["score"] for r in results) / len(results), 1)
    if avg >= 8:
        verdict = "Strong. You're close to interview ready on these topics."
    elif avg >= 6:
        verdict = "Solid, but tighten the gaps below."
    else:
        verdict = "Needs more practice. Review the missed points and try again."
    weakest = min(results, key=lambda r: r["score"])
    return {
        "questions_completed": len(results),
        "average_score": avg,
        "verdict": verdict,
        "results": results,
        "study_next": (
            {"problem": weakest["title"], "topic": weakest["topic"], "missed": weakest["missed"]}
            if weakest["missed"] else None
        ),
    }


def answer(session_id: str, answer_text: str) -> dict:
    state = store.load_interview(session_id)
    if not state:
        return {"error": f"Unknown session_id '{session_id}'. Start one with start_interview."}
    if state["status"] == "finished":
        return {"finished": True, "interviewer_says": "That interview is already finished.", "report": _report(state)}
    if not answer_text or not answer_text.strip():
        return {"question_done": False, "interviewer_says": "Sorry, I didn't catch that. Could you say it again?"}

    problem = PROBLEMS[state["problem_ids"][state["idx"]]]
    state["transcript"].append({"role": "candidate", "text": answer_text.strip()})
    state["turn"] += 1

    intent = _intent(answer_text)
    if intent == "idk" and state["turn"] < MAX_TURNS:
        # The candidate is stuck: give a nudge instead of re-asking, like a good interviewer.
        asked = state.setdefault("asked", [])
        open_points = [kp for kp in problem["key_points"] if kp["id"] not in state["covered"]]
        # Never nudge the same point twice: help with the point they were just asked about if it
        # hasn't been nudged yet, otherwise move on to another open point.
        nudged = state.setdefault("nudged", [])
        fresh = [k for k in open_points if k["id"] not in nudged]
        just_asked = next((k for k in fresh if asked and k["id"] == asked[-1]), None)
        kp = just_asked or (fresh[0] if fresh else None)
        if kp:
            nudged.append(kp["id"])
            if kp["id"] not in asked:
                asked.append(kp["id"])
            if len(nudged) == 1:
                nudge = (
                    "No worries, that's what practice is for. A strong answer would mention "
                    f"{kp['description']}. Can you say a bit more about how that works?"
                )
            else:
                nudge = (
                    "That's fine, let's come at it from another angle. It also helps to mention "
                    f"{kp['description']}. How might that fit into your answer?"
                )
            state["transcript"].append({"role": "interviewer", "text": nudge})
            store.save_interview(state["id"], state)
            return {"question_done": False, "turn": state["turn"], "interviewer_says": nudge, "grading_source": "rules"}

    covered, feedback, follow_up, source = _grade(problem, state["covered"], state["transcript"])
    state["covered"] = sorted(covered)
    missing = [kp for kp in problem["key_points"] if kp["id"] not in covered]

    if intent == "explain":
        feedback = "Sure, let me walk you through it."

    if missing and state["turn"] < MAX_TURNS and intent != "explain":
        if not follow_up:
            # Never repeat a probe: move on to the next missing point we haven't asked about.
            asked = state.setdefault("asked", [])
            nxt = next((kp for kp in missing if kp["id"] not in asked), None)
            if nxt:
                follow_up = nxt["probe"]
                asked.append(nxt["id"])
            else:
                follow_up = "Take your best guess, even a rough idea helps."
        state["transcript"].append({"role": "interviewer", "text": follow_up})
        store.save_interview(state["id"], state)
        return {
            "question_done": False,
            "turn": state["turn"],
            "interviewer_says": f"{feedback} {follow_up}".strip(),
            "grading_source": source,
        }

    # Question finished: score it, record it for weak-topic tracking, move on.
    total = len(problem["key_points"])
    score = round(10 * len(covered) / total)
    state["results"].append({
        "problem_id": problem["id"],
        "title": problem["title"],
        "topic": problem["topic"],
        "score": score,
        "covered": [kp["description"] for kp in problem["key_points"] if kp["id"] in covered],
        "missed": [kp["description"] for kp in missing],
        "model_answer": problem.get("model_answer", ""),
    })
    store.record_attempt(problem["id"], problem["topic"], score >= PASS_SCORE, state.get("user_id"))

    if missing:
        wrapup = f"Here's how I'd answer it: {problem.get('model_answer') or _join([kp['description'] for kp in missing[:3]])}"
    else:
        wrapup = "That covered everything I was looking for."
    lead = f"{feedback} {wrapup}".strip()

    state["idx"] += 1
    state["turn"] = 0
    state["covered"] = []
    state["asked"] = []
    state["nudged"] = []
    if state["idx"] >= len(state["problem_ids"]):
        state["status"] = "finished"
        state["transcript"] = []
        store.save_interview(state["id"], state)
        report = _report(state)
        return {
            "question_done": True,
            "question_score": score,
            "finished": True,
            "grading_source": source,
            "interviewer_says": (
                f"{lead} That's the end of the interview. Your average score is "
                f"{report['average_score']} out of 10. {report['verdict']}"
            ),
            "report": report,
        }

    nxt = PROBLEMS[state["problem_ids"][state["idx"]]]
    state["transcript"] = [{"role": "interviewer", "text": nxt["prompt"]}]
    store.save_interview(state["id"], state)
    return {
        "question_done": True,
        "question_score": score,
        "finished": False,
        "next_question_title": nxt["title"],
        "grading_source": source,
        "interviewer_says": f"{lead} Next question: {nxt['prompt']}",
    }


def report(session_id: str) -> dict:
    state = store.load_interview(session_id)
    if not state:
        return {"error": f"Unknown session_id '{session_id}'."}
    return {"finished": state["status"] == "finished", **_report(state)}
