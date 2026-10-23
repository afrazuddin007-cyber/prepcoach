"""Offline tests for mock interview mode. Run: python tests/test_interview.py"""
import io
import json
import os
import sys
import tempfile
import urllib.error
from unittest import mock

os.environ["PREPCOACH_DB"] = os.path.join(tempfile.mkdtemp(), "test.db")
for _var in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GEMINI_MODEL", "PREPCOACH_LLM",
             "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_PROFILE",
             "AWS_BEARER_TOKEN_BEDROCK"):
    os.environ.pop(_var, None)  # tests must never depend on, or use, real keys from the environment
# Tests must never reach a real AI service, whatever is set in the developer's shell.
for _var in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
    os.environ.pop(_var, None)
os.environ["LLM_PROVIDER"] = "auto"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from prepcoach import ai, interview, store  # noqa: E402


def ideal_answer(problem: dict) -> str:
    """An answer that hits every rubric point (first keyword of each)."""
    return "I would cover: " + ", ".join(kp["keywords"][0] for kp in problem["key_points"])


def current_problem(session_id: str) -> dict:
    state = store.load_interview(session_id)
    return interview.PROBLEMS[state["problem_ids"][state["idx"]]]


def no_bedrock(*args, **kwargs):
    return None


def test_fallback_flow():
    ai._ask_bedrock = no_bedrock
    store.reset()
    s = interview.start("hashmaps", 2)
    assert s["total_questions"] == 2 and "interviewer_says" in s, s
    sid = s["session_id"]

    r = interview.answer(sid, "I would just loop through it")
    assert r["question_done"] is False and r["turn"] == 1, r
    assert r["grading_source"] == "fallback"
    assert len(r["interviewer_says"]) > 20

    r = interview.answer(sid, ideal_answer(current_problem(sid)))
    assert r["question_done"] is True and r["question_score"] == 10 and r["finished"] is False, r
    assert "Next question" in r["interviewer_says"]

    r = interview.answer(sid, ideal_answer(current_problem(sid)))
    assert r["finished"] is True and r["report"]["average_score"] == 10.0, r
    assert "end of the interview" in r["interviewer_says"]

    again = interview.answer(sid, "hello?")
    assert again["finished"] is True
    stats = store.topic_stats()
    assert stats["hashmaps"]["attempts"] == 2 and stats["hashmaps"]["accuracy"] == 1.0, stats
    print("ok  fallback flow, follow-up, scoring, weak-topic tracking")


def test_turn_limit_and_low_score():
    ai._ask_bedrock = no_bedrock
    store.reset()
    sid = interview.start("sorting", 1)["session_id"]
    r = None
    for _ in range(interview.MAX_TURNS):
        r = interview.answer(sid, "I am not sure")
    assert r["question_done"] and r["finished"] and r["question_score"] == 0, r
    assert r["report"]["study_next"]["missed"], r
    assert store.topic_stats()["sorting"]["accuracy"] == 0.0
    print("ok  turn limit ends the question, low score recorded")


def test_no_repeated_followups():
    ai._ask_bedrock = no_bedrock
    store.reset()
    sid = interview.start("hashmaps", 1)["session_id"]
    first = interview.answer(sid, "Use a hashmap and doubly linked list" if
                             current_problem(sid)["id"] == "lru-cache" else "nothing")
    second = interview.answer(sid, "nothing")
    assert first["question_done"] is False and second["question_done"] is False, (first, second)
    probes = [t["text"] for t in store.load_interview(sid)["transcript"] if t["role"] == "interviewer"]
    followups = probes[1:]  # first entry is the opening question
    assert len(followups) == 2 and followups[0] != followups[1], followups
    assert "Okay." not in second["interviewer_says"][:6], second
    print("ok  follow-ups never repeat the same question")


def test_nudge_and_explain():
    ai._ask_bedrock = no_bedrock
    # Short "I don't know" gets a nudge, not a repeat of the question.
    store.reset()
    sid = interview.start("sorting", 1)["session_id"]
    r = interview.answer(sid, "sorry I don't know that")
    assert r["question_done"] is False and "A strong answer would mention" in r["interviewer_says"], r
    # "You explain" ends the question with a model answer and records the attempt.
    r = interview.answer(sid, "I don't know that you explain me")
    assert r["question_done"] and r["finished"], r
    assert "Here's how I'd answer it" in r["interviewer_says"], r
    assert r["report"]["results"][0]["model_answer"], r
    assert store.topic_stats()["sorting"]["accuracy"] == 0.0
    # A long real answer containing "not sure" is graded normally, not treated as "I don't know".
    store.reset()
    sid = interview.start("sorting", 1)["session_id"]
    long_answer = ("I'm not sure about the space part but I would split the array into two halves "
                   "and sort each half recursively then merging them")
    r = interview.answer(sid, long_answer)
    assert "A strong answer would mention" not in r["interviewer_says"], r
    print("ok  'I don't know' gets a nudge, 'explain' gets a model answer, long answers graded normally")


def test_nudges_do_not_repeat_and_bare_no():
    ai._ask_bedrock = no_bedrock
    assert interview._intent("No.") == "idk" and interview._intent("nope") == "idk"
    assert interview._intent("no extra space needed, I would use a hash map") is None
    store.reset()
    sid = interview.start("hashmaps", 1)["session_id"]
    st = store.load_interview(sid)
    st["problem_ids"], st["idx"] = ["lru-cache"], 0
    st["transcript"] = [{"role": "interviewer", "text": interview.PROBLEMS["lru-cache"]["prompt"]}]
    store.save_interview(sid, st)
    first = interview.answer(sid, "I don't know that")
    second = interview.answer(sid, "I genuinely don't know that")
    assert first["question_done"] is False and second["question_done"] is False, (first, second)
    assert first["interviewer_says"] != second["interviewer_says"], "the same nudge must not repeat"
    print("ok  repeated 'I don't know' gets a different nudge; a bare 'No.' counts as stuck")


def test_probes_are_self_contained():
    for p in interview.PROBLEMS.values():
        assert p.get("model_answer"), p["id"]
        for kp in p["key_points"]:
            low = kp["probe"].lower()
            assert "that element" not in low and "the running sum" not in low, (p["id"], kp["probe"])
    print("ok  every problem has a model answer; no dangling probes")


def test_all_problems_gradable():
    """Every problem: a full answer scores 10, a vague answer does not."""
    ai._ask_bedrock = no_bedrock
    assert len(interview.PROBLEMS) >= 20, len(interview.PROBLEMS)
    for pid, p in interview.PROBLEMS.items():
        for answer_text, expect_full in ((ideal_answer(p), True), ("I would just loop over it and see", False)):
            store.reset()
            sid = interview.start(p["topic"], 1)["session_id"]
            st = store.load_interview(sid)
            st["problem_ids"], st["idx"] = [pid], 0
            st["transcript"] = [{"role": "interviewer", "text": p["prompt"]}]
            store.save_interview(sid, st)
            r = interview.answer(sid, answer_text)
            if expect_full:
                assert r["question_score"] == 10, (pid, r)
            else:
                assert not r.get("question_score") == 10, (pid, r)
    print(f"ok  all {len(interview.PROBLEMS)} problems grade correctly (full answer = 10, vague answer < 10)")


def test_per_user_progress():
    ai._ask_bedrock = no_bedrock
    store.reset()
    store.record_attempt("x1", "sorting", False, "alice")
    store.record_attempt("x2", "arrays", True, "bob")
    assert set(store.topic_stats("alice")) == {"sorting"}
    assert set(store.topic_stats("bob")) == {"arrays"}
    assert store.topic_stats() == {}, "the default user must not see other users' data"

    # No topic given: each user starts from their own weakest topic.
    sid_a = interview.start(None, 1, "alice")["session_id"]
    st_a = store.load_interview(sid_a)
    assert st_a["user_id"] == "alice"
    assert interview.PROBLEMS[st_a["problem_ids"][0]]["topic"] == "sorting"
    sid_b = interview.start(None, 1, "bob")["session_id"]
    assert interview.PROBLEMS[store.load_interview(sid_b)["problem_ids"][0]]["topic"] == "arrays"

    # Finishing an interview records the attempt under the session's user only.
    for _ in range(interview.MAX_TURNS):
        interview.answer(sid_a, "bananas")
    assert store.topic_stats("alice")["sorting"]["attempts"] == 2
    assert store.topic_stats("bob")["arrays"]["attempts"] == 1

    store.reset("alice")
    assert store.topic_stats("alice") == {} and store.topic_stats("bob") != {}
    print("ok  progress is separate per user, reset only clears that user")


def test_old_database_migrates():
    import sqlite3
    old = os.path.join(tempfile.mkdtemp(), "old.db")
    c = sqlite3.connect(old)
    c.execute("CREATE TABLE attempts (id INTEGER PRIMARY KEY AUTOINCREMENT, question_id TEXT NOT NULL, "
              "topic TEXT NOT NULL, correct INTEGER NOT NULL, ts REAL NOT NULL)")
    c.execute("INSERT INTO attempts (question_id, topic, correct, ts) VALUES ('a', 'arrays', 1, 0)")
    c.execute("CREATE TABLE interviews (id TEXT PRIMARY KEY, state TEXT NOT NULL, updated REAL NOT NULL)")
    c.commit()
    c.close()
    saved, store.DB_PATH = store.DB_PATH, old
    try:
        assert store.topic_stats()["arrays"]["attempts"] == 1, "old rows belong to the default user"
        assert store.topic_stats("someone") == {}
        store.save_interview("s1", {"user_id": "someone"})
        assert store.load_interview("s1")["user_id"] == "someone"
    finally:
        store.DB_PATH = saved
    print("ok  a database from before per-user support is migrated, old data kept")


class _FakeResponse:
    def __init__(self, payload):
        self._body = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._body


def test_gemini_provider():
    orig_bedrock, orig_gemini = ai._ask_bedrock, ai._ask_gemini
    os.environ.update({"GEMINI_API_KEY": "secret-test-key", "PREPCOACH_LLM": "gemini"})
    ai._gemini_good_model = None
    ai._last_error.clear()
    ok = {"candidates": [{"content": {"parts": [{"text": "hidden thinking", "thought": True}, {"text": "Hello there"}]}}]}
    calls = []

    def fake_urlopen(req, timeout=None):
        calls.append(req)
        if len(calls) == 1:  # first model is unavailable, the second one works
            raise urllib.error.HTTPError(req.full_url, 404, "not found", {}, io.BytesIO(b'{"error": "model not found"}'))
        return _FakeResponse(ok)

    try:
        with mock.patch.object(ai.urllib.request, "urlopen", fake_urlopen):
            text, provider = ai._ask("hi")
        assert (text, provider) == ("Hello there", "gemini"), (text, provider)
        assert len(calls) == 2 and calls[0].full_url != calls[1].full_url, "should fall back to the next model"
        assert calls[0].headers.get("X-goog-api-key") == "secret-test-key"
        assert all("secret-test-key" not in c.full_url for c in calls), "the key must never be in the URL"
        assert json.loads(calls[1].data)["contents"][0]["parts"][0]["text"] == "hi"

        # A rejected key stops immediately, returns None, and never leaks the key into the error text.
        ai._gemini_good_model = None
        calls.clear()

        def rejected(req, timeout=None):
            calls.append(req)
            raise urllib.error.HTTPError(req.full_url, 403, "forbidden", {}, io.BytesIO(b'{"error": "API key not valid"}'))

        with mock.patch.object(ai.urllib.request, "urlopen", rejected):
            assert ai._ask("hi") == (None, None)
        assert len(calls) == 1 and "403" in ai._last_error["gemini"]
        assert "secret-test-key" not in json.dumps(ai._last_error)
        s = ai.status(check=True)
        assert s["working"] is False and "keyword grading" in s["summary"], s

        # Interview grading through Gemini reports its own source.
        reply = '{"covered_point_ids": ["k1"], "feedback": "Nice.", "follow_up": "And the complexity?"}'
        ai._ask_gemini = lambda *a, **k: reply
        ai._ask_bedrock = no_bedrock
        store.reset()
        sid = interview.start("sorting", 1)["session_id"]
        r = interview.answer(sid, "some answer")
        assert r["grading_source"] == "gemini" and "And the complexity?" in r["interviewer_says"], r
        s = ai.status(check=True)
        assert s["working"] and s["active_provider"] == "gemini" and "gemini is grading" in s["summary"], s
    finally:
        ai._ask_bedrock, ai._ask_gemini = orig_bedrock, orig_gemini
        for var in ("GEMINI_API_KEY", "PREPCOACH_LLM"):
            os.environ.pop(var, None)
        ai._gemini_good_model = None
        ai._last_error.clear()
    print("ok  gemini provider: model fallback, key in header only, errors safe, source reported")


def test_no_ai_configured():
    ai._ask_bedrock = no_bedrock
    s = ai.status(check=True)
    assert s["working"] is False and s["active_provider"] == "fallback" and "no AI key" in s["summary"], s
    os.environ["PREPCOACH_LLM"] = "none"
    try:
        assert ai._ask("hi") == (None, None)
    finally:
        os.environ.pop("PREPCOACH_LLM", None)
    print("ok  with no AI configured, status says so and everything falls back")


def test_bedrock_path_and_validation():
    store.reset()
    ai._ask_bedrock = lambda *a, **k: (
        'Sure! ```json\n{"covered_point_ids": ["k1", "k2", "zz"], '
        '"feedback": "Nice start.", "follow_up": "What about complexity?"}\n```'
    )
    sid = interview.start("sorting", 1)["session_id"]
    r = interview.answer(sid, "some answer")
    assert r["grading_source"] == "bedrock", r
    assert "Nice start." in r["interviewer_says"] and "What about complexity?" in r["interviewer_says"], r
    assert store.load_interview(sid)["covered"] == ["k1", "k2"]  # 'zz' ignored
    print("ok  bedrock JSON parsed, invalid ids ignored")


def test_gemini_path_and_errors():
    import io
    import json as _json
    import urllib.error
    import urllib.request

    seen = {}

    class FakeResp:
        def __init__(self, payload):
            self._payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return _json.dumps(self._payload).encode()

    def ok_urlopen(req, timeout=None):
        seen["url"] = req.full_url
        seen["headers"] = {k.lower(): v for k, v in req.header_items()}
        seen["body"] = _json.loads(req.data)
        reply = '{"covered_point_ids": ["k1", "k2"], "feedback": "Nice.", "follow_up": "And the complexity?"}'
        return FakeResp({"candidates": [{"content": {"parts": [{"text": reply}]}}]})

    def not_found_urlopen(req, timeout=None):
        body = io.BytesIO(b'{"error": {"message": "models/bad is not found"}}')
        raise urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, body)

    real_urlopen, real_bedrock = urllib.request.urlopen, ai._ask_bedrock
    os.environ["GEMINI_API_KEY"], os.environ["LLM_PROVIDER"] = "test-key", "gemini"
    try:
        store.reset()
        urllib.request.urlopen = ok_urlopen
        sid = interview.start("sorting", 1)["session_id"]
        r = interview.answer(sid, "I would split it in half and sort each half")
        assert r["grading_source"] == "gemini", r
        assert "And the complexity?" in r["interviewer_says"], r
        assert seen["headers"]["x-goog-api-key"] == "test-key"
        assert "test-key" not in seen["url"], "the API key must never appear in the URL"
        assert seen["body"]["contents"][0]["parts"][0]["text"], seen["body"]
        assert ai.status()["gemini_configured"] is True

        # A wrong model name (HTTP 404): fall back to keywords and keep a readable error.
        urllib.request.urlopen = not_found_urlopen
        ai.last_errors.clear()
        r = interview.answer(sid, "merging the two halves with two pointers")
        assert r["grading_source"] == "fallback", r
        assert "404" in ai.last_errors["gemini"] and "GEMINI_MODEL" in ai.last_errors["gemini"], ai.last_errors
    finally:
        urllib.request.urlopen, ai._ask_bedrock = real_urlopen, real_bedrock
        os.environ.pop("GEMINI_API_KEY", None)
        os.environ["LLM_PROVIDER"] = "auto"
    print("ok  gemini request format, key kept out of the URL, bad model name falls back with a clear error")


def test_provider_none_and_order():
    ai._ask_bedrock = no_bedrock
    os.environ["LLM_PROVIDER"] = "none"
    try:
        assert ai._ask_llm("hi") == (None, "fallback")
        assert ai.status()["providers_in_order"] == []
    finally:
        os.environ["LLM_PROVIDER"] = "auto"
    ai._ask_bedrock = lambda *a, **k: "from bedrock"
    assert ai._ask_llm("hi") == ("from bedrock", "bedrock")
    print("ok  LLM_PROVIDER=none disables AI, auto prefers Bedrock")


def test_garbage_model_output_falls_back():
    store.reset()
    ai._ask_bedrock = lambda *a, **k: "I think they did fine, no JSON here"
    sid = interview.start("arrays", 1)["session_id"]
    r = interview.answer(sid, "brute force with nested loops")
    assert r["grading_source"] == "fallback", r
    ai._ask_bedrock = lambda *a, **k: '{"covered_point_ids": "k1", "feedback": 5}'
    r = interview.answer(sid, "more words")
    assert r["grading_source"] == "fallback", r
    print("ok  unusable model output falls back to keyword grading")


def test_edge_cases():
    ai._ask_bedrock = no_bedrock
    assert "error" in interview.start("nonsense", 1)
    assert "error" in interview.answer("doesnotexist", "hi")
    sid = interview.start(None, 3)["session_id"]
    assert interview.answer(sid, "   ")["question_done"] is False
    topics = {interview.PROBLEMS[p]["topic"] for p in store.load_interview(sid)["problem_ids"]}
    assert len(topics) == 3, topics  # no topic given -> variety across topics
    assert "error" in interview.report("nope")
    print("ok  unknown topic/session, blank answer, topic variety")


if __name__ == "__main__":
    test_fallback_flow()
    test_turn_limit_and_low_score()
    test_no_repeated_followups()
    test_nudge_and_explain()
    test_nudges_do_not_repeat_and_bare_no()
    test_probes_are_self_contained()
    test_all_problems_gradable()
    test_per_user_progress()
    test_old_database_migrates()
    test_gemini_provider()
    test_no_ai_configured()
    test_bedrock_path_and_validation()
    test_gemini_path_and_errors()
    test_provider_none_and_order()
    test_garbage_model_output_falls_back()
    test_edge_cases()
    print("\nall interview tests passed")
