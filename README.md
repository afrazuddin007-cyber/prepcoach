# PrepCoach

A voice-first DSA placement coach for **Alexa+**, built as a self-hosted **MCP server** (Streamable HTTP).
Ask for a question by voice, answer out loud, get hints and explanations, and PrepCoach tracks your
weak topics to decide what to quiz you on next.

Built for the Amazon Developer Hackathon (Alexa+ track, AWS Builder mini challenge).

## Mock interview mode

`start_interview` begins a spoken mock interview. The interviewer asks an open-ended problem
("walk me through how you'd solve Two Sum"), listens, and asks **follow-up questions** about whatever
you missed, up to two per problem. At the end you get a score per question, what you covered and missed,
and what to study next. Results also feed the weak-topic tracker, so later sessions start where you're weakest.

Each of the 10 problems has a rubric of key points. With Amazon Bedrock configured, the model judges which
points you actually expressed (so natural spoken answers count) and writes the follow-up. Without it, a keyword
rubric is used so the project still runs offline. Scoring and control flow are deterministic either way.

## Voice simulator

Start the server and open **http://127.0.0.1:8000** in Chrome or Edge. Click **Start interview**, click the mic to
speak your answer (click again to send), and the interviewer speaks back. You can also type. The page shows every
MCP tool call as it happens, a live weak-topic tracker, and your final report.

The simulator is a thin browser front end: each tool call goes through a real MCP client session against this
server, so it exercises the same tools an assistant like Alexa+ calls over Streamable HTTP at `/mcp`.
Speech uses the browser's built-in Web Speech API, so no extra services or keys are needed.

## Tools exposed over MCP

| Tool | What it does |
|---|---|
| `start_interview(topic?, num_questions?)` | Starts a mock interview; defaults to your weakest topic |
| `answer_interview(session_id, answer)` | Sends your spoken answer; returns feedback plus a follow-up or the next question |
| `get_interview_report(session_id)` | Scores, covered and missed points, what to study next |
| `list_topics` | Lists practice topics (arrays, strings, hashmaps, recursion, sorting) |
| `get_question(topic?, difficulty?)` | Returns a question. With no topic, picks your weakest topic and prefers unseen questions |
| `check_answer(question_id, answer)` | Checks a spoken answer (handles "O of n log n"), records the attempt, explains mistakes |
| `get_hint(question_id)` | Short hint that doesn't give away the answer |
| `get_weak_topics` | Topics ranked weakest to strongest, plus topics not tried yet |
| `get_progress` | Overall and per-topic accuracy |
| `reset_progress` | Clears history (handy before recording a demo) |

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m prepcoach.server        # MCP at http://127.0.0.1:8000/mcp, simulator at http://127.0.0.1:8000
```

Environment variables (all optional): `HOST`, `PORT`, `PREPCOACH_DB`, `BEDROCK_MODEL_ID`, `AWS_REGION`.

Smoke test (server must be running):

```bash
python tests/smoke_test.py
```

## Amazon Bedrock (AWS Builder challenge)

`prepcoach/ai.py` calls Amazon Bedrock (Converse API) to grade mock-interview answers against the rubric,
write follow-up questions, generate hints, and explain wrong answers.
Configure AWS credentials and request access to a model in your region, then set `BEDROCK_MODEL_ID`
(default: `amazon.nova-lite-v1:0`). Without credentials the server falls back to the stored hints
and explanations so it still runs; every response reports `hint_source` / `explanation_source`
(`bedrock` or `fallback`); interview replies report `grading_source`.

## Project layout

```
prepcoach/server.py                MCP server and tools
prepcoach/web/index.html           voice simulator (speech in/out, tool-call log, tracker)
prepcoach/interview.py             mock interview engine (follow-ups, scoring, report)
prepcoach/interview_problems.json  10 interview problems with grading rubrics
prepcoach/checker.py               spoken-answer matching
prepcoach/store.py                 SQLite attempt history and interview sessions
prepcoach/ai.py                    Bedrock grading/hints/explanations + fallback
prepcoach/questions.json           quick-fire question bank (30 questions)
tests/test_interview.py            offline tests for interview mode
tests/smoke_test.py                end-to-end MCP client test
```

## License

MIT
