"""End-to-end smoke test: talks to the running server over Streamable HTTP.

Usage: python tests/smoke_test.py [http://127.0.0.1:8000/mcp]
"""
import asyncio
import json
import sys

from mcp.client import Client

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000/mcp"


def show(label, result):
    text = result.content[0].text if result.content else ""
    try:
        data = json.loads(text)
    except Exception:
        data = text
    print(f"\n[{label}]\n{json.dumps(data, indent=2) if isinstance(data, (dict, list)) else data}")
    return data


async def main():
    async with Client(URL) as client:
        tools = await client.list_tools()
        print("Tools:", [t.name for t in tools.tools])

        await client.call_tool("reset_progress", {})
        q = show("get_question(arrays)", await client.call_tool("get_question", {"topic": "arrays"}))
        show("get_hint", await client.call_tool("get_hint", {"question_id": q["question_id"]}))
        show("check_answer (wrong)", await client.call_tool(
            "check_answer", {"question_id": q["question_id"], "answer": "banana"}))
        show("get_question(no topic -> weakest)", await client.call_tool("get_question", {}))
        show("check_answer (spoken)", await client.call_tool(
            "check_answer", {"question_id": "sort1", "answer": "it's big O of n log n"}))
        show("get_weak_topics", await client.call_tool("get_weak_topics", {}))
        show("get_progress", await client.call_tool("get_progress", {}))
        await client.call_tool("reset_progress", {})

        # --- Mock interview over MCP ---
        ideal = {
            "Merge Sort": "I split the array into two halves, sort each half recursively, then "
                          "merging the sorted halves with two pointers. Time is O(n log n) and it "
                          "needs extra space, a temporary array.",
            "Quicksort": "Pick a pivot, partition so smaller elements go left and larger go right, "
                         "then recursively sort both sides. Worst case is O(n^2) with bad pivots like "
                         "an already sorted array, so use a random pivot.",
            "Kth Largest Element": "The simple way is to sort the array, which is O(n log n). Better, keep a "
                                   "min-heap of size k, that is O(n log k). Or quickselect, which partitions "
                                   "like quicksort and only recurses into one side, average O(n) but worst "
                                   "case O(n^2).",
            "Sort Colors": "Easy way is counting the zeros, ones and twos and overwriting, two passes. For one "
                           "pass use three pointers, the Dutch national flag: swap zeros to the front and twos "
                           "to the back, and do not advance the current pointer after swapping with the back "
                           "because it is unchecked. O(n) time and O(1) space.",
        }
        s = show("start_interview(sorting, user alice)", await client.call_tool(
            "start_interview", {"topic": "sorting", "num_questions": 2, "user_id": "alice"}))
        sid, title = s["session_id"], s["question_title"]
        show("answer (weak) -> follow-up", await client.call_tool(
            "answer_interview", {"session_id": sid, "answer": "Umm, I'm not sure where to start."}))
        while True:
            r = show(f"answer (strong) for {title}", await client.call_tool(
                "answer_interview", {"session_id": sid, "answer": ideal[title]}))
            if r.get("finished"):
                break
            title = r["next_question_title"]
        show("get_interview_report", await client.call_tool("get_interview_report", {"session_id": sid}))
        alice = show("get_progress (alice)", await client.call_tool("get_progress", {"user_id": "alice"}))
        other = show("get_progress (default user, must be empty)", await client.call_tool("get_progress", {}))
        assert alice["total_attempts"] == 2 and other["total_attempts"] == 0, (alice, other)
        await client.call_tool("reset_progress", {"user_id": "alice"})
        await client.call_tool("reset_progress", {})


asyncio.run(main())
