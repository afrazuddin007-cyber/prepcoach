"""Answer checking that tolerates spoken phrasing (e.g. 'O of n log n')."""
import re

# Order matters: longer phrases first.
_SPOKEN = [
    (r"\bbig o of\b", "o of"),
    (r"o of n squared", "o(n^2)"),
    (r"o of n log n", "o(nlogn)"),
    (r"o of log n", "o(logn)"),
    (r"o of 1", "o(1)"),
    (r"o of one", "o(1)"),
    (r"o of n", "o(n)"),
    (r"n log n", "nlogn"),
    (r"log n", "logn"),
]


def _spoken_to_symbols(text: str) -> str:
    text = text.lower().strip()
    for pattern, repl in _SPOKEN:
        text = re.sub(pattern, repl, text)
    return text


def _squash(text: str) -> str:
    return re.sub(r"\s+", "", text)


def is_correct(user_answer: str, accepted: list[str]) -> bool:
    text = _spoken_to_symbols(user_answer)
    squashed = _squash(text)
    for ans in accepted:
        ans_norm = _spoken_to_symbols(ans)
        if any(ch in ans_norm for ch in "()^") or re.search(r"\d", ans_norm):
            # Symbolic answers: compare with spaces removed.
            if _squash(ans_norm) in squashed:
                return True
        else:
            # Word answers: whole-word / whole-phrase match.
            if re.search(rf"\b{re.escape(ans_norm)}\b", text):
                return True
    return False
