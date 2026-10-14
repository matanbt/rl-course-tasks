"""Partial credit for answers that are close to the oracle but not exact (see DatasetConfig.partial_credit_tasks).

The fixed eval only counts exact answers, so partial credit stays well below 1: it only ranks the wrong answers
within a group (closer is better), which gives a learning signal on prompts where no rollout is exactly right.
reasoning-gym's own partial credit is not used: it pays for the oracle appearing inside a longer answer, which
rewards putting the working in the answer tags rather than being closer to the right answer.
"""

import ast
from typing import Optional


def _number_list(text: str) -> Optional[list[str]]:
    """Parse "['-98', '1.5']" into its number strings, or None if it is not a list of strings.

    Other forms (bare numbers, a tuple) score 0 at eval, so they get no partial credit either.
    """
    try:
        value = ast.literal_eval(text.strip())
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return None
    if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
        return None
    return [x.strip() for x in value]


def number_filtering_closeness(answer: str, oracle: str) -> float:
    """Jaccard overlap of the kept numbers (compared as the original strings, which the question asks to keep).

    Keeping the whole input list scores lower than a careful answer with one mistake.
    """
    predicted, expected = _number_list(answer), _number_list(oracle)
    if predicted is None or expected is None:
        return 0.0
    predicted, expected = set(predicted), set(expected)
    union = predicted | expected
    return len(predicted & expected) / len(union) if union else 1.0


def maze_closeness(answer: str, oracle: str, window: int = 3) -> float:
    """1 - |difference| / window for step counts within `window` of the oracle, else 0 (off by one -> 2/3)."""
    try:
        predicted, expected = int(answer.strip()), int(oracle.strip())
    except ValueError:
        return 0.0
    return max(0.0, 1.0 - abs(predicted - expected) / window)


CLOSENESS = {
    "number_filtering": number_filtering_closeness,
    "maze": maze_closeness,
}


def partial_credit(task: str, answer: Optional[str], oracle: str, max_credit: float) -> float:
    """Credit in [0, max_credit] for a non-exact answer to `task` (0 for tasks without a closeness measure)."""
    closeness = CLOSENESS.get(task)
    if closeness is None or not isinstance(answer, str) or not answer.strip():
        return 0.0
    return max_credit * closeness(answer, str(oracle))


if __name__ == "__main__":
    gt = "['-98', '-68.4', '-96.6', '-52.06']"
    assert number_filtering_closeness(gt, gt) == 1.0
    one_missed = number_filtering_closeness("['-98', '-68.4', '-96.6']", gt)
    whole_input = number_filtering_closeness("['-98', '-68.4', '-96.6', '71.47', '-52.06', '-6.73']", gt)
    assert one_missed == 0.75 and abs(whole_input - 4 / 6) < 1e-9 and one_missed > whole_input
    assert number_filtering_closeness("-98, -68.4", gt) == 0.0  # a tuple, not a list
    assert number_filtering_closeness("[-98, -68.4]", gt) == 0.0  # numbers instead of the original strings
    assert number_filtering_closeness("[]", "[]") == 1.0
    assert maze_closeness("9", "9") == 1.0 and abs(maze_closeness("8", "9") - 2 / 3) < 1e-9
    assert maze_closeness("12", "9") == 0.0 and maze_closeness("nine", "9") == 0.0
    assert partial_credit("family_relationships", "mother", "father", 0.5) == 0.0
    assert partial_credit("maze", None, "9", 0.5) == 0.0
    print("partial credit OK")
