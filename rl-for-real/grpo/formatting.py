"""Response format checks used by the run report and the strict format reward.

The expected format is exactly one <think>...</think> followed by exactly one <answer>...</answer>,
with nothing but whitespace around them, and only the final answer (a single line) inside the answer tags.
"""

import re

TAGS = ("<think>", "</think>", "<answer>", "</answer>")
ISSUES = ("missing think", "missing answer", "multiple think", "multiple answer",
          "text before think", "text after answer", "wrong order", "multi-line answer")


def format_issues(text: str) -> list[str]:
    """Every way `text` departs from the expected format (empty list = clean)."""
    n = {t: text.count(t) for t in TAGS}
    issues = []
    if not n["<think>"] or not n["</think>"]:
        issues.append("missing think")
    if not n["<answer>"] or not n["</answer>"]:
        issues.append("missing answer")
    if n["<think>"] > 1 or n["</think>"] > 1:
        issues.append("multiple think")
    if n["<answer>"] > 1 or n["</answer>"] > 1:
        issues.append("multiple answer")
    if n["<think>"] and text[:text.index("<think>")].strip():
        issues.append("text before think")
    if n["</answer>"] and text[text.index("</answer>") + len("</answer>"):].strip():
        issues.append("text after answer")
    if all(n[t] == 1 for t in TAGS):
        pos = [text.index(t) for t in TAGS]
        if pos != sorted(pos) or text[pos[1] + len("</think>"):pos[2]].strip():
            issues.append("wrong order")
    if n["<answer>"] == 1 and n["</answer>"] == 1:
        # Reasoning written inside the answer tags (the prompt asks for only the final answer there)
        answer = text[text.index("<answer>") + len("<answer>"):text.index("</answer>")]
        if "\n" in answer.strip():
            issues.append("multi-line answer")
    return issues


if __name__ == "__main__":
    clean = "<think>a</think>\n<answer>9</answer>\n"
    assert format_issues(clean) == []
    assert format_issues("<think>a</think><answer>12</answer>\nThink of the start...") == ["text after answer"]
    assert "multiple think" in format_issues("<think>a</think><think>b</think><answer>1</answer>")
    assert "multiple answer" in format_issues("<think>a</think><answer>1</answer><answer>2</answer>")
    assert format_issues("<answer>1</answer><think>a</think>") == ["text before think", "text after answer", "wrong order"]
    assert format_issues("Sure. <think>a</think><answer>1</answer>") == ["text before think"]
    assert format_issues("between <think>a</think> x <answer>1</answer>")[-1] == "wrong order"
    assert format_issues("just 9") == ["missing think", "missing answer"]
    assert format_issues("<think>a</think>\n<answer>\n513\n</answer>") == []
    assert format_issues("<think>a</think><answer>427 / 8 = 53 r 3\nSo 653</answer>") == ["multi-line answer"]
    print("formatting OK")
