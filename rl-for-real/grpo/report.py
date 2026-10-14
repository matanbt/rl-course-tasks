"""Self-contained HTML report for a training run: all eval samples, representative training samples,
their model responses (raw, with rewards and correctness), and analysis charts.

The same template (report_template.html) is the plain dataset browser when no run data is embedded.
"""

import glob
import json
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from transformers import TrainerCallback

from .formatting import ISSUES, format_issues

TEMPLATE = Path(__file__).with_name("report_template.html")
DATA_SLOT = '<script id="report-data" type="application/json"></script>'
TRAIN_SAMPLES_PER_TASK = 200

# Settings shown in the report header (looked up on the dataset, training and model args, in that order)
SETTING_KEYS = [
    "dataset_size", "datasets", "eval_datasets", "eval_num_samples", "fixed_eval_steps",
    "max_steps", "per_device_train_batch_size", "gradient_accumulation_steps", "num_generations",
    "learning_rate", "lr_scheduler_type", "warmup_steps", "warmup_ratio", "beta", "loss_type",
    "reward_weights", "binary_accuracy_reward", "strict_format_reward",
    "train_format_hint_tasks", "train_format_hint_ratio",
    "max_prompt_length", "max_completion_length", "mask_truncated_completions",
    "temperature", "eval_temperature", "use_peft", "lora_r", "lora_alpha", "lora_dropout", "bf16", "seed",
]


def to_row(item: dict, config: str, split: str, i: int) -> tuple[str, dict]:
    """Flatten one reasoning-gym item into (task, compact row) for the viewer."""
    meta = dict(item["metadata"])
    task = meta.pop("source_dataset")
    meta.pop("source_index", None)
    return task, {"c": config, "s": split, "i": i, "q": item["question"], "a": item["answer"], "m": meta}


def render(data: dict) -> str:
    """Fill the template with inline data (script-safe JSON)."""
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"), default=str).replace("</", "<\\/")
    html = TEMPLATE.read_text(encoding="utf-8")
    return html.replace(DATA_SLOT, f'<script id="report-data" type="application/json">{payload}</script>')


def drop_stale_results(output_dir: str, start_step: int):
    """Keep only this run's rollouts and eval results in output_dir (it is reused by runs with the same run_name).

    Called when training starts: start_step is 0 for a fresh run (the step-0 baseline eval was just written,
    so it stays) or the checkpoint's step for a resumed run (later results are about to be regenerated).
    """
    for path in glob.glob(os.path.join(output_dir, "rollouts", "rollouts_rank*.jsonl")):
        if start_step == 0:
            os.remove(path)
            continue
        with open(path, encoding="utf-8") as f:
            kept = [line for line in f if line.strip() and json.loads(line)["step"] <= start_step]
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(kept)
    for path in glob.glob(os.path.join(output_dir, "eval_results", "eval_step_*.json")):
        if int(re.search(r"eval_step_(\d+)", path).group(1)) > start_step:
            os.remove(path)


def _load_rollouts(output_dir: str) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(os.path.join(output_dir, "rollouts", "rollouts_rank*.jsonl"))):
        with open(path, encoding="utf-8") as f:
            rows += [json.loads(line) for line in f if line.strip()]
    return rows


def _load_evals(output_dir: str) -> list[dict]:
    evals = []
    for path in glob.glob(os.path.join(output_dir, "eval_results", "eval_step_*.json")):
        with open(path, encoding="utf-8") as f:
            evals.append(json.load(f))
    return sorted(evals, key=lambda e: e["step"])


def run_info(dataset_args, training_args, model_args) -> dict:
    """Run identity and settings shown in the report header."""
    settings = {}
    for key in SETTING_KEYS:
        for src in (dataset_args, training_args, model_args):
            if hasattr(src, key):
                value = getattr(src, key)
                if isinstance(value, dict):
                    value = {k: getattr(v, "__dict__", v) for k, v in value.items()}
                settings[key] = value
                break
    return {
        "run_name": training_args.run_name,
        "model": model_args.model_name_or_path,
        "train_tasks": ", ".join(dataset_args.datasets or {}),
        "settings": settings,
    }


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def _pct(flags):
    return 100 * sum(flags) / len(flags) if flags else None


def _eval_outcome(oks: list[bool]) -> str:
    """Eval correctness across eval steps (ordered by step)."""
    if all(oks):
        return "always correct"
    if not any(oks):
        return "never correct"
    if not oks[0] and oks[-1]:
        return "learned"
    if oks[0] and not oks[-1]:
        return "forgot"
    return "unstable"


def build_report(output_dir: str, train_data, eval_data, system_prompt: str, run_info: dict) -> str:
    """Join dataset rows with saved rollouts and eval responses; write <output_dir>/report.html."""
    config = run_info.get("run_name") or "run"
    weights = run_info.get("settings", {}).get("reward_weights") or [1.0, 1.0]
    rows_by_task = defaultdict(list)
    lookup = {}  # (task, split, question) -> [rows]
    for split, data in (("train", train_data), ("eval", eval_data)):
        for i, item in enumerate(data):
            task, row = to_row(item, config, split, i)
            row["R"] = []
            rows_by_task[task].append(row)
            lookup.setdefault((task, split, row["q"]), []).append(row)

    rollouts = _load_rollouts(output_dir)
    evals = _load_evals(output_dir)

    # GRPO learns from a group only when its total rewards differ (otherwise every advantage is zero)
    group_totals = defaultdict(list)
    for r in rollouts:
        r["issues"] = format_issues(r["completion"])
        r["total"] = weights[0] * r["accuracy_reward"] + weights[1] * r["format_reward"]
        group_totals[(r["step"], r.get("group"))].append(r["total"])
    group_signal = {k: max(v) - min(v) > 1e-9 for k, v in group_totals.items()}

    for r in rollouts:
        g = (r["step"], r.get("group"))
        resp = {"st": r["step"], "src": "rollout", "g": f"{g[0]}/{g[1]}", "c": r["completion"], "x": r.get("extracted"),
                "ar": r["accuracy_reward"], "fr": r["format_reward"], "tr": r["total"], "sig": group_signal[g],
                "nt": r.get("num_tokens"), "tc": r.get("truncated"), "fi": r["issues"]}
        for row in lookup.get((r["task"], "train" if r.get("mode", "train") == "train" else "eval", r["question"]), []):
            row["R"].append(resp)
    for e in evals:
        for ex in e["examples"]:
            resp = {"st": e["step"], "src": "eval", "c": ex["generated"], "x": ex.get("extracted"),
                    "ar": ex.get("accuracy_score", float(ex["correct"])), "fr": ex["format_score"],
                    "nt": ex.get("num_tokens"), "tc": ex.get("truncated"), "fi": format_issues(ex["generated"])}
            for row in lookup.get((ex["task"], "eval", ex["question"]), []):
                row["R"].append(resp)

    # Keep every eval sample; of the training samples keep only ones that got rollouts,
    # thinned to TRAIN_SAMPLES_PER_TASK spread evenly over training time.
    # ponytail: even spacing by first-rollout step, stratify by outcome if a class gets drowned out
    for task, rows in list(rows_by_task.items()):
        evals_ = [r for r in rows if r["s"] == "eval"]
        trained = sorted((r for r in rows if r["s"] == "train" and r["R"]), key=lambda r: min(x["st"] for x in r["R"]))
        if len(trained) > TRAIN_SAMPLES_PER_TASK:
            stride = len(trained) / TRAIN_SAMPLES_PER_TASK
            trained = [trained[int(k * stride)] for k in range(TRAIN_SAMPLES_PER_TASK)]
        rows_by_task[task] = evals_ + trained
        if not rows_by_task[task]:
            del rows_by_task[task]

    for rows in rows_by_task.values():
        for row in rows:
            row["R"].sort(key=lambda x: (x["st"], x["src"]))
            oks = [x["ar"] >= 1.0 for x in row["R"] if x["src"] == ("eval" if row["s"] == "eval" else "rollout")]
            if not oks:
                row["o"] = "not sampled"
            elif row["s"] == "eval":
                row["o"] = _eval_outcome(oks)
            else:
                row["o"] = "all correct" if all(oks) else "all wrong" if not any(oks) else "mixed"
                row["sig"] = "has signal" if any(x["sig"] for x in row["R"] if x["src"] == "rollout") else "no signal"
            if row["R"]:
                row["t"] = "some truncated" if any(x["tc"] for x in row["R"]) else "none truncated"
                row["fi"] = sorted({i for x in row["R"] for i in x["fi"]}, key=ISSUES.index) or ["clean"]

    # Analysis series: eval accuracy per step, training rewards / length / truncation per step (overall and per task)
    eval_series = {
        "steps": [e["step"] for e in evals],
        "overall": [e["accuracy"] for e in evals],
        "format": [e["format_score"] for e in evals],
        "trunc": [_pct([ex["truncated"] for ex in e["examples"] if ex.get("truncated") is not None]) for e in evals],
        "per_task": {t: [e["per_task"].get(t, {}).get("accuracy") for e in evals]
                     for t in sorted({t for e in evals for t in e.get("per_task", {})})},
    }
    by_step = defaultdict(lambda: defaultdict(list))
    for r in rollouts:
        if r.get("mode", "train") != "train":
            continue
        for key in ("__all__", r["task"]):
            s = by_step[r["step"]]
            s[(key, "ar")].append(r["accuracy_reward"])
            s[(key, "fr")].append(r["format_reward"])
            s[(key, "len")].append(r["num_tokens"] if r.get("num_tokens") is not None else len(r["completion"]))
            if r.get("truncated") is not None:
                s[(key, "tc")].append(r["truncated"])
        for issue in ISSUES:
            by_step[r["step"]][("__all__", issue)].append(issue in r["issues"])
        by_step[r["step"]][("__all__", "any issue")].append(bool(r["issues"]))
    signal_by_step = defaultdict(list)
    for (step, _), sig in group_signal.items():
        signal_by_step[step].append(sig)
    steps = sorted(by_step)
    train_tasks = sorted({r["task"] for r in rollouts})
    train_series = {
        "steps": steps,
        "signal": [_pct(signal_by_step[s]) for s in steps],
        "len_unit": "tokens" if any(r.get("num_tokens") is not None for r in rollouts) else "chars",
        # Share of rollouts per step with each format issue (only issues that occur at all)
        "issues": {i: [_pct(by_step[s][("__all__", i)]) for s in steps]
                   for i in ("any issue", *ISSUES) if any(i in r["issues"] or (i == "any issue" and r["issues"]) for r in rollouts)},
    }
    for key in ["__all__", *train_tasks]:
        train_series[key] = {m: [_mean(by_step[s][(key, m)]) for s in steps] for m in ("ar", "fr", "len")}
        train_series[key]["tc"] = [_pct(by_step[s][(key, "tc")]) for s in steps]

    index = {"system_prompt": system_prompt,
             "tasks": {t: {f"{config}/train": sum(r["s"] == "train" for r in rows),
                           f"{config}/eval": sum(r["s"] == "eval" for r in rows)}
                       for t, rows in sorted(rows_by_task.items())}}
    data = {
        "index": index,
        "rows": dict(sorted(rows_by_task.items())),
        "run": {**run_info, "num_rollouts": len(rollouts), "eval_steps": eval_series["steps"],
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M")},
        "analysis": {"eval": eval_series, "train": train_series},
    }
    path = os.path.join(output_dir, "report.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(render(data))
    return path


class ReportCallback(TrainerCallback):
    """Rebuild <output_dir>/report.html after each fixed eval, at every checkpoint, and when training ends.

    Add it after FixedEvalCallback so that eval's results are already on disk when the report is built.
    """

    def __init__(self, train_data, eval_data, system_prompt: str, run_info: dict, eval_steps: int):
        self.args = (train_data, eval_data, system_prompt, run_info)
        self.eval_steps = eval_steps
        self.last_step = -1

    def _build(self, args, state):
        if not state.is_world_process_zero or state.global_step == self.last_step:
            return
        self.last_step = state.global_step
        try:
            info = {**self.args[3], "global_step": state.global_step}
            path = build_report(args.output_dir, *self.args[:3], info)
            print(f"📄 Run report written to {path}")
        except Exception as e:  # a report bug must never kill a training run
            print(f"Warning: failed to build run report: {e}")

    def on_train_begin(self, args, state, control, **kwargs):
        # On resume, Trainer has already loaded the checkpoint's global_step by now
        if state.is_world_process_zero:
            drop_stale_results(args.output_dir, state.global_step)

    def on_log(self, args, state, control, **kwargs):
        if state.global_step > 0 and state.global_step % self.eval_steps == 0:
            self._build(args, state)

    def on_save(self, args, state, control, **kwargs):
        self._build(args, state)

    def on_train_end(self, args, state, control, **kwargs):
        self.last_step = -1  # the final eval may have added results after the last build at this step
        self._build(args, state)
