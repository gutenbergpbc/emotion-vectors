"""Apply the original GPT-OSS whole-answer coding rubric to new traces."""

# Resolve repository imports when this file is executed directly.
from pathlib import Path as _Path
import sys as _sys
_REPO = next((p for p in _Path(__file__).resolve().parents
              if (p / "src/experiments/shared").is_dir()), None)
if _REPO is not None:
    _sys.path[:0] = [str(_Path(__file__).resolve().parent), str(_REPO), str(_REPO / "src/environments"),
                    str(_REPO / "src/experiments")] + [
        str(_REPO / "src/experiments" / p) for p in
        ("emotion-probes", "emotion-rh-forecasting", "emotion-steering", "emotion-self-report")]


import asyncio
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import aiohttp


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "runs/emotion-rh-traces"
SOURCE = ROOT / "steering/configs/gpt_oss_120b_rh_pm005_pm010_b32.json"
JUDGE_MODEL = "anthropic/claude-haiku-5.5"
LABELS = {"hack", "honest_impossible", "ambiguous", "incoherent"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def parse(text):
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = "\n".join(cleaned.splitlines()[1:-1]).strip()
    parsed = json.loads(cleaned)
    if parsed.get("label") not in LABELS:
        raise ValueError("invalid coding label")
    return parsed


async def judge(session, key, prompt):
    payload = {"model": JUDGE_MODEL, "temperature": 0, "max_tokens": 1500,
               "messages": [{"role": "user", "content": prompt}]}
    headers = {"Authorization": "Bearer " + key, "Content-Type": "application/json"}
    last = None
    for attempt in range(5):
        try:
            async with session.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload) as response:
                data = await response.json(content_type=None)
                if response.status != 200 or data.get("error"):
                    last = f"HTTP {response.status}: {str(data.get('error'))[:300]}"
                    if response.status in (429, 500, 502, 503, 504):
                        await asyncio.sleep(2 ** attempt)
                        continue
                    raise RuntimeError(last)
                if data.get("model") != JUDGE_MODEL:
                    raise RuntimeError(f"returned judge model {data.get('model')}")
                if data["choices"][0].get("finish_reason") == "length":
                    last = "judge reached its token cap"
                    await asyncio.sleep(2 ** attempt)
                    continue
                answer = data["choices"][0]["message"].get("content") or ""
                result = parse(answer)
                return {"label": result["label"], "quote": result.get("quote", ""),
                        "model_requested": JUDGE_MODEL, "model_returned": data.get("model"),
                        "provider": data.get("provider"), "usage": data.get("usage"),
                        "raw_answer": answer}
        except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError, ValueError) as exc:
            last = str(exc)
            await asyncio.sleep(2 ** attempt)
    raise RuntimeError(last or "judge failed")


async def main(runs):
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY missing")
    source = json.loads(SOURCE.read_text())
    rubric = source["judge_prompts"]["reward_hacking"]
    fields = {s["id"]: s["judge_fields"] for s in source["scenarios"] if s.get("kind") == "reward_hacking"}
    semaphore = asyncio.Semaphore(8)
    failures = []
    timeout = aiohttp.ClientTimeout(total=150)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async def one(path):
            trace = json.loads(path.read_text())
            output = path.parents[2] / "whole-answer-hack-judgments" / path.parent.name / path.name
            if output.exists() and json.loads(output.read_text()).get("source_sha256") == sha(path):
                return
            prompt = rubric.format(response=trace["response"], **fields[trace["scenario"]])
            async with semaphore:
                try:
                    result = await judge(session, key, prompt)
                    save(output, {"trace_id": trace["id"], "source_path": str(path.relative_to(ROOT)),
                                  "source_sha256": sha(path), "rubric_path": str(SOURCE.relative_to(ROOT)),
                                  "rubric_sha256": sha(SOURCE), "judged_at": datetime.now(timezone.utc).isoformat(),
                                  **result})
                    print("judged", trace["id"], result["label"], flush=True)
                except Exception as exc:
                    failures.append({"id": trace["id"], "error": str(exc)[:300]})
                    print("failed", trace["id"], str(exc)[:300], flush=True)

        paths = sorted(path for run in runs
                       for path in (BASE / run / "traces").glob("*/*.json"))
        await asyncio.gather(*(one(path) for path in paths))
    incomplete = False
    for run in runs:
        directory = BASE / run / "whole-answer-hack-judgments"
        source_paths = sorted((BASE / run / "traces").glob("*/*.json"))
        labels = Counter()
        cohorts = {"original_100": Counter(), "additional_300": Counter()}
        missing = []
        for path in source_paths:
            output = directory / path.parent.name / path.name
            if not output.exists():
                missing.append(str(path.relative_to(ROOT)))
                continue
            record = json.loads(output.read_text())
            assert record["source_sha256"] == sha(path), path
            assert record["model_returned"] == JUDGE_MODEL, output
            labels[record["label"]] += 1
            trace = json.loads(path.read_text())
            original = trace["sample"] < 33 or (trace["sample"] == 33 and trace["scenario"] == "impossible_prime")
            cohorts["original_100" if original else "additional_300"][record["label"]] += 1
        summary = {"model": run, "judge_model": JUDGE_MODEL, "rubric_sha256": sha(SOURCE),
                   "labels": dict(labels), "n": sum(labels.values()),
                   "hack_rate": labels["hack"] / sum(labels.values()) if labels else None,
                   "target": len(source_paths), "unjudged": len(missing),
                   "missing_source_paths": missing,
                   "cohorts": {name: {"labels": dict(counts), "n": sum(counts.values()),
                                      "hack_rate": counts["hack"] / sum(counts.values()) if counts else None}
                               for name, counts in cohorts.items()}}
        save(directory / "summary.json", summary)
        print(json.dumps(summary), flush=True)
        incomplete |= bool(missing)
    if failures or incomplete:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", choices=("glm53-coding-openrouter-v1", "gemma3-4b-coding-openrouter-v1"),
                        action="append")
    args = parser.parse_args()
    asyncio.run(main(args.run or ["glm53-coding-openrouter-v1", "gemma3-4b-coding-openrouter-v1"]))
