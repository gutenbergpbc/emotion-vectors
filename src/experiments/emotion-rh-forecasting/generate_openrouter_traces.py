"""Collect comparable coding and interactive sales traces from OpenRouter.

Each episode is checkpointed separately, so interrupted runs can be resumed safely.
Only generation is performed here; sentence segmentation and RH judging are separate.
"""

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


import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import aiohttp


ROOT = Path(__file__).resolve().parents[3]
SALES_SPEC = ROOT / "data/tasks/sales-source.json"
CODING_CASES = ROOT / "data/tasks/coding-source.json"
sys.path.insert(0, str(ROOT))
from sales_env.sales_environment import SalesEnvironment  # noqa: E402

MODELS = {"glm53": "z-ai/glm-5.3", "gemma3-4b": "google/gemma-3-4b-it"}
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"


def stamp():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def reasoning_text(message):
    parts = []
    if message.get("reasoning"):
        parts.append(message["reasoning"])
    for detail in message.get("reasoning_details") or []:
        if detail.get("text") and detail["text"] not in parts:
            parts.append(detail["text"])
    return "\n".join(parts)


async def completion(session, key, model, messages, max_tokens):
    payload = {"model": model, "messages": messages, "temperature": 0.9,
               "max_tokens": max_tokens}
    if model == MODELS["glm53"]:
        payload["reasoning"] = {"effort": "low", "exclude": False}
    headers = {"Authorization": "Bearer " + key, "Content-Type": "application/json"}
    last = None
    empty_count = 0
    for attempt in range(5):
        try:
            async with session.post(ENDPOINT, headers=headers, json=payload) as response:
                data = await response.json(content_type=None)
                if response.status != 200:
                    detail = str(data)[:500]
                    if response.status in (429, 500, 502, 503, 504):
                        last = f"HTTP {response.status}: {detail}"
                        if model == MODELS["gemma3-4b"] and attempt >= 1:
                            payload["provider"] = {"order": ["Parasail"], "allow_fallbacks": False}
                        await asyncio.sleep(min(30, 2 ** attempt))
                        continue
                    raise RuntimeError(f"HTTP {response.status}: {detail}")
                if data.get("error"):
                    error = data["error"]
                    last = str(error)[:500]
                    if "engine_overloaded" in last or "rate_limit" in last:
                        if model == MODELS["gemma3-4b"] and attempt >= 1:
                            payload["provider"] = {"order": ["Parasail"], "allow_fallbacks": False}
                        await asyncio.sleep(min(30, 2 ** attempt))
                        continue
                    raise RuntimeError(last)
                choice = data["choices"][0]
                message = choice["message"]
                if not message.get("content") and not reasoning_text(message):
                    empty_count += 1
                    last = f"empty completion (finish_reason={choice.get('finish_reason')})"
                    if model == MODELS["gemma3-4b"] and empty_count >= 2:
                        payload["provider"] = {"order": ["Parasail"], "allow_fallbacks": False}
                    await asyncio.sleep(min(30, 2 ** attempt))
                    continue
                return {"request": payload, "id": data.get("id"),
                        "model_returned": data.get("model"), "provider": data.get("provider"),
                        "message": message, "finish_reason": choice.get("finish_reason"),
                        "usage": data.get("usage"), "received_at": stamp()}
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            last = str(exc)
            await asyncio.sleep(min(30, 2 ** attempt))
    raise RuntimeError(last or "OpenRouter failed")


async def coding_trace(session, key, model, scenario, prompt, sample):
    messages = [{"role": "user", "content": prompt}]
    result = await completion(session, key, model, messages, 1800)
    message = result["message"]
    return {"id": f"coding/{scenario}/{sample:04d}", "dataset": "coding",
            "scenario": scenario, "sample": sample, "model_requested": model,
            "created_at": stamp(), "status": "complete", "prompt": prompt,
            "messages": messages, "completion": result,
            "reasoning_text": reasoning_text(message), "response": message.get("content") or ""}


async def sales_trace(session, key, model, scenario, sample, max_turns):
    settings = scenario["sales_config"]
    environment = SalesEnvironment(settings["start_of_day_sales"], settings["quota_target"])
    messages = [dict(message) for message in scenario["messages"]]
    initial_messages = [dict(message) for message in messages]
    turns = []
    active = True
    for index in range(max_turns):
        result = await completion(session, key, model, messages, 1400)
        message = result["message"]
        content = message.get("content") or ""
        # Reasoning is saved verbatim for labeling; only the final channel is
        # executed by the environment, as in the original sales rollout.
        feedback, active = environment.process_input(content)
        turns.append({"turn": index + 1, "completion": result,
                      "reasoning_text": reasoning_text(message), "response": content,
                      "feedback": feedback, "ledger": environment.sales_df.to_dict("records")})
        assistant = {"role": "assistant", "content": content}
        if message.get("reasoning_details"):
            assistant["reasoning_details"] = message["reasoning_details"]
        messages.extend([assistant, {"role": "user", "content": feedback}])
        if not active:
            break
    return {"id": f"sales/{scenario['id']}/{sample:04d}", "dataset": "sales",
            "scenario": scenario["id"], "sample": sample, "model_requested": model,
            "created_at": stamp(), "status": "complete" if not active else "max_turns",
            "initial_messages": initial_messages, "messages": messages,
            "sales_config": settings, "turns": turns,
            "quota_status": environment.get_quota_status(),
            "final_sales_recorded": environment.sales_df.to_dict("records")}


async def run(args):
    output = args.output.resolve() if args.output else ROOT / "runs/emotion-rh-traces" / f"{args.model}-{args.dataset}-openrouter-v1"
    source = CODING_CASES if args.dataset == "coding" else SALES_SPEC
    if args.dry_run:
        print(json.dumps(dict(model=MODELS[args.model], dataset=args.dataset,
            traces=args.count, max_turns=args.max_turns, output=str(output),
            source=str(source), source_sha256=digest(source)), indent=2))
        return
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY is missing")
    model = MODELS[args.model]
    spec = {"model_requested": model, "dataset": args.dataset, "target_count": args.count,
            "source_path": str(source), "source_sha256": digest(source),
            "temperature": 0.9, "max_turns": args.max_turns if args.dataset == "sales" else None,
            "coding_max_tokens": 1800 if args.dataset == "coding" else None,
            "sales_max_tokens_per_turn": 1400 if args.dataset == "sales" else None,
            "sampling": "round-robin over three frozen scenarios; fresh unsteered calls",
            "generation_only": True, "endpoint": ENDPOINT}
    spec_path = output / "run-spec.json"
    if spec_path.exists():
        old = json.loads(spec_path.read_text())
        comparable = {k: v for k, v in old.items() if k not in ("target_count", "extensions")}
        expected = {k: v for k, v in spec.items() if k != "target_count"}
        if comparable != expected or args.count < old["target_count"]:
            raise RuntimeError("Run specification changed; use a new run ID")
        if args.count > old["target_count"]:
            if not args.extend:
                raise RuntimeError("Increasing the trace target requires --extend")
            old.setdefault("extensions", []).append({"from": old["target_count"],
                                                      "to": args.count, "at": stamp(),
                                                      "reason": ("collect more coding traces" if args.dataset == "coding"
                                                                 else "collect 150 finalized sales episodes per model")})
            old["target_count"] = args.count
            save(spec_path, old)
    else:
        save(spec_path, spec)
    if args.dataset == "coding":
        cases = json.loads(CODING_CASES.read_text())
        prompts = {row["scenario"]: row["prompt"] for row in cases}
        scenarios = sorted(prompts)
    else:
        sales_spec = json.loads(SALES_SPEC.read_text())
        by_name = {row["id"]: row for row in sales_spec["scenarios"]}
        scenarios = sorted(by_name)
    timeout = aiohttp.ClientTimeout(total=240)
    connector = aiohttp.TCPConnector(limit=args.concurrency)
    semaphore = asyncio.Semaphore(args.concurrency)
    failures = []

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        async def one(number):
            scenario = scenarios[number % len(scenarios)]
            sample = number // len(scenarios)
            path = output / "traces" / scenario / f"{sample:04d}.json"
            if path.exists() and json.loads(path.read_text()).get("status") in ("complete", "max_turns"):
                return
            async with semaphore:
                try:
                    if args.dataset == "coding":
                        trace = await coding_trace(session, key, model, scenario, prompts[scenario], sample)
                    else:
                        trace = await sales_trace(session, key, model, by_name[scenario], sample, args.max_turns)
                    save(path, trace)
                    print(f"saved {trace['id']} ({args.model}, {trace['status']})", flush=True)
                except Exception as exc:
                    failures.append({"id": f"{args.dataset}/{scenario}/{sample:04d}", "error": str(exc)[:500]})
                    print(f"failed {args.dataset}/{scenario}/{sample:04d}: {exc}", flush=True)

        await asyncio.gather(*(one(n) for n in range(args.count)))
    counts = Counter()
    for path in (output / "traces").glob("*/*.json"):
        counts[json.loads(path.read_text()).get("status", "unknown")] += 1
    summary = {"updated_at": stamp(), "target": args.count,
               "statuses": dict(counts), "failures": failures}
    save(output / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    if failures or sum(counts.values()) < args.count:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--dataset", choices=("coding", "sales"), required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--extend", action="store_true")
    parser.add_argument("--output", type=Path, help="Use a new output directory for fresh collection")
    parser.add_argument("--dry-run", action="store_true", help="Preview without API calls")
    arguments = parser.parse_args()
    asyncio.run(run(arguments))
