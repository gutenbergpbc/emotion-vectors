"""Native Harbor tasks use a condition-bound chat endpoint and fresh Docker trials."""

import json
import os
import signal
import subprocess
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from shared.io import atomic, digest, read, require, sha

from .config import tasks

HARBOR_REVISION = "64afbbcb62165950301e1a6407c729aa26d844ff"


def endpoint_identity(spec):
    # Transport URL is learned when serving starts; it does not change the intervention.
    return digest(
        {
            k: spec[k]
            for k in (
                "model",
                "probes",
                "strengths",
                "seed",
                "samples",
                "max_new_tokens",
                "temperature",
                "top_p",
                "scenarios",
                "source_hashes",
            )
        }
    )


def api_base(spec):
    value = spec["terminalwrench"].get("api_base") or os.environ.get(
        "TERMINALWRENCH_API_BASE"
    )
    require(
        value and value.startswith(("http://", "https://")),
        "Set TERMINALWRENCH_API_BASE to the live endpoint /v1",
    )
    return value.rstrip("/")


def preflight(spec):
    cfg = spec["terminalwrench"]
    executable = cfg.get("harbor", "harbor")
    version = subprocess.check_output([executable, "--version"], text=True)
    require("0.21.0" in version, "Use pinned Harbor 0.21.0")
    import shutil

    installed = Path(shutil.which(executable) or executable)
    shebang = installed.read_text().splitlines()[0]
    require(
        shebang.startswith("#!") and "python" in shebang,
        "Use the pinned Harbor Python console script",
    )
    source = json.loads(
        subprocess.check_output(
            [
                shebang[2:].strip(),
                "-c",
                'import importlib.metadata; print(importlib.metadata.distribution("harbor").read_text("direct_url.json") or "{}")',
            ],
            text=True,
        )
    )
    require(
        source.get("vcs_info", {}).get("commit_id") == HARBOR_REVISION,
        "Harbor Git revision mismatch",
    )
    request = urllib.request.Request(api_base(spec) + "/steering-info")
    token = os.environ.get("TERMINALWRENCH_API_KEY")
    if token:
        request.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(request, timeout=30) as response:
        info = json.load(response)
    require(
        info["identity"] == endpoint_identity(spec),
        "Endpoint serves a different model/intervention/dataset",
    )
    for scenario in spec["scenarios"]:
        folder = Path(scenario["task_path"])
        actual = {
            str(p.relative_to(folder)): sha(p)
            for p in sorted(folder.rglob("*"))
            if p.is_file()
        }
        require(actual == scenario["task_hashes"], "Native TerminalWrench task changed")


def alias(task):
    return "steering-" + digest(task["id"])[:24]


def generate_one(scenario, task, spec, root):
    directory = root / "native-trials" / alias(task)
    require(
        not directory.exists(),
        f"Uncommitted Harbor trial exists: {directory}; inspect before retrying in a new run",
    )
    directory.mkdir(parents=True)
    policy = spec["terminalwrench"]
    name = alias(task)
    config = {
        "job_name": name,
        "jobs_dir": str(directory / "jobs"),
        "n_attempts": 1,
        "n_concurrent_trials": 1,
        "environment": {"type": policy.get("environment", "docker"), "force_build": False, "delete": True, "kwargs": policy.get("environment_kwargs", {})},
        "agents": [
            {
                "name": "terminus-2",
                "override_timeout_sec": policy.get("wall_watchdog_seconds", policy.get("agent_timeout_seconds", 900)),
                "model_name": "hosted_vllm/" + name,
                "kwargs": {
                    "api_base": api_base(spec),
                    "parser_name": "json",
                    "temperature": spec["temperature"],
                    "max_turns": spec["max_turns"],
                    "enable_summarize": False,
                    "store_all_messages": True,
                    "record_terminal_session": True,
                    "trajectory_config": {"raw_content": False, "linear_history": True},
                    "model_info": {
                        "max_input_tokens": spec["model"].get("max_context", 131072),
                        "max_output_tokens": spec["max_new_tokens"],
                        "input_cost_per_token": 0.0,
                        "output_cost_per_token": 0.0,
                    },
                    "llm_call_kwargs": {
                        "seed": task["seed"],
                        "max_tokens": spec["max_new_tokens"],
                        "top_p": spec["top_p"],
                    },
                },
            }
        ],
        "tasks": [{"path": scenario["task_path"]}],
    }
    if policy.get("timing_policy") == "active-time-v2":
        agent=config["agents"][0]
        agent.pop("name",None)
        agent["import_path"]="emotion_steering.active_budget:ActiveTimeTerminus2"
        agent["kwargs"]["budget_url"]=policy["budget_base"]+"/clock/"+name
        if policy.get("budget_file_base"):
            agent["kwargs"]["budget_file"]=str(Path(policy["budget_file_base"])/(name+".json"))
        agent["kwargs"]["llm_call_kwargs"]["timeout"]=6300
    atomic(directory / "job.json", config)
    failure = None
    with (
        (directory / "stdout.log").open("w") as out,
        (directory / "stderr.log").open("w") as err,
    ):
        child_env = dict(os.environ)
        canonical = Path(__file__).resolve().parents[2]
        child_env["PYTHONPATH"] = str(canonical) + os.pathsep + str(canonical / "emotion-steering") + os.pathsep + child_env.get("PYTHONPATH", "")
        if os.environ.get("TERMINALWRENCH_API_KEY"):
            child_env.update(
                OPENAI_API_KEY=os.environ["TERMINALWRENCH_API_KEY"],
                HOSTED_VLLM_API_KEY=os.environ["TERMINALWRENCH_API_KEY"],
            )
        proc = subprocess.Popen(
            [
                policy.get("harbor", "harbor"),
                "run",
                "-c",
                str(directory / "job.json"),
                "--yes",
            ],
            stdout=out,
            stderr=err,
            start_new_session=True,
            env=child_env,
        )
        try:
            proc.wait(timeout=policy.get("timeout_seconds", 3600))
            if proc.returncode:
                failure = f"harbor_exit_{proc.returncode}"
        except subprocess.TimeoutExpired:
            failure = "job_wall_watchdog_timeout"
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
    results = list((directory / "jobs" / name).glob("*/result.json"))
    trajectories = list((directory / "jobs" / name).glob("*/agent/trajectory.json"))
    native = read(results[0]) if len(results) == 1 else None
    trajectory = read(trajectories[0]) if len(trajectories) == 1 else None
    if not native or trajectory is None:
        failure = failure or "incomplete_native_trial"
    if native and native.get("exception_info"):
        exception=native["exception_info"]
        kind=exception.get("exception_type", "unknown")
        if kind=="ActiveBudgetTimeoutError":failure="active_budget_timeout"
        elif kind=="BudgetInfrastructureError":failure="queue_watchdog_timeout" if "queue_watchdog_timeout" in exception.get("exception_message", "") else "budget_infrastructure_error"
        elif kind=="AgentTimeoutError" and policy.get("timing_policy")=="active-time-v2":failure="wall_watchdog_timeout"
        else:failure="native_exception:"+kind
    return {
        "response": "",
        "trajectory_text": json.dumps(trajectory, ensure_ascii=False)
        if trajectory
        else "",
        "finish_reason": failure or "eos",
        "termination": failure or "complete",
        "generation_error": failure,
        "native_exception": native.get("exception_info") if native else None,
        "objective": {
            "native_verifier": native.get("verifier_result") if native else None
        },
        "native_artifacts": [
            {"path": str(p.relative_to(root)), "sha256": sha(p)}
            for p in results + trajectories
        ],
        "note": "Native reward is not a reward-hacking behavior label",
    }


class Service:
    def __init__(self, spec, root, backend):
        self.spec, self.root, self.backend = spec, Path(root), backend
        self.lock = threading.Lock()
        self.tasks = {alias(t): t for t in tasks(spec)}
        self.task_locks = {name: threading.Lock() for name in self.tasks}
        self.probes = {p["emotion"]: p for p in spec["probes"]}

    def complete(self, body):
        require(body.get("model") in self.tasks, "Unknown episode model alias")
        require(
            not body.get("stream") and body.get("n", 1) == 1,
            "Only non-streaming, single completions supported",
        )
        require(
            not any(
                body.get(k)
                for k in ("tools", "functions", "stop", "logit_bias", "tool_choice")
            ),
            "Unsupported native JSON-agent request",
        )
        task = self.tasks[body["model"]]
        for field, expected in [
            ("seed", task["seed"]),
            ("temperature", self.spec["temperature"]),
            ("top_p", self.spec["top_p"]),
            ("max_tokens", self.spec["max_new_tokens"]),
        ]:
            require(
                body.get(field, expected) == expected,
                "Request differs from frozen " + field,
            )
        messages = body["messages"]
        require(
            messages and all(isinstance(m.get("content"), str) for m in messages),
            "Text chat messages required",
        )
        turn = sum(m["role"] == "assistant" for m in messages)
        with self.task_locks[body["model"]]:
            key = digest({"task": task["id"], "messages": messages})
            path = self.root / "calls" / f"{key}.json"
            if path.exists():
                output = read(path)["output"]
            else:
                complete = (lambda *args: self.backend.complete_for_task(task, *args)) if hasattr(self.backend, "complete_for_task") else self.backend.complete
                output = complete(
                    messages,
                    self.probes.get(task["emotion"]),
                    task["strength"],
                    task["seed"] + turn * 1000003,
                    self.spec["max_new_tokens"],
                )
                atomic(
                    path,
                    {
                        "task": task,
                        "messages": messages,
                        "output": output,
                        "identity": endpoint_identity(self.spec),
                    },
                )
        return {
            "id": "chatcmpl-" + key[:20],
            "object": "chat.completion",
            "model": body["model"],
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": output["response"],
                        "reasoning_content": output.get("reasoning_text", ""),
                    },
                    "finish_reason": "stop"
                    if output["finish_reason"] == "eos"
                    else "length",
                }
            ],
            "usage": {
                "prompt_tokens": output["input_tokens"],
                "completion_tokens": output["generated_tokens"],
                "total_tokens": output["input_tokens"] + output["generated_tokens"],
            },
        }


def serve(spec, root, backend, host="127.0.0.1", port=8000, clock=None):
    import hmac

    service = Service(spec, root, backend)
    token = os.environ.get("TERMINALWRENCH_API_KEY")
    require(
        host in ("127.0.0.1", "localhost") or token,
        "A non-loopback endpoint requires TERMINALWRENCH_API_KEY",
    )

    class Handler(BaseHTTPRequestHandler):
        def authorized(self):
            return not token or hmac.compare_digest(
                self.headers.get("Authorization", ""), "Bearer " + token
            )

        def send_json(self, status, data):
            body = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def serve_clock_route(self, start=False):
            """Serve budget state on the same forwarded port as model requests."""
            if clock is None or not self.path.startswith("/clock/"):
                return False
            from urllib.parse import parse_qs, urlsplit

            parsed = urlsplit(self.path)
            key = parsed.path.removeprefix("/clock/")
            try:
                action = parse_qs(parsed.query).get("action", [""])[0]
                if action == "stop":
                    state = clock.stop(key)
                elif start:
                    state = clock.start(key)
                else:
                    state = clock.snapshot(key)
                self.send_json(200, state)
            except Exception as exc:
                self.send_json(500, {"error": repr(exc)})
            return True

        def do_GET(self):
            if not self.authorized():
                self.send_json(401, {"error": "Unauthorized"})
                return
            if self.serve_clock_route():
                return
            if self.path == "/v1/steering-info":
                self.send_json(
                    200,
                    {
                        "identity": endpoint_identity(spec),
                        "model": spec["model"]["name"],
                    },
                )
            else:
                self.send_json(404, {"error": "Not found"})

        def do_POST(self):
            if not self.authorized():
                self.send_json(401, {"error": "Unauthorized"})
                return
            if self.serve_clock_route(start=True):
                return
            if self.path != "/v1/chat/completions":
                self.send_json(404, {"error": "Not found"})
                return
            try:
                size = int(self.headers.get("Content-Length", 0))
                require(0 < size < 8_000_000, "Invalid request size")
                result = service.complete(json.loads(self.rfile.read(size)))
                self.send_json(200, result)
            except ValueError as exc:
                self.send_json(400, {"error": str(exc)})
            except Exception:
                self.send_json(
                    500, {"error": "Generation failed; inspect endpoint logs"}
                )
                raise

    ThreadingHTTPServer((host, port), Handler).serve_forever()


if __name__ == "__main__":
    import argparse

    from .backend import make_backend

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    spec = read(args.run_dir / "config.resolved.json")
    serve(spec, args.run_dir, make_backend(spec), args.host, args.port)
