"""Versioned Harmony replay: native channels, preserved text, frozen date."""
import hashlib
from pathlib import Path
from .common import digest, require

RENDERER = "gpt-oss-harmony-preserved-streams-v1"
OPEN, CLOSE = "\ue000PROBE_TARGET_START\ue001", "\ue000PROBE_TARGET_END\ue001"


def render_case(case, tokenizer):
    t = case["target"]
    require(
        t["message_index"] == len(case["messages"]) - 1,
        "Target must be in final supplied message",
    )

    def build(mark):
        messages = []
        system = []
        for i, m in enumerate(case["messages"]):
            role = m["role"]
            content = m.get("content", "")
            reason = m.get("reasoning", "")
            require(
                OPEN not in content + reason and CLOSE not in content + reason,
                "Reserved marker in input",
            )
            if case["render_mode"] == "sales_with_reasoning":
                content, reason = content.strip(), reason.strip()
            if i == t["message_index"]:
                require(role == "assistant", "Target must be assistant")
                body = reason if t["stream"] == "reasoning" else content
                a, b = t["char_start"], t["char_end"]
                require(body[a:b] == t["text"], "Target character offsets changed")
                if mark:
                    body = body[:a] + OPEN + body[a:b] + CLOSE + body[b:]
                if t["stream"] == "reasoning":
                    reason = body
                else:
                    content = body
            require(role in ("system", "user", "assistant", "tool"), f"Unsupported role: {role}")
            item = {"role": role, "content": content}
            if reason:
                require(role == "assistant", "Reasoning on non-assistant message")
                item["thinking"] = reason
            messages.append(item)
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
            chat_template=Path(__file__).with_name("replay_template.jinja").read_text(),
        )

    marked, rendered = build(True), build(False)
    require(
        marked.count(OPEN) == marked.count(CLOSE) == 1, "Template changed span markers"
    )
    start = marked.index(OPEN)
    end = marked.index(CLOSE) - len(OPEN)
    require(
        marked.replace(OPEN, "").replace(CLOSE, "") == rendered,
        "Marker round-trip changed rendering",
    )
    require(rendered[start:end] == t["text"], "Rendered target mismatch")
    return rendered, start, end


def tokenize_case(case, tokenizer, max_tokens):
    rendered, start, end = render_case(case, tokenizer)
    enc = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    positions = [
        i
        for i, (a, b) in enumerate(enc["offset_mapping"])
        if b > a and b > start and a < end
    ]
    require(
        positions and positions == list(range(positions[0], positions[-1] + 1)),
        "Noncontiguous or empty target tokens",
    )
    ids = enc["input_ids"][: positions[-1] + 1]
    require(
        len(ids) <= max_tokens,
        f"{case['id']}: causal prefix {len(ids)} exceeds {max_tokens}; no truncation allowed",
    )
    return {
        "id": case["id"],
        "input_ids": ids,
        "positions": positions,
        "target_token_count": len(positions),
        "prefix_token_count": len(ids),
        "case_sha256": digest(case),
        "rendered_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
        "token_ids_sha256": digest(ids),
        "target_token_ids_sha256": digest([ids[i] for i in positions]),
        "target_char_span": [start, end],
        "target_token_span": [positions[0], positions[-1] + 1],
    }
