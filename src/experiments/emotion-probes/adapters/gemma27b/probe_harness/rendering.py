"""Gemma-native roles; explicit text representation of supplied tool/reasoning streams.

No input case is edited. System text is prefixed to the first user turn; tool
messages become user turns with a visible wrapper; adjacent same-role turns
join with two newlines. Saved reasoning is ordinary assistant text in <think>.
Assistant-first histories receive an empty user turn (no added prose).
Both training and evaluation use this identical adapter. Marker round-trip
checks establish exact character offsets even for repeated target strings.
"""

import hashlib
from .common import digest, require

RENDERER = "gemma-native-preserved-streams-v2"
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
            if role == "assistant" and reason:
                content = "<think>\n" + reason + "\n</think>\n\n" + content
            if role == "system":
                require(not messages, "System messages must precede conversation")
                system.append(content)
                continue
            if role == "tool":
                role, content = "user", "[Tool response]\n" + content
            require(role in ("user", "assistant"), f"Unsupported role: {role}")
            if system:
                require(role == "user", "System prefix requires first user turn")
                content = "\n\n".join(system) + "\n\n" + content
                system = []
            if messages and messages[-1]["role"] == role:
                messages[-1]["content"] += "\n\n" + content
            else:
                messages.append({"role": role, "content": content})
        require(not system, "Unconsumed system prefix")
        # Gemma requires a user-first chat. Two frozen calm cases start with
        # assistant context; an empty user turn preserves every supplied utterance.
        if messages and messages[0]["role"] == "assistant":
            messages.insert(0, {"role": "user", "content": ""})
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
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
