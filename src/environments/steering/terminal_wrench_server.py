"""OpenAI chat endpoint binding each TerminalWrench episode to its intervention.

Run on the GPU host; tunnel its port to the Docker/Harbor host. No tools or shell
commands execute on the GPU host. The pinned native Terminus2 parser executes
only the returned final-channel JSON in each fresh Harbor task environment.
"""
from __future__ import annotations

import argparse
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import threading
import time
import urllib.request
import uuid

from .terminal_wrench import ROOT, digest, episodes, freeze, read, verify_plan


def parse_harmony(raw, *, complete=True):
    marker = '<|channel|>final<|message|>'
    if marker in raw:
        analysis, final = raw.split(marker, 1)
    elif raw.startswith('final<|message|>'):
        analysis, final = '', raw[len('final<|message|>'):]
    elif not complete:
        return dict(content='', reasoning_content=raw)
    else:
        raise ValueError('Completed native GPT-OSS response has no final channel')
    final = re.split(r'<\|(?:im_end|fim_suffix|ghissue)\|>', final)[0].strip()
    analysis = re.sub(r'^(?:<\|channel\|>)?analysis<\|message\|>', '', analysis)
    if complete and not final:
        raise ValueError('Completed native GPT-OSS final channel is empty')
    return dict(content=final, reasoning_content=analysis.strip())


def endpoint_info(api_base):
    request = urllib.request.Request(api_base.rstrip('/') + '/steering-info')
    token = os.environ.get('TERMINALWRENCH_API_KEY')
    if token:
        request.add_header('Authorization', 'Bearer ' + token)
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


class CompletionService:
    def __init__(self, spec, output, backend=None, on_checkpoint=None):
        verify_plan(spec)
        self.spec = spec
        self.output = Path(output)
        self.lock = threading.Lock()
        self.on_checkpoint = on_checkpoint or (lambda: None)
        self.episodes = {e['id']: e for e in episodes(spec)}
        self.probes = {p['emotion']: p for p in spec['probes']}
        from .backend import load_vector
        self.vectors = {e: load_vector(p, ROOT) for e, p in self.probes.items()}
        if backend is None:
            from .gpt_oss_backend import make_backend
            backend = make_backend(spec, ROOT)
        self.backend = backend
        for emotion, vector in self.vectors.items():
            if len(vector) != backend.hidden_size or not 0 <= self.probes[emotion]['layer_zero_based'] < backend.num_layers:
                raise ValueError('Probe/model dimensions differ')
        self.info = dict(plan_sha256=spec['manifest_sha256'], model=spec['model'],
                         steering_scope=spec['steering_scope'], episode_count=len(self.episodes))

    def complete(self, body):
        episode = self.episodes.get(body.get('model'))
        if episode is None:
            raise ValueError('Unknown episode model alias; use a model from this plan')
        if body.get('stream') or body.get('n', 1) != 1:
            raise ValueError('This endpoint requires stream=false and n=1')
        for name in ('tools', 'tool_choice', 'functions', 'response_format', 'stop', 'logit_bias'):
            if body.get(name):
                raise ValueError(f'{name} is unsupported by the native JSON-mono-tool adapter')
        policy = self.spec['generation']
        expected = dict(seed=episode['seed'], max_tokens=policy['max_new_tokens'],
                        temperature=policy['temperature'], top_p=policy['top_p'])
        for key, value in expected.items():
            if key in body and body[key] != value:
                raise ValueError(f'Request {key} differs from the frozen episode policy')
        if body.get('reasoning_effort') not in (None, policy['reasoning_effort']):
            raise ValueError('Request reasoning effort differs from frozen policy')
        messages = body.get('messages')
        if not isinstance(messages, list) or not messages:
            raise ValueError('messages must be a nonempty native chat history')
        for message in messages:
            if message.get('role') not in ('system', 'user', 'assistant') or not isinstance(message.get('content'), str):
                raise ValueError('Expected text-only Terminus2 chat messages')
            if message.get('tool_calls'):
                raise ValueError('Native Terminus2 JSON commands belong in message content')
        emotion = episode['emotion']
        call_id = uuid.uuid4().hex
        directory = self.output / episode['id'] / call_id
        # Bind hooks AND RNG inside one lock so concurrent Harbor episodes cannot
        # inherit one another's steering direction or random seed.
        with self.lock:
            started = time.monotonic()
            freeze(directory / 'request.json', dict(plan_sha256=self.spec['manifest_sha256'],
                episode=episode, request=body, effective_generation=expected))
            try:
                result = self.backend.chat_completion(messages, self.probes.get(emotion),
                    self.vectors.get(emotion), strength=episode['strength'], seed=episode['seed'],
                    max_new_tokens=policy['max_new_tokens'], temperature=policy['temperature'],
                    top_p=policy['top_p'], context_limit=policy['context_limit'])
            except Exception as error:
                freeze(directory / 'error.json', dict(type=type(error).__name__, message=str(error)))
                self.on_checkpoint()
                raise
            freeze(directory / 'generation.json', dict(**result, elapsed_seconds=time.monotonic() - started))
            self.on_checkpoint()
        return dict(id='chatcmpl-' + call_id, object='chat.completion', created=int(time.time()),
            model=episode['id'], choices=[dict(index=0, finish_reason=result['finish_reason'],
                message=dict(role='assistant', content=result['content'], reasoning_content=result['reasoning_content']))],
            usage=dict(prompt_tokens=result['input_tokens'], completion_tokens=result['generated_tokens'],
                       total_tokens=result['input_tokens'] + result['generated_tokens']),
            system_fingerprint='tw-' + self.spec['manifest_sha256'][:24])


def make_server(service, host='127.0.0.1', port=8000):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send_json(self, status, payload):
            data = json.dumps(payload, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def authorized(self):
            token = os.environ.get('TERMINALWRENCH_API_KEY')
            if token and not hmac.compare_digest(self.headers.get('Authorization', ''), 'Bearer ' + token):
                self.send_json(401, dict(error=dict(message='Invalid API key')))
                return False
            return True

        def do_GET(self):
            if not self.authorized():
                return
            if self.path == '/v1/steering-info':
                self.send_json(200, service.info)
            elif self.path == '/v1/models':
                self.send_json(200, dict(object='list', data=[dict(id=k, object='model', owned_by='local')
                    for k in service.episodes]))
            else:
                self.send_json(404, dict(error=dict(message='Unknown endpoint')))

        def do_POST(self):
            if not self.authorized():
                return
            if self.path != '/v1/chat/completions':
                self.send_json(404, dict(error=dict(message='Unknown endpoint')))
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 32 * 1024 * 1024:
                    raise ValueError('Invalid request size')
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError('Expected a JSON request object')
                result = service.complete(body)
            except (ValueError, KeyError, TypeError) as error:
                self.send_json(400, dict(error=dict(message=str(error), type='invalid_request_error',
                    code='context_length_exceeded' if 'context_length_exceeded' in str(error) else 'invalid_request')))
            except Exception as error:
                self.send_json(500, dict(error=dict(message=str(error), type='server_error')))
            else:
                self.send_json(200, result)

    return ThreadingHTTPServer((host, port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='GPU-host raw request/response directory')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--modal-volume', help='Commit raw records after each completion when running on Modal')
    args = parser.parse_args()
    checkpoint = None
    if args.modal_volume:
        import modal
        checkpoint = modal.Volume.from_name(args.modal_volume).commit
    service = CompletionService(read(args.plan), args.output, on_checkpoint=checkpoint)
    print(json.dumps(dict(ready=True, **service.info)), flush=True)
    with make_server(service, args.host, args.port) as server:
        server.serve_forever()


if __name__ == '__main__':
    main()
