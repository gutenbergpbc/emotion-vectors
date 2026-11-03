"""Live queue and execution timing for condition-isolated inference batches."""
import queue
import threading
import time
import uuid
from concurrent.futures import CancelledError, Future


class CompletionQueue:
    """One engine owner, with live monotonic timing independent of RPC threads."""
    def __init__(self, backend, batch_size=4, collect_seconds=0.04):
        self.backend = backend
        self.batch_size = batch_size
        self.collect_seconds = collect_seconds
        self.pending = queue.Queue()
        self.lock = threading.Lock()
        self.requests = {}
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def status(self, request_id):
        with self.lock:
            item = self.requests.get(request_id)
            if item is None:
                return {"request_id": request_id, "state": "unknown",
                        "queued_seconds": 0.0, "execution_seconds": 0.0}
            now = time.monotonic()
            started = item.get("started_mono")
            ended = item.get("ended_mono")
            return {"request_id": request_id, "state": item["state"],
                    "queued_seconds": (started if started is not None else ended if ended is not None else now) - item["received_mono"],
                    "execution_seconds": 0.0 if started is None else (ended if ended is not None else now) - started,
                    "worker_received_at": item["received_at"],
                    "execution_started_at": item.get("started_at"),
                    "execution_ended_at": item.get("ended_at"),
                    "batch_id": item.get("batch_id"),
                    "abandoned": item.get("abandoned", False),
                    "cancel_requested_at": item.get("cancel_requested_at")}

    def cancel(self, request_id):
        """Drop queued work; executing synchronous vLLM batches cannot be interrupted."""
        with self.lock:
            item = self.requests.get(request_id)
            if item is None:
                # A tombstone prevents dispatch/status RPC ordering from resurrecting work.
                self.requests[request_id] = {"state": "cancelled", "received_mono": time.monotonic(),
                    "received_at": time.time(), "ended_mono": time.monotonic(),
                    "ended_at": time.time(), "abandoned": True, "cancel_requested_at": time.time()}
                return {"cancelled_before_execution": True, "execution_interrupt_supported": False}
            if item["state"] == "queued":
                item.update(state="cancelled", ended_mono=time.monotonic(), ended_at=time.time(),
                            abandoned=True, cancel_requested_at=time.time())
                item.pop("future").set_exception(CancelledError("Episode cancelled before inference execution"))
                return {"cancelled_before_execution": True, "execution_interrupt_supported": False}
            if item["state"] == "executing":
                item.update(abandoned=True, cancel_requested_at=time.time())
            return {"cancelled_before_execution": item["state"] == "cancelled",
                    "execution_interrupt_supported": False, "state": item["state"]}

    def complete(self, request, request_id=None):
        request_id = request_id or str(uuid.uuid4())
        if not isinstance(request_id, str) or not request_id or len(request_id) > 200:
            raise ValueError("Invalid inference request id")
        future = Future()
        with self.lock:
            if request_id in self.requests:
                raise ValueError("Duplicate inference request id")
            self.requests[request_id] = {"state": "queued", "received_mono": time.monotonic(),
                                         "received_at": time.time(), "future": future}
        self.pending.put((request, future, request_id))
        return future.result()

    def _run(self):
        while True:
            batch = [self.pending.get()]
            deadline = time.monotonic() + self.collect_seconds
            while len(batch) < self.batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    batch.append(self.pending.get(timeout=remaining))
                except queue.Empty:
                    break
            batch_id = str(uuid.uuid4())
            with self.lock:
                now, wall = time.monotonic(), time.time()
                batch = [entry for entry in batch if self.requests[entry[2]]["state"] != "cancelled"]
                for _, _, request_id in batch:
                    self.requests[request_id].pop("future", None)
                    self.requests[request_id].update(state="executing", started_mono=now,
                                                     started_at=wall, batch_id=batch_id)
            if not batch:
                continue
            try:
                outputs = self.backend.complete_batch([request for request, _, _ in batch])
                if len(outputs) != len(batch):
                    raise RuntimeError("Incomplete inference batch")
            except BaseException as exc:
                with self.lock:
                    now, wall = time.monotonic(), time.time()
                    for _, _, request_id in batch:
                        self.requests[request_id].update(state="failed", ended_mono=now, ended_at=wall)
                for _, future, _ in batch:
                    future.set_exception(exc)
            else:
                with self.lock:
                    now, wall = time.monotonic(), time.time()
                    for _, _, request_id in batch:
                        item = self.requests[request_id]
                        item.update(state="abandoned" if item.get("abandoned") else "completed", ended_mono=now, ended_at=wall)
                for (_, future, request_id), output in zip(batch, outputs, strict=True):
                    future.set_result({**output, "inference_timing": self.status(request_id)})
