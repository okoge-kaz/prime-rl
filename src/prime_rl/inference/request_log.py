"""Per-request serving-side JSONL logs, off the engine hot path.

Enabled by ``PRIME_REQUEST_LOG_DIR`` (set by the launcher from
``inference.enable_request_logs``). Two record kinds land in one file per
process (``<dir>/<hostname>-pid<pid>.jsonl``):

- ``request``: one line per finished request from vLLM's stat-logger hook
  (API-server process) — queue/prefill/decode durations, cached/prompt/output
  token counts, finish reason, engine identity.
- ``kv_transfer``: one line per finished NIXL recv/send notification
  (engine-core process, emitted from the KV-xfer scheduler patch).

The ``request_id`` embeds the rollout's trace id (the router derives it from
the ``X-Session-ID`` header), so records join back to ``traces.jsonl`` nodes;
within a trajectory, turns are sequential, so finish-time order recovers the
turn index.

Writers buffer records in memory and a daemon thread flushes every
``_FLUSH_INTERVAL`` seconds — callers only pay a dict append under a lock,
never file I/O on the event loop or scheduler loop.
"""

import atexit
import os
import socket
import threading
import time
from pathlib import Path

import orjson

_FLUSH_INTERVAL = 2.0

ENV_VAR = "PRIME_REQUEST_LOG_DIR"


class JsonlWriter:
    """Append-only buffered JSONL writer with a background flush thread."""

    def __init__(self, path: Path):
        self.path = path
        self._buffer: list[dict] = []
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._flush_loop, name="request-log-writer", daemon=True)
        self._thread.start()
        atexit.register(self.flush)

    def write(self, record: dict) -> None:
        with self._lock:
            self._buffer.append(record)

    def flush(self) -> None:
        with self._lock:
            buffer, self._buffer = self._buffer, []
        if not buffer:
            return
        blob = b"".join(orjson.dumps(r, default=str, option=orjson.OPT_APPEND_NEWLINE) for r in buffer)
        with open(self.path, "ab") as f:
            f.write(blob)

    def _flush_loop(self) -> None:
        while True:
            time.sleep(_FLUSH_INTERVAL)
            try:
                self.flush()
            except Exception:
                # A failed flush (e.g. transient FS error) must never take the
                # serving process down; records are retried next interval.
                pass


_writer: JsonlWriter | None = None
_writer_lock = threading.Lock()


def get_writer() -> JsonlWriter | None:
    """Process-wide writer, or None when request logging is disabled."""
    log_dir = os.environ.get(ENV_VAR)
    if not log_dir:
        return None
    global _writer
    with _writer_lock:
        if _writer is None:
            _writer = JsonlWriter(Path(log_dir) / f"{socket.gethostname()}-pid{os.getpid()}.jsonl")
        return _writer


def kv_role(vllm_config) -> str:
    kv_config = vllm_config.kv_transfer_config
    if kv_config is None:
        return "unified"
    if kv_config.is_kv_producer:
        return "prefill"
    if kv_config.is_kv_consumer:
        return "decode"
    return "unified"


def make_request_stat_logger():
    """Build the ``StatLoggerBase`` subclass lazily so this module stays importable
    in processes where vLLM isn't (fully) initialized."""
    from vllm.v1.metrics.loggers import StatLoggerBase

    class RequestStatLogger(StatLoggerBase):
        """Emits one JSONL record per finished request.

        ``record()`` runs in the AsyncLLM output-handler loop; per iteration it
        only touches ``iteration_stats.finished_requests`` (usually empty), so
        the added work is one small dict append per *finished* request.
        """

        def __init__(self, vllm_config, engine_index: int = 0):
            self.engine_index = engine_index
            self.role = kv_role(vllm_config)
            self.host = socket.gethostname()
            self.writer = get_writer()

        def record(self, scheduler_stats, iteration_stats, mm_cache_stats=None, engine_idx: int = 0):
            if self.writer is None or iteration_stats is None:
                return
            now = time.time()
            for finished in iteration_stats.finished_requests:
                self.writer.write(
                    {
                        "kind": "request",
                        "ts": now,
                        "request_id": finished.request_id,
                        "role": self.role,
                        "host": self.host,
                        "engine": engine_idx,
                        "finish_reason": str(finished.finish_reason),
                        "num_prompt_tokens": finished.num_prompt_tokens,
                        "num_cached_tokens": finished.num_cached_tokens,
                        "num_generation_tokens": finished.num_generation_tokens,
                        "queued_time": finished.queued_time,
                        "prefill_time": finished.prefill_time,
                        "inference_time": finished.inference_time,
                        "decode_time": finished.decode_time,
                        "e2e_latency": finished.e2e_latency,
                    }
                )

        def log_engine_initialized(self):
            pass

    return RequestStatLogger


def log_kv_transfer_event(event: str, request_id: str) -> None:
    """Record a finished NIXL recv/send notification (engine-core process)."""
    writer = get_writer()
    if writer is None:
        return
    writer.write({"kind": "kv_transfer", "ts": time.time(), "event": event, "request_id": request_id})


def monkey_patch_request_stat_logger():
    """Append the per-request JSONL stat logger to every StatLoggerManager.

    No-op unless ``PRIME_REQUEST_LOG_DIR`` is set. Custom factories are the
    supported extension point (``StatLoggerManager(custom_stat_loggers=...)``),
    but the CLI ``run_server`` path never threads them through — hence the patch.
    """
    if not os.environ.get(ENV_VAR):
        return

    from vllm.v1.metrics.loggers import StatLoggerManager

    if getattr(StatLoggerManager.__init__, "_prime_rl_request_log", False):
        return

    original_init = StatLoggerManager.__init__

    def patched_init(self, *args, custom_stat_loggers=None, **kwargs):
        factories = list(custom_stat_loggers or [])
        factories.append(make_request_stat_logger())
        original_init(self, *args, custom_stat_loggers=factories, **kwargs)

    patched_init._prime_rl_request_log = True
    StatLoggerManager.__init__ = patched_init
