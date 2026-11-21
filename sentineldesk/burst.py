"""Live outage / burst detection.

Keeps a sliding window of recently triaged incident-like tickets. When a new ticket is very
similar to enough recent ones (default: ≥3 tickets within 15 min at cosine ≥ 0.80), the group is
flagged as a probable major incident so the console can raise a banner and the gate can page
on-call - even if each ticket on its own looked routine.
"""

from __future__ import annotations

import threading
import time
from collections import Counter, deque
from dataclasses import dataclass

import numpy as np

from . import config


@dataclass
class _Entry:
    ts: float
    ref: str
    subject: str
    queue: str
    emb: np.ndarray


class BurstDetector:
    def __init__(self, window_s: float = config.BURST_WINDOW_S,
                 min_tickets: int = config.BURST_MIN_TICKETS,
                 threshold: float = config.BURST_SIMILARITY):
        self.window_s, self.min_tickets, self.threshold = window_s, min_tickets, threshold
        self._items: deque[_Entry] = deque(maxlen=500)
        self._lock = threading.Lock()

    def _evict(self, now: float) -> None:
        while self._items and now - self._items[0].ts > self.window_s:
            self._items.popleft()

    def observe(self, emb: np.ndarray, *, ref: str, subject: str, queue: str, type_: str,
                now: float | None = None) -> dict:
        now = time.time() if now is None else now
        with self._lock:
            self._evict(now)
            matches = [e for e in self._items if float(e.emb @ emb) >= self.threshold]
            size = len(matches) + 1
            if type_ in ("Incident", "Problem"):
                self._items.append(_Entry(now, ref, subject, queue, emb))
        active = size >= self.min_tickets
        queues = Counter([m.queue for m in matches] + [queue])
        return {
            "active": active,
            "size": size,
            "window_min": self.window_s / 60,
            "members": [{"ref": m.ref, "subject": m.subject} for m in matches[-6:]],
            "queue": queues.most_common(1)[0][0],
        }

    def reset(self) -> None:
        with self._lock:
            self._items.clear()
