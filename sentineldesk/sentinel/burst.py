"""Live outage / burst detection.

Every triaged ticket is compared with tickets from the last N minutes. When enough
of them describe the same failure, a major-incident alert is raised.
"""
from __future__ import annotations

import threading
import time
from collections import deque

from . import config
from .retrieval import similarity
from .textutils import keywords


class BurstDetector:
    def __init__(self, sim_threshold: float, window_minutes: int = config.BURST_WINDOW_MIN,
                 min_tickets: int = config.BURST_MIN_TICKETS, max_items: int = 500) -> None:
        self.sim_threshold = sim_threshold
        self.window_s = window_minutes * 60
        self.min_tickets = min_tickets
        self.items: deque = deque(maxlen=max_items)
        self.alert: dict | None = None
        self._lock = threading.Lock()  # Streamlit sessions share this object

    def add(self, vec, info: dict, now: float | None = None) -> dict | None:
        now = now or time.time()
        with self._lock:
            recent = [it for it in self.items if it["ts"] >= now - self.window_s]
            similar = [it for it in recent if similarity(vec, it["vec"]) >= self.sim_threshold]
            item = {"ts": now, "vec": vec, **info}
            self.items.append(item)
            cluster = similar + [item]
            if len(cluster) >= self.min_tickets:
                self.alert = {
                    "ts": now,
                    "size": len(cluster),
                    "keywords": keywords([c.get("text", "") for c in cluster]),
                    "subjects": [c.get("subject", "") for c in cluster][-5:],
                }
                return self.alert
        return None

    def current_alert(self, now: float | None = None) -> dict | None:
        now = now or time.time()
        if self.alert and now - self.alert["ts"] <= self.window_s:
            return self.alert
        return None

    def clear(self) -> None:
        with self._lock:
            self.items.clear()
            self.alert = None

    def log(self) -> list[dict]:
        rows = []
        for it in reversed(self.items):
            rows.append({
                "time": time.strftime("%H:%M:%S", time.localtime(it["ts"])),
                "subject": it.get("subject", ""),
                "type": it.get("type", ""),
                "priority": it.get("priority", ""),
                "status": it.get("status", ""),
            })
        return rows
