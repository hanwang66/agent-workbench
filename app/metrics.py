from __future__ import annotations

from dataclasses import dataclass
from threading import Lock


@dataclass
class _Timer:
    count: int = 0
    total_seconds: float = 0.0


def _escape_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


class MetricsCollector:
    """Small Prometheus-compatible collector with no runtime dependency."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._requests: dict[tuple[str, str, str], int] = {}
        self._durations: dict[tuple[str, str], _Timer] = {}

    def observe_http(self, method: str, path: str, status_code: int | str, duration_seconds: float) -> None:
        request_key = (method.upper(), path, str(status_code))
        duration_key = (method.upper(), path)
        with self._lock:
            self._requests[request_key] = self._requests.get(request_key, 0) + 1
            timer = self._durations.setdefault(duration_key, _Timer())
            timer.count += 1
            timer.total_seconds += max(0.0, duration_seconds)

    def render_prometheus(self) -> str:
        with self._lock:
            requests = sorted(self._requests.items())
            durations = sorted(self._durations.items())

        lines = [
            "# HELP agent_workbench_http_requests_total Total HTTP requests.",
            "# TYPE agent_workbench_http_requests_total counter",
        ]
        for (method, path, status), count in requests:
            labels = (
                f'method="{_escape_label(method)}",'
                f'path="{_escape_label(path)}",'
                f'status="{_escape_label(status)}"'
            )
            lines.append(f"agent_workbench_http_requests_total{{{labels}}} {count}")

        lines.extend(
            [
                "# HELP agent_workbench_http_request_duration_seconds HTTP request duration summary.",
                "# TYPE agent_workbench_http_request_duration_seconds summary",
            ]
        )
        for (method, path), timer in durations:
            labels = f'method="{_escape_label(method)}",path="{_escape_label(path)}"'
            lines.append(
                f"agent_workbench_http_request_duration_seconds_sum{{{labels}}} "
                f"{timer.total_seconds:.6f}"
            )
            lines.append(
                f"agent_workbench_http_request_duration_seconds_count{{{labels}}} {timer.count}"
            )
        return "\n".join(lines) + "\n"


__all__ = ["MetricsCollector"]
