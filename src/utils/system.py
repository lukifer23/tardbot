"""Device and system monitoring helpers for on-device training."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional

import psutil
import torch


@dataclass
class SystemStats:
    cpu_percent: float
    ram_percent: float
    ram_available_gb: float
    swap_percent: float
    device_allocated_gb: float
    device_reserved_gb: float
    timestamp: float


def _mps_available() -> bool:
    return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())  # type: ignore[attr-defined]


def get_preferred_device() -> torch.device:
    if _mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def _device_memory(device: torch.device) -> Dict[str, float]:
    if device.type == "mps" and _mps_available():
        allocated = torch.mps.current_allocated_memory() / (1024 ** 3)  # type: ignore[attr-defined]
        reserved = torch.mps.driver_allocated_memory() / (1024 ** 3)  # type: ignore[attr-defined]
        return {"allocated": allocated, "reserved": reserved}
    return {"allocated": 0.0, "reserved": 0.0}


def collect_system_stats(device: Optional[torch.device] = None) -> SystemStats:
    device = device or get_preferred_device()
    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()
    device_mem = _device_memory(device)

    return SystemStats(
        cpu_percent=psutil.cpu_percent(interval=None),
        ram_percent=vm.percent,
        ram_available_gb=vm.available / (1024 ** 3),
        swap_percent=swap.percent,
        device_allocated_gb=device_mem["allocated"],
        device_reserved_gb=device_mem["reserved"],
        timestamp=time.time(),
    )


def format_system_stats(prefix: str, stats: SystemStats) -> str:
    return (
        f"{prefix} CPU:{stats.cpu_percent:.1f}% RAM:{stats.ram_percent:.1f}% "
        f"({stats.ram_available_gb:.1f} GB free) Device:{stats.device_allocated_gb:.2f}G/"
        f"{stats.device_reserved_gb:.2f}G"
    )
