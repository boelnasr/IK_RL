import json
import os
from datetime import datetime
from typing import Dict, Any, Optional


def _write_section(handle, title: str, payload: Dict[str, Any]) -> None:
    handle.write(f"{title}\n")
    handle.write(json.dumps(payload, indent=2, sort_keys=True))
    handle.write("\n\n")


def write_reward_report(
    config_snapshot: Dict[str, Any],
    reward_snapshot: Dict[str, Any],
    env_snapshot: Dict[str, Any],
    *,
    output_dir: Optional[str] = None,
    run_label: Optional[str] = None,
) -> str:
    """
    Persist a text report describing the reward function structure used in a run.
    """
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    target_dir = output_dir or "run_reports"
    os.makedirs(target_dir, exist_ok=True)
    filename = f"reward_report_{timestamp}.txt"
    path = os.path.join(target_dir, filename)

    metadata = {
        "timestamp_utc": timestamp,
        "run_label": run_label or "unknown",
    }

    with open(path, "w", encoding="ascii") as handle:
        _write_section(handle, "# Metadata", metadata)
        _write_section(handle, "# Reward Snapshot", reward_snapshot)
        _write_section(handle, "# Environment Snapshot", env_snapshot)
        _write_section(handle, "# Training Config", config_snapshot)

    return path
