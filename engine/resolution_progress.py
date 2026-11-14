"""Publish the worker's current operation for the web job that launched it."""
import json
import os


def report(stage, blowups=None):
    target = os.environ.get("ATLAS_PROGRESS_FILE")
    if not target:
        return
    snapshot = {"stage": stage}
    method = os.environ.get("ATLAS_PROGRESS_METHOD")
    if method:
        snapshot["method"] = method
    if blowups is not None:
        snapshot["blowups"] = blowups
    # The server reads while workers compute. Publish only complete snapshots.
    temporary = f"{target}.{os.getpid()}.tmp"
    try:
        with open(temporary, "w", encoding="utf-8") as stream:
            json.dump(snapshot, stream)
        os.replace(temporary, target)
    except OSError:
        pass  # Losing a progress update must never interrupt a calculation.


def recorded_blowups(record):
    """Toric modifications need not specify a factorization into blowups."""
    steps = record.get("steps")
    if steps is None or record.get("status") != "resolved":
        return None
    if all("center" in step or "center_charts" in step or "blowup" in step.get("type", "") for step in steps):
        return len(steps)
    return None
