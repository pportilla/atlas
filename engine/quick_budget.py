"""The time budget of a quick search, and the limits that are parts of it.

A quick search is the same search as an exhaustive one, under a time
budget. Whoever runs the search in a process of its own enforces the budget
(`resolve_cli.run_job`, the native supervisor of the command line, a search
job): the process is stopped when the budget ends, and the answer is the
record of `expired`. That record is a normal answer. It says how long the
search ran and what it was doing, and that a search without a budget can go
on.

The fast methods come first and have limits of their own. Each limit is a
share of the budget, so that a search with any budget reaches the safe
method and still has time to certify what it finds. The shares were chosen
on the screen of 2,000 fourfolds and fivefolds described in
docs/thematic/resolution-pipeline.md#quick-search.
"""
import json
from math import isfinite

# Seconds of a quick search when the request names no budget.
QUICK_SECONDS = 20

# The share of the budget for the point repairs of the Newton method, before
# the germ is handed over to the smooth-centre rules. No germ of the sample
# needed more of them, and germs that a later method resolves wait for it.
NEWTON_SHARE = 0.2
# The share for the five prescribed centres of the repeated-tangent rule.
TANGENT_SHARE = 0.25
# The share for the weighted construction, and again for its certificate,
# when the weighted method was asked for.
WEIGHTED_SHARE = 0.25
# The share kept for the smooth-centre rules when they must replace a
# proposal whose verification has not finished: the verification of the
# first proposal ends when this much of the budget is left.
FALLBACK_SHARE = 0.45
# The share for the optional search for nearby strata, after the zeta
# functions are computed.
NEARBY_SHARE = 0.15
# Seconds kept at the end of the budget for writing the answer. An optional
# search never runs into them.
FINAL_SECONDS = 1.5
# The share of the first turn of each smooth-centre rule. The turns double.
TURN_SHARE = 0.25

# The total that the limits of a bounded attempt were sized for. Such an
# attempt is made inside another computation and keeps its fixed limits.
BOUNDED_SECONDS = 55


def checked_seconds(value):
    """The budget `value` as a number of seconds, or a ValueError that says what a budget is."""
    if type(value) not in (int, float) or not isfinite(value) or value <= 0:
        raise ValueError("The time budget of a quick search is a positive number of seconds.")
    return value


def request_seconds(request):
    """The budget of a resolution request in seconds, or None when its search has none.

    `seconds` in the request names the budget and makes the search a quick
    one. An exhaustive search has no budget, so asking for both is an error.
    """
    seconds = request.get("seconds")
    search = request.get("search")
    if seconds is not None:
        if search == "exhaustive":
            raise ValueError("An exhaustive search has no time budget: give seconds with a quick search.")
        return checked_seconds(seconds)
    return QUICK_SECONDS if search == "quick" else None


def with_budget(request):
    """The request with its search policy and budget written out: `seconds` implies a quick search."""
    if request.get("seconds") is None:
        return request
    request_seconds(request)
    return dict(request, search="quick")


def read_progress(path):
    """The last progress that a search published in the file `path`, or None."""
    try:
        with open(path, "rb") as stream:
            snapshot = json.loads(stream.read())
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(snapshot, dict) or not isinstance(snapshot.get("stage"), str):
        return None
    progress = {"stage": snapshot["stage"]}
    if isinstance(snapshot.get("method"), str):
        progress["method"] = snapshot["method"]
    if type(snapshot.get("blowups")) is int and snapshot["blowups"] >= 0:
        progress["blowups"] = snapshot["blowups"]
    return progress


def summary(seconds):
    return f"Not resolved within {seconds:g} second{'' if seconds == 1 else 's'}."


def _doing(progress):
    """What the search was doing when it last reported, as a sentence, or ""."""
    if not progress or not progress.get("stage"):
        return ""
    stage, done = progress["stage"], progress.get("blowups")
    after = "" if done is None else f", after {done} blowup{'' if done == 1 else 's'}"
    if stage.split(" ", 1)[0].endswith("ing"):
        return f" Stopped while {stage[:1].lower()}{stage[1:]}{after}."
    return f" Its last stage: {stage}{after}."


def expired(request, seconds, elapsed=None, progress=None):
    """The answer of a quick search whose budget ended: (200, record).

    The record keeps `status: "inconclusive"` and `can_search`, as every
    answer without a certified resolution does. `budget` gives the budget and
    the time used in seconds, and the stage that the search had reported
    last, with the number of blowups it had made when it reported one.
    `summary` says that the budget ended and `diagnostic.description` adds
    what the search was doing. Nothing in the record is a method's reason.
    """
    from .resolve import inconclusive
    elapsed = seconds if elapsed is None else elapsed
    text = summary(seconds)
    budget = {"seconds": seconds, "elapsed_seconds": round(elapsed, 3), "expired": True}
    budget.update(progress or {})
    record = inconclusive(request, {"code": "budget_expired", "description": text + _doing(progress)}, 1000 * elapsed)
    record.update(engine="quick-search", summary=text, budget=budget)
    for key, value in record.items():
        if key.endswith("_zeta") and isinstance(value, dict) and "reason" in value:
            value["reason"] = text
    return 200, record


def expired_line(record):
    """One line for a person: what the search was doing when its budget ended, and how to go on."""
    return f"{record['diagnostic']['description']} Run without --quick to continue the search."
