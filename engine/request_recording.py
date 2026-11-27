"""Write exact native requests and replies to a corpus for differential replay."""

import json
import os
from pathlib import Path
from threading import get_ident
from time import perf_counter_ns


class PendingRequest:
    def __init__(self, directory, transport, request, **fields):
        self.row = dict(transport=transport, request=request, **fields)
        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        worker = f"{os.getpid()}-{get_ident()}"
        self.stream = output / f"requests-{worker}.jsonl"
        self.pending = output / f"pending-{worker}-{perf_counter_ns()}.json"
        # The supervisor can cancel a losing lane in the middle of a kernel.
        # Leave its input available even when there will never be a reply.
        self.pending.write_text(json.dumps(self.row, separators=(",", ":")), encoding="utf-8")
        self.start = perf_counter_ns()

    def finish(self, response, **fields):
        self.row.update(response=response, elapsed_ns=perf_counter_ns() - self.start, **fields)
        with self.stream.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(self.row, separators=(",", ":")) + "\n")
        self.pending.unlink()
