# RAG/TAG Tiger - timer.py
# Copyright (c) 2024 Stuart Riffle
# github.com/stuartriffle/ragtag-tiger

import time
from lograg import lograg_verbose

class TimerScope:
    """An object that measures the time spent inside a scope"""
    def __init__(self, exitfunc=None, **kwargs):
        super().__init__(**kwargs) 
        self.exitfunc = exitfunc
    def __enter__(self):
        self.start_time = time.time()
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.exitfunc and not exc_type:
            elapsed = time.time() - self.start_time
            self.exitfunc(elapsed)

class TimerUntil(TimerScope):
    """Prints total time inside the scope"""
    def __init__(self, msg, prefix="\t...", suffix="", **kwargs):
        super().__init__(lambda elapsed: self.on_exit(elapsed), **kwargs)
        self.prefix = prefix
        self.suffix = suffix
        self.msg    = msg
    def on_exit(self, elapsed):
        lograg_verbose(f"{self.prefix}{self.msg} ({elapsed:.3f} sec){self.suffix}")

def time_since(before):
    return f"{time.time() - before:.3f} sec"

