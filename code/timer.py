import time

def time_since(before):
    return f"{time.time() - before:.3f} sec"

class TimerScope:
    """A C++ style timer for a block of code"""
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
        print(f"{self.prefix}{self.msg} ({elapsed:.3f} sec){self.suffix}")

