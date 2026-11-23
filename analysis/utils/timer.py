"""
Decorator for timing stages
"""
import time

def format_duration(seconds: float) -> str:
    """Human-readable duration: "12.3s" below a minute, otherwise "3m 05s"."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(seconds, 60)
    return f"{int(minutes)}m {secs:.0f}s"

def timed(label_fn):
    """
    Decorator factory for methods: print "<label> done in <duration>" after each call.

    `label_fn(self, *args, **kwargs)` builds the label, so it can name the stage and study being run.
    """
    def decorator(fn):
        def wrapper(self, *args, **kwargs):
            """Call the wrapped method and report how long it took."""
            start = time.perf_counter()
            result = fn(self, *args, **kwargs)
            print(f"{label_fn(self, *args, **kwargs)} done in {format_duration(time.perf_counter() - start)}")
            return result
        return wrapper
    return decorator