"""
Utility functions.
"""
import time
from contextlib import contextmanager


@contextmanager
def timer(name: str = "Operation"):
    """
    Context manager to time operations.
    
    Args:
        name: Name of the operation being timed
        
    Usage:
        with timer("Training model"):
            model.fit(data)
    """
    start = time.time()
    print(f"[{name}] Starting...")
    try:
        yield
    finally:
        elapsed = time.time() - start
        print(f"[{name}] Completed in {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"

