"""
Miscellaneous utilities.
"""
import hashlib
import pickle
from pathlib import Path
from functools import wraps


def progress_bar(step, total_steps, bar_length=30, fill='#', end_text=''):
    """
    Simple progress bar.
    """
    filled = int(bar_length * step / total_steps)
    text = f"[{filled * fill :<{bar_length}}] {step}/{total_steps}"
    end = '\r' if step < total_steps else '\n' 
    print(text + end_text, end=end)


def disk_cache(cache_dir):
    """
    A decorator to automatically cache experiment results (as .pkl files).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    def decorator(func):
        @wraps(func)
        def wrapper(*args, force=False, **kwargs):

            key_bytes = pickle.dumps((args, kwargs))
            key = hashlib.md5(key_bytes).hexdigest()

            file_path = cache_dir / f"{func.__name__}_{key}.pkl"

            if file_path.exists() and not force:
                with open(file_path, "rb") as f:
                    return pickle.load(f)

            result = func(*args, **kwargs)

            with open(file_path, "wb") as f:
                pickle.dump(result, f)

            return result

        return wrapper

    return decorator
