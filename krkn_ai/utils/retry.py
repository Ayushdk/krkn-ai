import time
from typing import Callable, Optional, TypeVar

T = TypeVar("T")


def retry_with_backoff(
    func: Callable[[], T],
    retries: int,
    base_delay: int = 1,
    exponential: bool = True,
    on_exception: Optional[Callable[[Exception, int], None]] = None,
) -> T:
    last_error: Optional[Exception] = None

    for attempt in range(1, retries + 1):
        try:
            return func()
        except Exception as exc:
            last_error = exc

            if on_exception:
                on_exception(exc, attempt)

            if attempt < retries:
                delay = (
                    base_delay * (2 ** (attempt - 1))
                    if exponential
                    else base_delay
                )
                time.sleep(delay)

    raise last_error
