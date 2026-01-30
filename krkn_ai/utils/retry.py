import time
from typing import Callable, Optional, TypeVar

from krkn_ai.utils.logger import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


def retry_with_backoff(
    func: Callable[[], T],
    *,
    retries: int,
    base_delay: int,
    operation_name: str = "operation",
) -> T:
    last_error: Optional[BaseException] = None

    for attempt in range(1, retries + 1):
        try:
            return func()

        except Exception as exc:
            last_error = exc

            if attempt >= retries:
                break

            delay = base_delay * (2 ** (attempt - 1))

            logger.warning(
                "%s failed (attempt %s/%s): %s. Retrying in %ss",
                operation_name,
                attempt,
                retries,
                exc,
                delay,
            )

            time.sleep(delay)

    assert last_error is not None
    raise last_error
