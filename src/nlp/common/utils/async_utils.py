import asyncio
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Coroutine
from typing import Any


def sync_to_async_func[R](sync_func: Callable[..., R]) -> Callable[..., Awaitable[R]]:
    """Convert a synchronous callable into an asynchronous callable."""

    async def wrapper(*args: object, **kwargs: object) -> R:
        return await asyncio.to_thread(sync_func, *args, **kwargs)

    wrapper.__name__ = sync_func.__name__
    wrapper.__doc__ = sync_func.__doc__
    return wrapper


def async_to_sync_func[R](async_func: Callable[..., Coroutine[Any, Any, R]]) -> Callable[..., R]:
    """Convert an asynchronous callable into a synchronous callable."""

    def wrapper(*args: object, **kwargs: object) -> R:
        return asyncio.run(async_func(*args, **kwargs))

    wrapper.__name__ = async_func.__name__
    wrapper.__doc__ = async_func.__doc__
    return wrapper


async def run_async_function_with_semaphore[R](
    async_func: Callable[..., Awaitable[R]],
    concurrency_sema: asyncio.Semaphore | None,
    *args: object,
    **kwargs: object,
) -> R:
    """Execute async_func with an optional semaphore limiting concurrency."""
    if concurrency_sema is not None:
        async with concurrency_sema:
            return await async_func(*args, **kwargs)
    return await async_func(*args, **kwargs)


class AsyncResource[R](ABC):
    """Base class for async resources protected by a semaphore."""

    def __init__(self, concurrency: int = 1) -> None:
        self.semaphore = asyncio.Semaphore(concurrency)

    async def task(self, *args: object, **kwargs: object) -> R:
        async with self.semaphore:
            return await self.call(*args, **kwargs)

    @abstractmethod
    async def call(self, *args: object, **kwargs: object) -> R:
        """Execute the concrete asynchronous operation."""
        raise NotImplementedError
