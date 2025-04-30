import asyncio
from typing import Any, Callable
from abc import ABC, abstractmethod


def sync_to_async_func(sync_func: Callable) -> Callable:
    """
    同期関数を非同期関数として使えるように変換する
    """

    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        return await asyncio.to_thread(sync_func, *args, **kwargs)

    wrapper.__name__ = sync_func.__name__
    wrapper.__doc__ = sync_func.__doc__
    return wrapper


def async_to_sync_func(async_func: Callable) -> Callable:
    """
    非同期関数を同期関数として使えるように変換する
    """

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return asyncio.run(async_func(*args, **kwargs))

    wrapper.__name__ = async_func.__name__
    wrapper.__doc__ = async_func.__doc__
    return wrapper


async def run_async_function_with_semaphore(
    async_func: Callable, concurrency_sema: asyncio.Semaphore | None, *args: Any, **kwargs: Any
) -> Any:
    """
    指定した関数 func を、セマフォで同時実行数を制限して呼び出す関数。
    concurrency_sema が None の場合は制限しない。
    """
    if concurrency_sema is not None:
        async with concurrency_sema:
            return await async_func(*args, **kwargs)
    else:
        return await async_func(*args, **kwargs)

class AsyncResource(ABC):
    def __init__(self, concurrency: int = 1) -> None:
        self.semaphore = asyncio.Semaphore(concurrency)

    async def task(self, *args, **kwargs) -> Any:
        async with self.semaphore:
            return await self.call(*args, **kwargs)

    @abstractmethod
    async def call(self, *args, **kwargs) -> Any:
        pass
