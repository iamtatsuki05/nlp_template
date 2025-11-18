import asyncio
import inspect
from unittest.mock import AsyncMock

import pytest

from nlp.common.utils.async_utils import (
    AsyncResource,
    async_to_sync_func,
    run_async_function_with_semaphore,
    sync_to_async_func,
)


def test_sync_to_async_func_preserves_metadata() -> None:
    """Test that sync_to_async_func preserves the function name and docstring."""

    def sample_func(x: int) -> int:
        """Sample docstring."""
        return x * 2

    async_func = sync_to_async_func(sample_func)

    assert async_func.__name__ == sample_func.__name__
    assert async_func.__doc__ == sample_func.__doc__


@pytest.mark.parametrize(
    ('input_value', 'expected'),
    [
        (1, 2),
        (5, 10),
        (0, 0),
        (-3, -6),
    ],
)
@pytest.mark.asyncio
async def test_sync_to_async_func_results(input_value: int, expected: int) -> None:
    """Test that sync_to_async_func correctly executes the wrapped function."""

    def multiply_by_two(x: int) -> int:
        return x * 2

    async_multiply = sync_to_async_func(multiply_by_two)
    result = await async_multiply(input_value)

    assert result == expected


def test_async_to_sync_func_preserves_metadata() -> None:
    """Test that async_to_sync_func preserves the function name and docstring."""

    async def sample_async_func(x: int) -> int:
        """Sample async docstring."""
        return x * 2

    sync_func = async_to_sync_func(sample_async_func)

    assert sync_func.__name__ == sample_async_func.__name__
    assert sync_func.__doc__ == sample_async_func.__doc__


@pytest.mark.parametrize(
    ('input_value', 'expected'),
    [
        (1, 2),
        (5, 10),
        (0, 0),
        (-3, -6),
    ],
)
def test_async_to_sync_func_results(input_value: int, expected: int) -> None:
    """Test that async_to_sync_func correctly executes the wrapped function."""

    async def async_multiply_by_two(x: int) -> int:
        await asyncio.sleep(0.01)  # Small delay to simulate async work
        return x * 2

    sync_multiply = async_to_sync_func(async_multiply_by_two)
    result = sync_multiply(input_value)

    assert result == expected


@pytest.mark.parametrize('use_semaphore_tuple', [(True,), (False,)])
@pytest.mark.asyncio
async def test_run_async_function_with_semaphore(use_semaphore_tuple: tuple[bool, ...]) -> None:
    """Test that run_async_function_with_semaphore correctly manages the semaphore."""
    use_semaphore = use_semaphore_tuple[0]
    mock_async_func = AsyncMock(return_value='result')
    mock_semaphore: AsyncMock | None = AsyncMock() if use_semaphore else None

    # If a semaphore is provided, it should be used via async with
    if use_semaphore:
        assert mock_semaphore is not None
        mock_semaphore.__aenter__ = AsyncMock()
        mock_semaphore.__aexit__ = AsyncMock()

    result = await run_async_function_with_semaphore(mock_async_func, mock_semaphore, 'arg1', 'arg2', kwarg1='kwarg1')

    # Verify the async function was called with the correct arguments
    mock_async_func.assert_called_once_with('arg1', 'arg2', kwarg1='kwarg1')

    # If a semaphore was provided, verify it was acquired and released
    if use_semaphore:
        assert mock_semaphore is not None
        mock_semaphore.__aenter__.assert_called_once()
        mock_semaphore.__aexit__.assert_called_once()

    assert result == 'result'


class TestAsyncResource:
    class TestResource(AsyncResource):
        """Test implementation of AsyncResource."""

        def __init__(self, concurrency: int = 1) -> None:
            super().__init__(concurrency=concurrency)
            self.call_mock = AsyncMock()

        async def call(self, *args: object, **kwargs: object) -> str:
            return await self.call_mock(*args, **kwargs)

    @pytest.mark.parametrize(
        ('concurrency', 'num_tasks'),
        [
            (1, 5),  # Single concurrency should process one at a time
            (3, 5),  # Higher concurrency allows multiple tasks
        ],
    )
    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrency(self, concurrency: int, num_tasks: int) -> None:
        """Test that AsyncResource correctly limits concurrency using its semaphore."""
        resource = self.TestResource(concurrency=concurrency)
        max_concurrent = 0
        current_concurrent = 0

        async def tracked_call(*_args: object, **_kwargs: object) -> str:
            nonlocal current_concurrent, max_concurrent
            current_concurrent += 1
            max_concurrent = max(max_concurrent, current_concurrent)
            try:
                await asyncio.sleep(0.01)
                return 'test_result'
            finally:
                current_concurrent -= 1

        resource.call_mock.side_effect = tracked_call

        # Create and gather multiple tasks
        tasks = [resource.task(f'arg{i}') for i in range(num_tasks)]
        results = await asyncio.gather(*tasks)

        # Verify all tasks completed successfully
        assert all(result == 'test_result' for result in results)

        # Verify the concurrency limit was respected
        assert max_concurrent <= concurrency

        # Verify the call method was called with the correct arguments
        assert resource.call_mock.call_count == num_tasks
        for i in range(num_tasks):
            resource.call_mock.assert_any_call(f'arg{i}')

    @pytest.mark.asyncio
    async def test_abstract_call_method(self) -> None:
        """Test that AsyncResource.call is abstract and must be implemented."""
        assert inspect.isabstract(AsyncResource)
