import importlib
import inspect
import sys
from pathlib import Path
from typing import Callable


def import_function(function_file_path: str, function_name: str | None = None) -> Callable:
    function_file_path = Path(function_file_path).resolve()
    function_name = function_name or function_file_path.stem

    project_root = Path.cwd()
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))

    module_path = '.'.join(function_file_path.relative_to(project_root).with_suffix('').parts)
    module = importlib.import_module(module_path)
    return getattr(module, function_name)


def get_imported_function_path(obj: Callable) -> str:
    return inspect.getfile(obj)
