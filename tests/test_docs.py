import importlib
import inspect
import pkgutil
from pathlib import Path

import pytest

import velox_tools

DOCS = Path(__file__).parents[1] / 'docs'
MODULES = [m.name for m in pkgutil.iter_modules(velox_tools.__path__) if m.name != 'notebooks']


@pytest.mark.parametrize('module', MODULES)
def test_public_api_is_documented(module):
    """Every public function and class is listed in the API overview and
    has a page (docs/api/<module>.md)."""
    mod = importlib.import_module(f'velox_tools.{module}')
    public = [name for name, obj in vars(mod).items()
              if not name.startswith('_') and (inspect.isfunction(obj) or inspect.isclass(obj))
              and obj.__module__ == mod.__name__]
    overview = (DOCS / 'api' / 'index.md').read_text()
    assert (DOCS / 'api' / f'{module}.md').exists()
    missing = [name for name in public if f'{module}.{name}\n' not in overview]
    assert not missing, f'add to docs/api/index.md: {missing}'
