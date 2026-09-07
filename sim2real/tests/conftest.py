from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)


def pytest_addoption(parser):
    return None


def pytest_collection_modifyitems(config, items):
    return None
