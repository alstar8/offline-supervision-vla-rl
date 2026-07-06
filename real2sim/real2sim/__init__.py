# Installable Real2Sim helpers and validation entrypoints.

from pathlib import Path
import sys


_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_VENDORED_PYTHON_API_ROOT = _PACKAGE_ROOT / "vendor" / "python_api"

if _VENDORED_PYTHON_API_ROOT.is_dir():
    vendored_python_api_root_str = str(_VENDORED_PYTHON_API_ROOT)
    if vendored_python_api_root_str not in sys.path:
        sys.path.insert(0, vendored_python_api_root_str)
