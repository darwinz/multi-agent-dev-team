"""Subprocess entry point for exec_python. Runs in the child process, never in the agent.

    python -s _runner.py script|pytest <target> <workspace root> <record json>

Runs the target and records which workspace files it loaded (the target plus every
module imported from the workspace), each with its SHA-256 at load time.
"""

import hashlib
import importlib.machinery
import json
import os
import runpy
import sys

mode, target, root, record_path = sys.argv[1:5]
root = os.path.realpath(root)
loaded = {}


def _record(path):
    real = os.path.realpath(path)
    if real.startswith(root + os.sep) and os.path.isfile(real):
        with open(real, "rb") as f:
            loaded[os.path.relpath(real, root).replace(os.sep, "/")] = hashlib.sha256(f.read()).hexdigest()


class _RecordingFinder:
    """Delegates to the normal path finder and notes workspace modules as they load."""

    @staticmethod
    def find_spec(name, path=None, target=None):
        spec = importlib.machinery.PathFinder.find_spec(name, path, target)
        if spec is not None and spec.origin:
            _record(spec.origin)
        return spec

    @staticmethod
    def invalidate_caches():
        importlib.machinery.PathFinder.invalidate_caches()


# sys.path[0] is this runner's own tools/ dir; the target's dir replaces it so
# workspace modules aren't shadowed by ours and sibling imports work.
sys.path[0] = os.path.dirname(os.path.abspath(target))
sys.argv = [target]
_record(target)
sys.meta_path.insert(0, _RecordingFinder)

try:
    if mode == "pytest":
        import pytest

        sys.exit(pytest.main(["-q", "-p", "no:cacheprovider", f"--rootdir={root}", target]))
    runpy.run_path(target, run_name="__main__")
finally:
    with open(record_path, "w") as f:
        json.dump({"loaded": loaded}, f)
