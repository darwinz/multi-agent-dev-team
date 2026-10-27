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
import shutil
import sys

mode, target, root, record_path = sys.argv[1:5]
root = os.path.realpath(root)
loaded = {}

# .pyc freshness is the source's mtime in whole seconds plus its size, so a same-size rewrite within
# a second would run stale bytecode under the new file's hash. Drop every cached workspace module
# (pytest's rewritten test files included) so this run compiles them from what is on disk now.
if sys.pycache_prefix:
    shutil.rmtree(os.path.join(sys.pycache_prefix, root.lstrip(os.sep)), ignore_errors=True)


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
    # Modules loaded by other importers (pytest's assertion-rewriting hook sits ahead of ours and
    # loads conftest.py and test helpers) are swept from sys.modules, hashed at exit
    for module in list(sys.modules.values()):
        path = getattr(module, "__file__", None)
        if path and os.path.realpath(path).startswith(root + os.sep):
            rel = os.path.relpath(os.path.realpath(path), root).replace(os.sep, "/")
            if rel not in loaded:
                _record(path)
    with open(record_path, "w") as f:
        json.dump({"loaded": loaded}, f)
