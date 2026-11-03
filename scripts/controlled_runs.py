"""Generate the controlled-run evidence (C3.2, C3.3): `uv run python -m scripts.controlled_runs`.

Both scenarios use scripted PM/Coder/QA models; pytest runs for real in a subprocess.
Each run goes through `team_cli.main(...)`, its output is saved as `transcript.txt`, and the run
directory's evidence files are copied next to it with the temporary project root replaced by `<project>`.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import shutil
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.config import Settings  # noqa: E402
from app import team_cli  # noqa: E402
from tests.scenarios import feedback_scripts, nonconverging_scripts, scripted_runner  # noqa: E402

DEFAULT_OUT = Path("docs/evidence/v3/controlled")
LABEL = "scripted PM/Coder/QA models, real pytest subprocess execution"
REQUIREMENT = ("Implement a Python module that contains a linked list class with insert, delete, search, and "
               "reverse methods. All methods must be covered by tests.")
QA_ROUNDS_NONCONVERGING = 2

# name -> (scripts factory, QA iteration limit, what the run shows)
SCENARIOS: dict[str, tuple[Callable[[], tuple[list, list, list]], int, str]] = {
    "feedback": (feedback_scripts, 5,
                 "Round 1 ships a linked list whose `reverse` is a no-op, so QA's `test_reverse` fails. "
                 "QA's `fix_instruction` reaches the Coder, round 2 relinks the nodes, and the retest "
                 "passes (`approved`).\n\n"
                 "Chain: `qa-T1-r1.json` (failing assertion) -> `a2a-log.jsonl` (`fix_instruction`) -> "
                 "`a2a-log.jsonl` (`review_request` round 2, revised file hashes) -> `qa-T1-r2.json` "
                 "(passing)."),
    "nonconverging": (lambda: nonconverging_scripts(QA_ROUNDS_NONCONVERGING), QA_ROUNDS_NONCONVERGING,
                      "The Coder never fixes `reverse` and `QA_MAX_ITERATIONS=2`. At the limit QA sends a "
                      "`final_report` listing the unresolved issue, the task is `failed` and the run is "
                      "`partial`. No success is claimed."),
}


def _scrub(text: str, roots: list[str]) -> str:
    for root in sorted(roots, key=len, reverse=True):
        text = text.replace(root, "<project>")
    return text


def run_scenario(name: str, out: Path) -> int:
    make_scripts, qa_limit, summary = SCENARIOS[name]
    dest = out / name
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix="controlled-") as tmp:
        project = Path(tmp)
        settings = Settings.from_env(project_dir=project, exec_timeout_s=15, qa_max_iterations=qa_limit)
        runner = scripted_runner(settings, make_scripts())
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = team_cli.main([REQUIREMENT], runner=runner)
        assert runner.run_dir is not None
        roots = [str(project), str(project.resolve())]
        (dest / "transcript.txt").write_text(_scrub(buf.getvalue(), roots), encoding="utf-8")
        for f in sorted(runner.run_dir.rglob("*")):
            if f.is_file():
                target = dest / f.relative_to(runner.run_dir)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(_scrub(f.read_text(encoding="utf-8"), roots), encoding="utf-8")
    (dest / "README.md").write_text(
        f"# Controlled run: {name}\n\n{LABEL}.\n\nCommand: `uv run python -m scripts.controlled_runs`\n"
        f"QA iteration limit: {qa_limit}\nExit code: {code}\n\n{summary}\n\n"
        "Files: `transcript.txt` is the CLI output; the rest is the run directory "
        "(`<project>` replaces the temporary project root).\n", encoding="utf-8")
    return code


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="controlled_runs", description=__doc__)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"output directory (default {DEFAULT_OUT})")
    args = p.parse_args(argv)
    codes = {name: run_scenario(name, args.out) for name in SCENARIOS}
    for name, code in codes.items():
        print(f"{name}: exit {code}")
    # feedback must succeed (0) and nonconverging must fail (1): that is the evidence
    return 0 if codes == {"feedback": 0, "nonconverging": 1} else 1


if __name__ == "__main__":
    sys.exit(main())
