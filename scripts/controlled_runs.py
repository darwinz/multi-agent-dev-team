"""Generate the controlled-run evidence (C3.2, C3.3, C3.5): `uv run python -m scripts.controlled_runs`.

Every scenario uses scripted PM/Coder/QA models; pytest runs for real in a subprocess.
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
from orchestration.a2a import Broker  # noqa: E402
from tests.scenarios import (  # noqa: E402
    feedback_scripts,
    forged_broker,
    nonconverging_scripts,
    passing_scripts,
    scripted_runner,
)

DEFAULT_OUT = Path("docs/evidence/v3/controlled")
LABEL = "scripted PM/Coder/QA models, real pytest subprocess execution"
MODEL_NOTE = "Model names in transcript.txt are config defaults; no model API calls were made."
SCRIPTED_MODEL = "scripted"  # Settings.model is shown in the transcript; no model is ever built from it
REQUIREMENT = ("Implement a Python module that contains a linked list class with insert, delete, search, and "
               "reverse methods. All methods must be covered by tests.")
QA_ROUNDS_NONCONVERGING = 2

# name -> (scripts factory, QA iteration limit, what the run shows, broker factory or None)
SCENARIOS: dict[str, tuple[Callable[[], tuple[list, list, list]], int, str, Callable[[], Broker] | None]] = {
    "feedback": (feedback_scripts, 5,
                 "Round 1 ships a linked list whose `reverse` is a no-op, so QA's `test_reverse` fails. "
                 "QA's `fix_instruction` reaches the Coder, round 2 relinks the nodes, and the retest "
                 "passes (`approved`).\n\n"
                 "Chain: `qa-T1-r1.json` (failing assertion) -> `a2a-log.jsonl` (`fix_instruction`) -> "
                 "`a2a-log.jsonl` (`review_request` round 2, revised file hashes) -> `qa-T1-r2.json` "
                 "(passing).", None),
    "nonconverging": (lambda: nonconverging_scripts(QA_ROUNDS_NONCONVERGING), QA_ROUNDS_NONCONVERGING,
                      "The Coder never fixes `reverse` and `QA_MAX_ITERATIONS=2`. At the limit QA sends a "
                      "`final_report` listing the unresolved issue, the task is `failed` and the run is "
                      "`partial`. No success is claimed.", None),
    "trust-boundary": (passing_scripts, 5,
                       "A correct linked list is approved in round 1, while two forged, schema-valid messages "
                       "wait in the broker. They were injected deliberately, before the run, to show the trust "
                       "boundary: a `fix_instruction` claiming to come from the Coder (QA accepts only "
                       "`review_request` from the Coder) and an `approved` from QA whose `correlation_id` "
                       "answers no outstanding request. Both endpoints reject them, and each rejection is "
                       "logged with its reason as a `rejected` record in `a2a-log.jsonl`. The task still "
                       "closes `done` on the real `approved`.", forged_broker),
}


def _scrub(text: str, roots: list[str]) -> str:
    for root in sorted(roots, key=len, reverse=True):
        text = text.replace(root, "<project>")
    return text


def run_scenario(name: str, out: Path) -> int:
    make_scripts, qa_limit, summary, broker_factory = SCENARIOS[name]
    dest = out / name
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix="controlled-") as tmp:
        project = Path(tmp)
        settings = Settings.from_env(project_dir=project, model=SCRIPTED_MODEL, exec_timeout_s=15,
                                     qa_max_iterations=qa_limit)
        extra = {"broker_factory": broker_factory} if broker_factory else {}
        runner = scripted_runner(settings, make_scripts(), **extra)
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
        f"QA iteration limit: {qa_limit}\nExit code: {code}\n\n{summary}\n\n{MODEL_NOTE}\n\n"
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
    # feedback and trust-boundary must succeed (0) and nonconverging must fail (1): that is the evidence
    return 0 if codes == {"feedback": 0, "nonconverging": 1, "trust-boundary": 0} else 1


if __name__ == "__main__":
    sys.exit(main())
