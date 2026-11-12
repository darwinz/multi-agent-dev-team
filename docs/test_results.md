# End-to-end test results

Module 5 asks for the full pipeline run on at least three different software-development tasks, with the failures, the fixes and the final pass/fail status recorded. This file covers the four live demos on the Module 5 code, the failure one of them hit and how it was fixed, and the failures from Module 4 that shaped the current code.

All live runs use the production configuration: Sonnet 5.5 for the PM, Haiku 5.5 for the Coder and QA, real API calls, real subprocess execution, and QA's real pytest runs. Every run's records (cost report, final state, structured event log, QA reports) are in [`evidence/v5/verify-runs/`](evidence/v5/verify-runs/).

## Final status

| Task | What it asks for | Run | Result | Tasks (QA round) | Cost |
|---|---|---|---|---|---:|
| **REST API** | FastAPI app with `GET /health` and `POST /echo`, input validation, 422 on bad input | [`066911bc96ca`](evidence/v5/verify-runs/api-rerun/) | **Pass** | 3/3 approved (round 1) | $0.0423 |
| **Large requirement** | To-do library with 11 separately tested features; forces task consolidation | [`d7cf5703df2c`](evidence/v5/verify-runs/large/) | **Pass** | 11 planned → 5 merged, 5/5 approved (round 1) | $0.1330 |
| **Linked list** | The Module 3 grading task: insert, delete, search, reverse, all tested | [`e3900583fb79`](evidence/v5/verify-runs/linked-list/) | **Pass** | 5/5 approved (round 1) | $0.1093 |
| **Binary search tree** | The Module 4 grading task, checked against the supplied fixture | [`31bf667cbc16`](evidence/v5/verify-runs/bst/) | **Pass** | 3/3 approved (round 1) | $0.0584 |

**Code version.** All four ran on the Module 5 code at `c96876e`. The API rerun also included the QA fix described below. The large, linked-list and BST runs came before that fix, which adds two rules to QA's test-writing prompt and changes one retry message. Nothing else differs.

Each row is a live test in [`tests/live/test_team_demo.py`](../tests/live/test_team_demo.py), and each test checks more than the run status:

- **REST API:** the PM planned at least 2 tasks, the run is `completed`, and an independent checker (`app/api_check.py`) starts the delivered app and calls every endpoint, including the 422 cases.
- **Large requirement:** the plan was consolidated, the executed tasks are within the task limit, and every spec criterion is covered by an executed task.
- **Linked list:** the run is `completed`, the demo check passes, and the agent-to-agent log pairs every review request with an approval.
- **BST:** the run is `completed` and the supplied fixture [`tests/fixtures/bst_fixture.py`](../tests/fixtures/bst_fixture.py) passes against the delivered `bst.py`.

**Other checks on the same code:**

| Check | Result | Log |
|---|---|---|
| Mocked suite (`make test`) | 523 passed, 0 failed (100%), 7 min | [`mocked-suite.txt`](evidence/v5/mocked-suite.txt) |
| Live suite (`make verify`): 2 smoke tests + the 4 demos | 5 passed, 1 failed (the API demo, below); the API demo then passed on its own after the fix | [`make-verify.log`](evidence/v5/make-verify.log), [`api-demo-rerun.log`](evidence/v5/api-demo-rerun.log) |
| Lint (`make lint`) | ruff: all checks passed | [`ruff.txt`](evidence/v5/ruff.txt) |
| Secret scan (`make secrets`) | gitleaks: 108 commits on every branch and the working tree, no leaks. [`.gitleaks.toml`](../.gitleaks.toml) allows one pattern only: the SHA-256 file hashes in QA reports, which the generic-api-key rule matched because a hashed file was named `test_api.py` | [`secrets-scan.txt`](evidence/v5/secrets-scan.txt) |

## Failure found in Module 5, and the fix

### REST API demo, run `f92f9ac42b5e`: `partial`

In the first `make verify` run, T1 and T3 were approved but T2 ended with `qa_error`, so the run was `partial` and the test failed. T2's criteria were: a nested object is echoed unchanged, `{}` is echoed, `GET /echo` returns 405, and **"Tests pass with pytest."**

What happened, from the cached QA replies:

1. **QA's first suite** had no test of its own for criterion 4. Instead it listed its three existing tests a second time, under criterion 4. The suite validator ([`orchestration/review.py`](../orchestration/review.py)) allows one criterion per test, so it rejected the suite with "duplicate test name in the manifest".
2. **The retry message** passed that problem back to the model. The hint explaining how to test a criterion about the tests themselves was only added when a criterion had *no* test, which wasn't the case here, so the model wasn't told how to cover criterion 4.
3. **QA's second suite** removed the duplicates, which left criterion 4 with no test at all. The validator rejected it again, and the task ended with `qa_error`.

The Coder also used all 10 of its iterations on T2 without calling `finish`, although its own tests had passed (5/5). That did not decide the outcome, because QA still ran on the delivered code. QA's invalid suite is what ended the task.

**Fix:**

- **[`agents/prompts.py`](../agents/prompts.py):** QA's suite prompt now says that each test maps to exactly one criterion, and that a criterion about the tests themselves gets its own test, which runs the Coder's test file with pytest in a subprocess.
- **[`agents/qa_agent.py`](../agents/qa_agent.py):** when a suite lists a test under two criteria, the retry message now explains the one-criterion rule and includes the same hint.
- **New tests** in [`tests/test_qa_agent.py`](../tests/test_qa_agent.py):
  - `test_test_mapped_to_two_criteria_retry_explains_one_criterion_per_test` reproduces the live failure with a scripted model;
  - `test_suite_prompt_maps_each_test_to_one_criterion` checks the prompt rule.

  Both failed before the change and pass after it.

**Rerun:** run [`066911bc96ca`](evidence/v5/verify-runs/api-rerun/) completed, with all three tasks approved in round 1. All three tasks had a "Tests pass with pytest" criterion, and each QA suite covered it with its own subprocess test ([`qa-tests-T2.py`](evidence/v5/verify-runs/api-rerun/qa-tests-T2.py)).

This is the second time this criterion has caused a failure (see `2e936a27d395` below). The Module 4 fix only took effect when QA left the criterion without a test. This time the first mistake was a different one, so that fix never applied. The rule is now in the prompt, so QA's first attempt has it too.

## Failures found in Module 4, and their fixes

These runs are on earlier code. They are listed because their fixes are part of the code tested above. Their records are in [`evidence/v4/`](evidence/v4/README.md#live-runs).

| Run | Failure | Fix |
|---|---|---|
| `4ef15a90745a` | `unresolved`: the final regression check timed out while the machine was out of memory and disk | None needed in the code: rerun once the machine had recovered |
| `b52d80ca7976` (BST) | `unresolved`: QA's suite reply hit Haiku's 8,192-token output limit and was cut off, and the retry then acted on a misleading parse error | `2cf61e9`: detect a cut-off reply, retry with a "write a shorter suite" ask, never cache a cut-off reply, `QA_MAX_TOKENS` default 16384 |
| `2e936a27d395` (BST) | `unresolved`: QA left the criterion "tests pass with pytest" without a test on both attempts | `36abc89`, `bf20077`: the retry quotes the uncovered criterion and explains how to test a criterion about the tests |
| `make verify` | All six live tests errored at setup, before any model call: an autouse fixture returned before its `yield` for live tests | `e909707` |

The graded BST run `e0d27f3759ec` then passed: 5/5 tasks approved, fixture 5/5, 63/63 regression tests, $0.1437.

## Cost of these runs

| Run | Calls | Prompt tokens | Output tokens | Cost |
|---|---:|---:|---:|---:|
| API `f92f9ac42b5e` (failed) | 44 | 360,458 | 31,239 | $0.0444 |
| Large `d7cf5703df2c` | 53 | 986,733 | 94,065 | $0.1330 |
| Linked list `e3900583fb79` | 50 | 898,947 | 95,008 | $0.1093 |
| BST `31bf667cbc16` | 28 | 246,731 | 39,830 | $0.0584 |
| API rerun `066911bc96ca` | 34 | 261,642 | 28,259 | $0.0423 |
| **Total** | 209 | 2,754,511 | 288,401 | **$0.3874** |

The two smoke tests add a few cents and write no cost report. The same four demos cost $0.7383 in Module 4's `make verify`. The Coder's history caching ([PERFORMANCE.md](PERFORMANCE.md)) is part of that drop, but the PM's task plans also differ from run to run, so the two totals are not a controlled comparison.

## Reproduce

```bash
make test                                            # mocked suite, no key, no spend
op run --env-file=.env.op -- make verify             # the live suite above, about $0.40
op run --env-file=.env.op -- make team-demo          # one demo at a time: team-demo, team-large,
                                                     # team-linked-list, team-bst
make lint
make secrets                                         # needs gitleaks
```

Without 1Password, put `ANTHROPIC_API_KEY` in `.env` (see [`.env.example`](../.env.example)) and drop the `op run … --` prefix.
