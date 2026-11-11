# Performance audit and optimisation

Module 5 asks for the highest-cost agent interaction in the Module 4 cost reports, at least one optimisation, and the before/after token delta. The optimisation is **prompt caching of the Coder's conversation history**. In a controlled A/B test on an identical conversation, it cut the tokens billed at the full input price from 101,815 to 22 and the input cost by 65%. A live BST run with the change passed every check.

## 1. Where the money goes

The graded Module 4 BST run [`e0d27f3759ec`](evidence/v4/live-bst/cost-report.json) cost $0.1437. By agent:

| Agent | Model | Calls | Prompt tokens | Read from cache | Cost | Share |
|---|---|---:|---:|---:|---:|---:|
| PM | Sonnet 5.5 | 1 | 1,257 | 0 | $0.0339 | 24% |
| **Coder** | Haiku 5.5 | **56** | **1,148,906** | 596,163 (52%) | **$0.0956** | **67%** |
| QA | Haiku 5.5 | 13 | 64,322 | 6,638 (10%) | $0.0142 | 10% |

The three earlier BST attempts in [`earlier-runs/`](evidence/v4/live-bst/earlier-runs/) have the same shape: the Coder was 42–58% of each run's cost, and in every run it accounted for over 90% of the prompt tokens.

The highest-cost interaction is the Coder's ReAct step, the call it makes on every iteration. The cost is input, not output. 552,743 of the Coder's prompt tokens (48%) were billed at the full input price, and that accounts for $0.055 of the Coder's $0.096.

My earlier notes named QA's test-suite generation as the likely target, because it once hit the 8,192-token output cap. The cost reports don't support that: QA is 10% of the run.

## 2. Why the Coder was expensive

Each ReAct iteration resends the whole conversation so far: the task, every earlier tool call and every tool result. A Coder session's prompt grows from about 2K to 27K tokens over 10–14 calls.

Module 4 placed two Anthropic prompt-cache breakpoints, one on the last tool definition and one on the system prompt. The conversation history after them had no breakpoint, so it was never cached. The per-call record shows this. Here is one session from the graded run:

| Call | Prompt tokens | Read from cache |
|---:|---:|---:|
| 1 | 15,534 | 10,243 |
| 2 | 22,571 | 10,243 |
| 3 | 23,542 | 10,243 |
| … | … | 10,243 |
| 9 | 27,482 | 10,243 |

The cached amount stays flat at the size of tools plus system prompt, while everything the conversation adds after it is billed in full on every call.

## 3. The change

In [`agents/model.py`](../agents/model.py), `ResilientChatModel.caching_history()` turns on a third breakpoint, on the last message of the request. When that message is a tool result, langchain-anthropic moves the marker onto the `tool_result` block.

The breakpoint goes at the end of the conversation, and the conversation only grows by appending. So each call's whole request is the start of the next call's request, and the next call reads all of it from the cache. It then pays the write price (1.25×) only for what the last turn added, instead of the full price (1×) for the whole conversation. Cache reads cost 0.1×.

- **Opt-in for the Coder only.** [`agents/coder_agent.py`](../agents/coder_agent.py) turns it on for the Coder's tools-bound ReAct model. It is opt-in because a one-shot call (PM spec, QA suite) has no later call to read the entry, so a breakpoint there would only add the 0.25× write premium.
- **Breakpoint budget.** The request now uses 3 of the API's 4 breakpoints: tools, system and history.
- **Tests.**
  - `test_history_breakpoint_on_last_message` checks the real Anthropic request payload: one history breakpoint, on the last `tool_result`, and the caller's messages are left unchanged.
  - `test_no_history_breakpoint_unless_enabled` checks that one-shot calls get no history breakpoint.
  - `test_coder_react_model_caches_history` checks that only the Coder's ReAct model has history caching turned on.

  All three are in [`tests/test_resilient_model.py`](../tests/test_resilient_model.py).

### A measurement bug fixed on the way

Every Module 4 cost report shows `cache_write_tokens: 0`, even though prompt-cache entries are written on the first call of every Coder session. The cause is in langchain-anthropic: when the API reports cache writes broken down by TTL, it moves the count into `ephemeral_5m_input_tokens` / `ephemeral_1h_input_tokens` and sets `cache_creation` to 0.

`usage_from_message` in [`observability/costs.py`](../observability/costs.py) now falls back to those fields (test: `test_usage_counts_cache_writes_reported_by_ttl`). Before the fix, writes were priced as ordinary input (1.0× instead of 1.25×), so the Module 4 reports understate their cost slightly. The "before" figures below inherit that understatement, so the savings reported here are conservative.

### An alternative I tried and dropped

The usual advice is to keep the system prompt frozen, and the Coder's system prompt includes its memory block. So my first version moved the memory block out of the system prompt to the end of each request, and never stored it in the conversation.

Before running it live, I checked the per-call data. Within a session the cache reads stay at a fixed size, which means the memory block does **not** change between iterations. It is also large: 8–17K tokens of recent conversation. Moving it to the end would have sent it uncached on every call, which costs more than it saves. I reverted that version and kept the change to a single breakpoint.

## 4. Before/after token delta

### Controlled A/B: same conversation, breakpoint off and on

[`scripts/cache_ab.py`](../scripts/cache_ab.py) sends one scripted 10-turn Coder conversation through the real Haiku model wrapper twice. It uses the production system prompt and tools, a fixed memory block, and about 2.2K tokens of tool output per turn. The model's replies are not fed back, so both arms send the same input. Results: [`evidence/v5/cache-ab.json`](evidence/v5/cache-ab.json).

| 10 calls, Haiku 5.5 | Before (tools + system cached) | After (+ history breakpoint) | Delta |
|---|---:|---:|---:|
| Prompt tokens sent | 131,555 | 131,575 | same input |
| Read from cache (0.1×) | 29,740 | 111,305 | +81,565 |
| Written to cache (1.25×) | 0 | 20,248 | +20,248 |
| **Billed at full input price (1×)** | **101,815** | **22** | **−101,793 (−99.98%)** |
| **Input cost** | **$0.010479** | **$0.003646** | **−65.2%** |

In the "before" arm, cache reads stay at 2,974 tokens per call (tools + system) while the prompt grows from 3,089 to 23,224. In the "after" arm, each call reads the previous call's whole prompt and writes about 2,237 new tokens. That is the expected pattern for a working cache.

Input cost falls by 65% rather than 90% because the 1.25× price on each new turn remains. The saving grows with conversation length: the longer a session runs, the more of each call is cache reads.

### Live BST run with the change

Run [`fb57925df870`](evidence/v5/cache-bst/) passed every gated check: `run status: completed`, all tasks approved in QA round 1, and the supplied fixture 5/5 ([demo-check.txt](evidence/v5/cache-bst/demo-check.txt)). It reported 95% coverage, with the same "QA's suites exited 1" note as the graded run (reported, not gated).

| Coder, live BST | Before `e0d27f3759ec` | After `fb57925df870` |
|---|---:|---:|
| Calls | 56 | 28 |
| Prompt tokens | 1,148,906 | 400,876 |
| Read from cache | 51.9% | 74.5% |
| **Billed at full input price** | **48.1%** (552,743) | **2.8%** (11,288) |
| Input cost per 1M prompt tokens | $0.0533 | $0.0386 (−28%) |
| Cost per Coder call | $0.00171 | $0.00105 (−39%) |
| Whole run | $0.1437 | $0.0541 |

**Caveat: the two live runs are not like-for-like.** The PM planned 3 tasks in the new run and 5 in the graded one, so the Coder made half as many calls with shorter conversations: 14.3K tokens per call on average, against 20.5K. The run-total drop from $0.144 to $0.054 is mostly that difference, not caching.

The per-token and per-call rows are the fairer comparison. They understate the effect for the same reason: shorter conversations benefit less from caching. The controlled A/B above is the clean measurement of the change itself.

## 5. What it costs and what it doesn't change

- **Behaviour:** none. The model receives the same messages in the same order. Only the `cache_control` markers in the request are new.
- **Cost:** a write premium of 0.25× on each turn's new tokens, which the next call's read repays. The cache TTL is 5 minutes, and Coder iterations start seconds apart, so entries don't expire mid-session.
- **Providers:** OpenAI and Ollama models are unaffected. The markers are added only for `anthropic:` models served through `ChatAnthropic`.

## 6. Reproduce

```bash
# Controlled A/B, about $0.03 (needs ANTHROPIC_API_KEY, or use 1Password)
op run --env-file=.env.op -- uv run python -m scripts.cache_ab docs/evidence/v5/cache-ab.json

# Live BST run, about $0.05–0.15 depending on the PM's plan
op run --env-file=.env.op -- uv run team --demo bst --reset-memory
```
