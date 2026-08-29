---
sidebar_position: 8
title: "Cost & Context Governance"
description: "Classify expensive work, reserve QA headroom, checkpoint partial handoffs, and narrow delegated toolsets before a child agent starts"
---

# Cost & Context Governance

Hermes can attach a lightweight governance controller to high-cost or long-running work. The controller does four things:

1. **Classifies the request** into a budget profile before deep execution.
2. **Tracks an envelope** for model calls, tokens, tool invocations, retries, delegation count, and wall-clock time.
3. **Preserves QA headroom** so a final review lane is not starved by earlier work.
4. **Hardens delegation** so child agents inherit the right engagement metadata and start with a narrowed toolset on their first turn.

This is especially useful for security reviews, MiniCISO-style assessments, long implementation lanes, and any workflow where delegated children should not inherit the parent's full budget by default.

## What gets created

When governance is active, Hermes writes a per-engagement workspace under:

```text
~/.hermes/<profile>/engagements/<engagement-id>/
```

Common artifacts:

- `brief.json` — objective, request class, selected agents, envelope, QA reserve
- `budget.json` — shared root budget plus per-child consumption
- `telemetry.jsonl` — turn, threshold, context-manifest, and tool-schema events
- `evidence_ledger.jsonl` — append-only evidence references
- `claim_ledger.jsonl` — append-only claim tracking
- `handoff-*.json` — resumable partial handoffs when a child or lane times out

These files are operational artifacts, not public documentation. Treat them as internal working state.

## Modes

```yaml
cost_context_governance:
  mode: disabled   # disabled | observe | enforce
```

- `disabled` — no controller, no budget enforcement.
- `observe` — tracks state and writes artifacts, but mostly logs instead of blocking.
- `enforce` — can pause or hard-stop model/tool activity when the envelope is exceeded or progress stalls.

## Request classification

Before the turn gets expensive, Hermes maps the request into one of the built-in profiles:

- `conversational` — short, normal chat
- `bounded` — contained review / investigation / analysis work
- `standard_engagement` — the default budget for assessment-style work
- `deep_engagement` — bigger, slower, multi-lane engagements

Out of the box, keyword-based routing is intentionally simple:

- engagement keywords like `miniciso`, `security`, `assessment`, `review`, `threat model`, `appsec`, `bug bounty` → `standard_engagement`
- words like `review`, `assessment`, `investigate`, `analyze` → `bounded`
- delegated children inherit the parent engagement and profile instead of reclassifying themselves from scratch

You can override the defaults with custom keywords, profiles, and per-profile envelopes.

## Default budget profiles

The shipped defaults are designed to be conservative:

```yaml
cost_context_governance:
  profiles:
    conversational:
      total_model_calls: 3
      tool_invocations: 6
      total_tokens: 9000
      delegation_count: 0
      qa_reserve_ratio: 0.0
    bounded:
      total_model_calls: 6
      tool_invocations: 15
      total_tokens: 26000
      delegation_count: 1
      qa_reserve_ratio: 0.10
    standard_engagement:
      total_model_calls: 8
      tool_invocations: 25
      total_tokens: 36000
      delegation_count: 3
      qa_reserve_ratio: 0.20
    deep_engagement:
      total_model_calls: 16
      tool_invocations: 50
      total_tokens: 84000
      delegation_count: 4
      qa_reserve_ratio: 0.25
```

Each profile also carries limits for:

- input tokens
- output tokens
- context tokens per request
- wall-clock seconds
- retries
- no-progress iterations
- model calls per delegated task

## Thresholds and progress controls

Governance evaluates consumption against configurable thresholds:

```yaml
cost_context_governance:
  threshold:
    informational_ratio: 0.50
    warning_ratio: 0.70
    approval_ratio: 0.85
    hard_stop_ratio: 1.00
  progress:
    equivalent_tool_calls: 3
    same_error_repeats: 3
    no_progress_iterations: 4
    timeout_checkpoint_seconds: 30
```

In `enforce` mode, Hermes can:

- pause before a model call if the projected request would consume QA reserve;
- deny further tool activity after repeated no-progress iterations;
- emit resumable handoff artifacts near timeout boundaries.

## QA reserve

A profile's `qa_reserve_ratio` protects part of the budget for a final review lane.

Example: if `standard_engagement.total_tokens=36000` and `qa_reserve_ratio=0.20`, Hermes keeps roughly 7.2k tokens in reserve for QA. Non-QA workers can consume the rest, but once they would cut into the reserve, governance can pause or reject more work in `enforce` mode.

The QA reserve applies to both **tokens** and **model calls**.

## Governance-aware delegation

Governance and delegation are wired together so the child agent starts in the right shape immediately.

### 1. Child budget seeding

When a parent delegates inside an active engagement, the child inherits a governance seed containing:

- `engagement_id`
- `request_class`
- `profile_key`
- `budget_override`
- `selected_agents`

That seed means the child joins the same engagement instead of inventing a new budget universe.

### 2. Role-based toolset caps before child startup

Hermes distinguishes **execution role** from **governance role**:

- execution role: `leaf` or `orchestrator`
- governance role: `chief`, `sme`, or `qa`

The governance role controls a configurable allowlist:

```yaml
cost_context_governance:
  role_toolsets:
    chief: [delegation, todo, session_search, file, search, skills, browser, web]
    sme: [file, search, terminal, web, browser]
    qa: [file, search, terminal, web, browser, session_search]
```

This cap is applied **before** the child `AIAgent` is constructed. That detail matters: the child starts with the reduced toolset on its first turn rather than narrowing only after initialization.

### 3. QA routing by brief text

Delegated tasks that explicitly look like final review work (for example `security qa`, `qa review`, or `final review`) are routed into the narrower QA governance role.

### 4. Partial handoffs on timeout/error

If a delegated lane times out or stalls, governance can persist a resumable handoff artifact with:

- current status
- errors seen so far
- evidence references
- recommended next step
- resume token

That gives the parent or a later run a concrete checkpoint instead of a dead end.

## Configuration example

```yaml
cost_context_governance:
  mode: observe
  workspace_dir: engagements
  default_profile: conversational
  force_for_profiles: [chief-of-staff]
  engagement_keywords:
    - miniciso
    - security
    - assessment
    - review
    - threat model
    - appsec
    - bug bounty
  threshold:
    informational_ratio: 0.50
    warning_ratio: 0.70
    approval_ratio: 0.85
    hard_stop_ratio: 1.00
  progress:
    equivalent_tool_calls: 3
    same_error_repeats: 3
    no_progress_iterations: 4
    timeout_checkpoint_seconds: 30
  profiles:
    bounded:
      total_model_calls: 6
      tool_invocations: 15
      total_tokens: 26000
      delegation_count: 1
      qa_reserve_ratio: 0.10
  role_toolsets:
    chief: [delegation, todo, session_search, file, search, skills, browser, web]
    sme: [file, search, terminal, web, browser]
    qa: [file, search, terminal, web, browser, session_search]
```

## When to turn it on

Good candidates:

- security assessments and code reviews
- long research or implementation lanes with multiple subagents
- cron or automation jobs where runaway context cost matters
- workflows with a required QA/final-review stage

Usually unnecessary:

- casual chat
- one-shot commands
- tiny file lookups or quick Q&A

## Relationship to other features

- **Delegation**: governance narrows the child's first-turn toolset and seeds child budgets.
- **Context compression**: governance tracks context size pressure, but it does not replace the normal compressor.
- **Cron**: useful for expensive recurring jobs when you want smaller toolsets and bounded envelopes.
- **Kanban / orchestration**: complements them by adding budget and handoff discipline.

## Current limitations

- Classification is keyword-based by default; if your workflows need richer routing, define custom keywords and profiles.
- Governance focuses on envelope control and resumability, not on business-policy approval workflows.
- The engagement artifacts are JSON/JSONL files, so external reporting still needs a separate renderer if you want polished reports.
