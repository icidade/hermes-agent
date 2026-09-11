---
sidebar_position: 8
title: "Cost & Context Governance"
description: "Bound model and tool work with observable budgets, checkpoints, and fail-closed enforcement"
---

# Cost & Context Governance

Cost & Context Governance is an optional controller for work whose model,
tool, delegation, or context usage needs an explicit envelope. It observes
the request lifecycle, records compact operational metadata, and—when
`enforce` is selected—can stop provider and tool execution before the next
operation would exceed the envelope or rely on untrusted state.

It is not a provider billing system, a policy engine, or a replacement for
normal context compression. It does not persist prompts, tool arguments or
results, capability values, callback tokens, or raw exception text in its
governance artifacts.

## Modes

Configure the controller under `cost_context_governance.mode`:

| Mode | Behavior |
| --- | --- |
| `disabled` | Do not create an active governance envelope. A small status artifact records that the operator disabled the controller. |
| `observe` | Run the lifecycle and emit artifacts and telemetry, but keep enforcement decisions informational where possible. |
| `enforce` | Apply the envelope and fail closed when accounting, identity, decision, schema, or persistence state cannot be trusted. |

The code default is `disabled`. A deployment may enable the controller in
`config.yaml`; the mode is read when the agent is constructed and is not
changed in the middle of a conversation.

## Request classification and identity

At the beginning of a governed turn, Hermes classifies the request into one
of these request classes:

- `conversational`
- `bounded`
- `engagement`

The request class selects a budget profile. The built-in profile keys are
`conversational`, `bounded`, `standard_engagement`, `deep_engagement`, and
`custom`. Classification is keyword-based by default and can be influenced
by `engagement_keywords`. A delegated child receives its parent's governance
seed rather than independently inventing a new engagement.

The identity hierarchy is intentionally explicit:

- **Profile** identifies the Hermes profile whose state is being used.
- **Engagement** is the validated, shared work identity.
- **Task** identifies the agent turn or delegated lane.
- **Logical call** identifies one conceptual provider operation.
- **Attempt** identifies one preflight/dispatch attempt for that logical call.
- **Request** identifies the concrete reservation and provider request.

The same logical call can be resumed with new attempt and request IDs. This
prevents retries or compaction from being mistaken for duplicate accounting.
An engagement identifier is normalized and validated before its persistence
path is constructed. Invalid sentinels such as `None`, `null`, or an empty
value are not used as directory names.

### Classification settings

`force_for_profiles` is a `list[str]`, defaulting to `["chief-of-staff"]`.
It does not force every request into engagement; for a listed profile, it
only makes a message containing `security` or `miniciso` an `engagement` using
`standard_engagement` when no earlier rule matched.

`engagement_keywords` is a `list[str]`, defaulting to `["miniciso",
"security", "assessment", "review", "threat model", "appsec", "bug
bounty"]`. Hermes lowercases the message and checks whether any configured
keyword occurs as a substring. A match selects `engagement` with
`standard_engagement`.

Classification precedence is: delegated child inheritance;
`engagement_keywords`; `force_for_profiles` for `security` or `miniciso`;
generic markers `review`, `assessment`, `investigate`, or `analyze` (selecting
`bounded`); otherwise `conversational`.

Minimal examples:

```yaml
cost_context_governance:
  force_for_profiles: []
  engagement_keywords: [audit]
```

With this configuration, `audit the config` is an `engagement`, while an
otherwise unmatched `security` message is not promoted by the profile-based
fallback.

## Lifecycle

Each outbound model or tool operation follows the same control boundary:

1. **Preflight** — resolve the current identity, classify the request, load
   the selected envelope, and evaluate projected usage and progress.
2. **Reserve** — atomically reserve the conservative estimate against the
   shared budget. The reservation is the authoritative estimate if usage is
   later missing or malformed.
3. **Dispatch** — mark the request as dispatched only after the dispatch
   transition is durably recorded. A failure to persist this transition
   latches the controller and blocks the provider/tool path in `enforce`.
4. **Reconcile and finalize** — normalize provider usage, reconcile it once
   by stable request ID, release a pre-dispatch reservation when no provider
   call occurred, and close the request in a terminal state.
5. **Accounting latch** — a dispatch, reconciliation, or related persistence
   failure remains visible across controller reconstruction. In `enforce`, new
   outbound work remains blocked until the state is reconciled.

The persisted request states distinguish normal completion from
`released_before_dispatch`, `provider_error_accounted`, `interrupted_accounted`,
`accounting_error`, and `blocked`. Reconciliation is idempotent for the same
request ID; an unknown request ID is an error rather than a new accounting
record.

## Budgets and delegated work

Every budget profile can constrain:

- total model calls and calls per agent;
- calls per delegated task;
- input, output, total, and per-request context tokens;
- wall-clock seconds;
- tool invocations;
- retries;
- delegation count;
- iterations without progress;
- a QA reserve ratio.

The root envelope is shared by the engagement. A child receives a seed with
the engagement ID, request class, profile key, budget override, selected
agents, and a trusted governance role when applicable. Role-based toolset
filtering happens before the child agent is constructed, so the first provider
payload cannot contain tools outside the child's allowed toolsets.

The execution role (`leaf` or `orchestrator`) is distinct from the governance
role (`chief`, `sme`, or `qa`). The QA reserve is calculated independently for
model calls and total tokens. Non-QA work cannot consume the protected reserve
in `enforce`; a trusted QA lane may use its allocation, but it also stops when
that allocation is exhausted.

## Context pressure and compaction

Governance evaluates context pressure before the next provider call. A
continuous pressure episode gets one compaction attempt. If compaction
recovers the envelope, the controller rearms for a later episode; it does not
count every loop iteration as a new episode.

Compaction retries are new outbound attempts. The original preflight is
released before a retry, and the retry receives a new request ID while keeping
the same logical-call identity. A review fork preserves the initial brief
snapshot and the protected fields needed to continue the review, including
scope, decisions, evidence IDs, claim IDs, QA obligations, artifact
references, and open questions.

If compaction result persistence fails, the loop produces a governed stop and
an accounting error rather than returning `stop: false`. If the auxiliary
provider is unavailable, no provider call is fabricated and the reservation
is released before dispatch.

## Blocking behavior

In `enforce`, governance blocks both sides of the operation:

- provider calls are blocked before dispatch when the envelope, identity,
  decision, or accounting state is not valid;
- tools are filtered before their schemas are sent and blocked before their
  bodies execute when the tool decision is not valid;
- an artifact decision outside the allowed decision set fails closed;
- a schema-filtering failure never falls back to the unfiltered schema set;
- a failed dispatch/accounting transition latches the controller and blocks
  subsequent outbound work.

In `observe`, these conditions remain visible through status and telemetry,
but the runtime can continue where the implementation explicitly allows an
informational result. Observe does not turn invalid state into a successful
accounting record.

## Human authorization

When `enforce` pauses an operation for `approval` or reaches `hard_stop`, the
controller creates a request-scoped authorization record. Approval or denial
is handled before another LLM call; an ordinary chat message is not an
authorization channel.

The Telegram integration uses an opaque, short callback value in the inline
button data. The callback is bound server-side to the profile, engagement,
logical call, owner, chat, and request context. It is consumed atomically,
cannot be reused or used after expiry/supersession, and is kept within
Telegram's 64-byte callback-data limit. The server derives the authorization
decision from the stored envelope; clients do not submit a replacement
envelope through the callback.

An unauthenticated, ambiguous, failed, or mismatched callback is a terminal
denial. Telegram authorization errors do not fall through to an environment
allow-all setting, pairing, or another profile's scoped configuration.

## Artifacts and telemetry

Governance stores artifacts below the active profile's canonical Hermes home:

```text
<profile Hermes home>/<workspace_dir>/<engagement_id>/
```

The default `workspace_dir` is `engagements`. Use the configured profile home
and the relative workspace name; do not copy a machine-specific absolute path
into configuration or documentation.

The main files are:

| Artifact | Purpose |
| --- | --- |
| `brief.json` | Objective metadata, request class, selected agents, envelope, QA reserve, and preserved review fields. |
| `budget.json` | Shared budget, reservations, request states, child allocations, logical-call state, authorizations, and accounting latch. |
| `governance_status.json` | Active/disabled status and the profile, engagement, and task identifiers. |
| `checkpoints/<timestamp>-<label>.json` | Metadata snapshots for threshold, compaction, tool, dispatch, or handoff boundaries. |
| `partial_handoffs/<task>-<timestamp>.json` | Resumable status, bounded error metadata, artifact references, open questions, usage, and resume identifiers. |
| `telemetry.jsonl` | Append-only lifecycle events such as `turn_begin`, `request_trace`, `context_manifest`, `tool_schema`, `model_call`, `tool_call`, `threshold_event`, `checkpoint`, `context_compaction`, `partial_handoff`, and `turn_close`. |
| `governance_artifact_errors.jsonl` | Sanitized records for invalid or unreadable governance artifacts. |

Telemetry records minimize sensitive data. Context blocks and exception
details are represented by hashes, character counts, sizes, categories,
fingerprints, and stable internal IDs. Prompts, tool arguments/results,
capabilities, callback tokens, authorization headers, and raw exception text
are intentionally absent from persisted governance payloads.

## Invalid artifacts

An absent artifact can be created during legitimate initialization. An
existing artifact is not silently treated as a fresh default when it fails
validation. The controller records a sanitized artifact error; in `enforce`,
the affected operation fails closed and the provider/tool path is blocked. In
`observe`, the condition remains informational and is exposed through the
status/telemetry path without claiming canonical accounting.

## Safe rollout and operations

Recommended rollout:

1. Start with `mode: observe`.
2. Verify that `brief.json`, `budget.json`, checkpoints, handoffs, and
   telemetry are created under the intended profile home.
3. Check that usage, reservations, child allocations, and QA reserve values
   match the intended workload.
4. Exercise a bounded delegated task and a compaction episode; confirm that
   the child starts with its narrowed toolsets and that a recovered episode
   rearms.
5. Move to `mode: enforce` only after the telemetry and artifact lifecycle are
   understood.

If the accounting latch is active, inspect the sanitized checkpoint,
partial-handoff, `budget.json` request state, and telemetry transition for the
affected request. Reconcile or discard the incomplete operation before
resuming; do not manually mark a provider call as successful.

Rollback is configuration-only: set `cost_context_governance.mode` back to
`observe` for diagnostic continuation or `disabled` to turn the active
controller off, using the normal `hermes config set` command. These are the
supported mode controls; governance does not provide an undocumented schema
migration or a separate rollback command.

## Known limitations

- In `observe`, an identity failure can allow a provider call without
  canonical accounting; the event is marked non-canonical rather than being
  presented as a normal governed result.
- Partial handoff persistence is best-effort when the storage layer is
  unavailable. The runtime still reports the bounded failure, but a resumable
  artifact may be missing.
- Default classification is keyword-based and should not be treated as a
  semantic risk classifier.
- Governance records operational metadata and does not render a polished
  report from the artifacts.

## Configuration reference

See [Hermes Agent Configuration](../configuration.md#cost--context-governance)
for the complete key reference and a minimal configuration example.