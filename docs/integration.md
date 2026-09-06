# Agent integration contract

kvpark owns inference state. Your agent owns the transcript and session
lifecycle. No Hermes imports or model-provider dependency exists in the core.

## Request identity

For every foreground chat-completion request, send `slot_archive_role` as
`foreground` and `slot_archive_key` as a stable, nonempty conversation identifier.
Namespace it by application/profile to avoid collisions between clients.

Use a new key for a new chat or branch. Keep the key across restarts. Never use a
user/account identifier, a prompt-content guess, or a detached worker's inherited
parent key as proof of foreground ownership. Background requests must send
`slot_archive_role: "background"`; missing roles are unverified and cannot park.

Optionally set `slot_archive_thread` to a UI thread identifier. New integrations
should park by exact conversation key rather than UI thread to avoid ambiguity
after branching or starting a new conversation in the same thread.

The `slot_archive_*` fields are private to the proxy and are stripped before
forwarding. `prompt_cache_key` and `session_id` remain legacy key fallbacks only
when the foreground role is explicit.

## Transcript and rendering

Replay the full compatible conversation, including assistant and tool messages.
Keep system prompts, tool definitions/order, template settings, and reasoning
effort stable. Changing them can invalidate the saved prefix. Compaction and edits
may force substantial prefill; checkpoints allow compatible rollback, not reuse
of arbitrary rewritten history. Persist the transcript before offering a park
action to the user.

All requests use the same proxy. Do not send unrelated generations or slot control
operations directly to the backend. The proxy serializes complete responses,
including draining disconnected clients, before another request may claim the slot.

## Controls

`GET /_kvpark/status` returns resident identity, archives, global activity, and recent
events. `GET /_kvpark/doctor` checks runtime identity and backend slot topology when
idle. `POST /_kvpark/save` and `POST /_kvpark/forget` take:

```json
{"key": "k-<SHA-256 of the UTF-8 slot_archive_key>"}
```

The CLI hashes `--session` for you: `kvpark save --session …` saves and
`kvpark forget --session …` removes that archive. Parking is a native command/action; it should
not require an LLM turn that could displace the state you wanted to save.
Resume is automatic on the next matching inference request. There is no manual
restore action that reconstructs a transcript.

An internal `POST /_kvpark/simulate-gap` control also saves and erases a resident
slot for diagnostics. It does not prove a process-restart or physical SSD test.

Control failures return HTTP 409 with a reason. Archive failures during inference
are reported in status; the proxy attempts to continue ordinary inference. Clients
must inspect status before claiming a successful save or resume.

## Reading metrics

`restored` means files were loaded. `completed` reports actual cache counters if
the backend supplied them. Associate events with your exact key and compare their
timestamps: a completion before the latest restore does not validate that restore.
Missing counters mean unknown, not zero. Recent events are bounded diagnostics,
not a durable audit log.

Streamed `ttft_seconds` measures the first content, reasoning, or tool-call delta,
including queueing and restore. Nonstream responses report reuse counters but do
not claim TTFT. Model token generation speed and physical disk read speed are
separate measurements.

## Authentication

The alpha binds only to `127.0.0.1` and rejects browser Origin requests. For an
additional local access boundary, set `KVPARK_API_KEY` on the proxy and CLI/client;
all endpoints then require that bearer token. If the backend requires a key, set
`KVPARK_UPSTREAM_KEY_FILE` to a private local file in the proxy environment and pass
the corresponding upstream authentication option when launching the backend.
Never publish keys, transcript files, or archive directories.

## Cleanup settings API

`GET /_kvpark/settings` reads `ttl_days`, `max_gib`, and `cleanup_enabled`.
`POST /_kvpark/settings` atomically persists any subset of those settings.
`POST /_kvpark/cleanup` runs a manual sweep with the saved age/budget rules.
Unknown keys, invalid booleans, negative ages, and nonpositive budgets are refused.
Status includes the running package `version`, `maintenance`, `service: "kvpark"` and `control_version: 4` for adapters.

## Hermes

The native adapter now ships as a `hermes_agent.plugins` entry point. Follow the
[Hermes guide](hermes.md) for launch-on-load, slash-command setup, parking,
deletion, and persistent cleanup controls. It does not edit Hermes internals.
Legacy `/qwen-slot` plugins use `/__slot_proxy/` and `SLOT_PROXY_*`; this package
uses `/_kvpark/` and `KVPARK_*`. The former sloth-memory package used `/_sloth/`;
its migration is documented in [updates](updates.md#migrate-from-sloth-memory).
The request identity fields and Hermes profile namespace remain unchanged by the rename.
