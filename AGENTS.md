# AGENTS.md

Project-level entrypoint for AI agents.

## Required Reading

Before project work, read in this order:

1. `ai/governance/File Registry.md`
2. Files required by the registry for the current task
3. `ai/state/Operation State.yaml` when continuing, handing off, or recovering after context compression

## Harness Lifecycle

```text
Task Intake → Session Start → Work Loop → Verification Gate → Session End
```

## Operating Rules

- Use `File Registry.md` as the source of truth for file status and read policy.
- Keep the task scope minimal unless the user explicitly expands it.
- Do not treat generated drafts, archived content, or external references as project facts until confirmed.
- Put unaccepted AI-generated work in `outputs/`.
- Stop when the task conflicts with security boundaries, technical invariants, or the current operation scope.

## Scope Handling

When the user gives an unclear goal, produce a scope inference before making changes:
- possible affected areas
- recommended open scope
- default locked scope
- reason for the split

Default policy: open only the minimum scope required for the task.

## Security

Never read, copy, summarize, output, log, commit, or preserve sensitive material.
Full policy: `ai/governance/Security Boundary.md`.

## Coding Tasks

Before coding, read `ai/governance/Code Quality Rules.md` (when available).
Default coding requirements:
- keep files focused
- keep functions single-purpose
- avoid unrelated refactors
- verify changes or state what remains unverified

## Recordkeeping

Do not create logs for routine chatter. Record only:
- decisions that affect future work
- verified error causes and fixes
- handoff state needed by the next agent

## Conflict Order

1. `Security Boundary.md`
2. `Project Invariants.md`
3. `Project Brief.md`
4. `Constraints And Priority.md`
5. `File Registry.md`
6. logs, references, drafts, and archives

If the conflict remains unclear, stop and ask the user.
