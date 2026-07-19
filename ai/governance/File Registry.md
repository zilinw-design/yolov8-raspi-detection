---
status: living
read_policy: before_every_task
owner: user
update_policy: user_or_authorized_ai
---

# File Registry

## Purpose

Define canonical file status, read policy, ownership, and update policy. Agents must use this file as the routing table.

## Status Values

| Status | Meaning |
|---|---|
| `living` | Active source of truth. Read according to its policy before related work. |
| `scoped-living` | Active only for related task types. Read when triggered. |
| `append-only` | Historical record. Append only. |
| `stable` | Stable reference. Read when related. |
| `generated` | AI-generated output not yet accepted as project fact. |
| `external` | External reference. Must be verified before becoming project fact. |
| `archive` | Historical material. Do not use as current truth. |

## Registry

| File or Directory | Status | Read Policy | Owner | Purpose |
|---|---|---|---|---|
| `AGENTS.md` | living | before_every_task | user | Agent entrypoint |
| `README.md` | stable | when_related | user | Human usage guide |
| `ai/governance/File Registry.md` | living | before_every_task | user | File status and routing |
| `ai/governance/Project Brief.md` | living | before_every_task | user | Project goals, non-goals, uncertainties |
| `ai/governance/Security Boundary.md` | living | before_every_task | user | Security red lines |
| `ai/governance/Constraints And Priority.md` | living | before_each_operation | user_or_lead_ai | Current scope and priority |
| `ai/governance/Project Invariants.md` | scoped-living | when_related | user | Non-security technical invariants |
| `ai/state/Operation State.yaml` | living | latest_only | current_ai | Current handoff state |
| `ai/logs/` | append-only | when_related | working_ai | Agent logs when needed |
| `ai/lessons/` | append-only | when_related | working_ai | Repeat-error prevention |
| `ai/decisions/Project Decision Log.md` | append-only | before_each_operation | user_or_lead_ai | 技术决策记录：方案、出处、否决原因 |
| `src/` | living | when_related | user_or_authorized_ai | Production source code |
| `tests/` | stable | when_related | user_or_authorized_ai | Test code |
| `docs/` | stable | when_related | user_or_authorized_ai | Formal documentation |
| `references/` | external | when_related | user_or_research_ai | External references |
| `outputs/` | generated | when_related | ai_or_user | Unaccepted AI outputs |
| `archive/` | archive | when_related | user | Archived material |
| `../ai_harness_framework/` | stable | when_related | user | 框架参考（非本项目文件） |
| `../ai_harness_framework/references/raw/opencv/` | external | when_related | user | OpenCV 4.14.0-pre 参考源码 |
| `../ai_harness_framework/references/raw/ultralytics/` | external | when_related | user | Ultralytics 8.4.100 参考源码 |

## Default Read Set

Before project work, read:

1. `AGENTS.md`
2. `ai/governance/File Registry.md`
3. `ai/governance/Project Brief.md`
4. `ai/governance/Security Boundary.md`
5. `ai/governance/Constraints And Priority.md`
6. `ai/state/Operation State.yaml`

## Conditional Read Rules

| Trigger | Also Read |
|---|---|
| New task or vague request | `Project Brief.md` (re-read) |
| Coding or file modification | `Constraints And Priority.md` (re-read scope) |
| Final answer or completion | `Operation State.yaml` (update) |
| Technology stack, API, or dependency changes | `Project Invariants.md` |
| 提出优化方案或技术选型 | `ai/decisions/Project Decision Log.md`（追加记录） |
| Command failure or repeated error | `ai/lessons/` |

## Conflict Rules

1. If this registry conflicts with `Security Boundary.md`, security wins.
2. If this registry conflicts with `Project Invariants.md` on technical constraints, invariants win.
3. If file metadata conflicts with this registry, this registry wins.
