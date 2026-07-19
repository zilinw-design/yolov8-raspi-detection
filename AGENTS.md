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

## Coding Pre-Execution Protocol

**Before writing any code，必须先执行以下步骤，获得用户审批后才能动手：**

1. **方案说明**：解释要实现什么功能、为什么这样做
2. **技术栈与工具**：列出涉及的库、API、参数
3. **预期效果**：描述实现后的行为和效果
4. **替代方案**（如有）：说明为什么选这个方案而不是其他，否决方案需注明原因
5. **文献依据**（如有）：列出参考的论文/GitHub项目/社区讨论，禁止无出处的幻想式方案
6. **等待审批**：用户确认后再开始写代码
7. **追加决策记录**：审批通过并实施后，将完整决策（含候选方案、否决方案、出处）追加到 `ai/decisions/Project Decision Log.md`

违反此协议的代码改动将被拒绝。

## Coding Tasks

Before coding, read `ai/governance/Code Quality Rules.md` (when available).
Default coding requirements:
- keep files focused
- keep functions single-purpose
- avoid unrelated refactors
- verify changes or state what remains unverified

## Recordkeeping

Do not create logs for routine chatter. Record only:
- decisions that affect future work → **必须追加到 `ai/decisions/Project Decision Log.md`**
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
