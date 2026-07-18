---
status: living
read_policy: before_every_task
owner: user
update_policy: user_only_or_explicit_authorization
---

# Security Boundary

## Purpose

Define content, files, directories, and outputs that AI agents must not access, copy, summarize, expose, record, or preserve.

## Sensitive Content

Agents must not read, copy, summarize, transform, output, log, or store standard sensitive material (API keys, tokens, passwords, secrets, etc.).

## Project-Specific Prohibitions

| Restriction | Reason |
|---|---|
| 不得录制或保存摄像头画面到文件（除非用户明确要求保存测试截图） | 隐私保护 |
| 不得读取或访问用户摄像头以外的视频设备 | 最小权限原则 |
| 不得将检测结果或视频流上传到任何外部服务 | 数据安全 |

## Prohibited File Patterns

Agents must not read content matching standard sensitive patterns (`.env`, `*secret*`, `*token*`, `*.key`, `*.pem`, etc.) by default.

## Prohibited Directories

Standard prohibited directories apply (`backup/`, `user_data/`, `production/`, etc.).

## Safe Substitutes

- `outputs/screenshots/` — 存放用户明确要求保存的测试截图
- 使用公开测试视频（如 YouTube 视频）或示例图片代替摄像头进行开发测试

## Stop Rules

Stop and ask the user when:
- the task requires reading a prohibited file
- the task requires exposing sensitive content
- sensitivity is unclear
- the task requires accessing a camera without explicit user permission

If a secret may have been exposed, advise removal and credential rotation.
