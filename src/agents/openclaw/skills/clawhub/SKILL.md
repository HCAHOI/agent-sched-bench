---
name: clawhub
description: Search and install agent skills from ClawHub, the public skill registry.
homepage: https://clawhub.ai
metadata: {"nanobot":{"emoji":"🦞"}}
---

# ClawHub

Public skill registry for AI agents. Search by natural language (vector search).

## When to use

Use this skill when the user asks any of:
- "find a skill for …"
- "search for skills"
- "install a skill"
- "what skills are available?"
- "update my skills"

## Search

```bash
npx --yes clawhub@latest search "web scraping" --limit 5
```

## Install

```bash
npx --yes clawhub@latest install <slug> --workdir ~/.local/state/openclaw/skills
```

Replace `<slug>` with the skill name from search results. **Always include `--workdir`** pointing to a directory *outside* the current task workspace (e.g. `~/.local/state/openclaw/skills`). Never omit `--workdir` — without it skills install into the current directory, which contaminates a task repository during evaluation.

## Update

```bash
npx --yes clawhub@latest update --all --workdir ~/.local/state/openclaw/skills
```

## List installed

```bash
npx --yes clawhub@latest list --workdir ~/.local/state/openclaw/skills
```

## Notes

- Requires Node.js (`npx` comes with it).
- No API key needed for search and install.
- Login (`npx --yes clawhub@latest login`) is only required for publishing.
- `--workdir` is mandatory and must point outside the task workspace; omitting it
  installs skills into the current directory and breaks evaluation diffs.
- Evaluation sessions load only runtime/builtin skills, so ClawHub-installed
  skills are intended for standalone interactive use, not benchmark runs.
- After install, remind the user to start a new session to load the skill.
