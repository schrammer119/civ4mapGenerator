---
name: commit
description: "Write branch names, commit messages, and squashed PR commit messages for PlanetSim using the README template."
---

# Commit and branch conventions

This file is the single source for branch and commit message format. README and AGENTS.md link here instead of repeating it. Do not invent a different format.

## Branch names

Use `<type>/<short-description>`, where `<type>` is `feat`, `fix`, or `tool`, and `<short-description>` is a short hyphen-separated phrase.

Examples: `feat/river-lake-threshold`, `fix/resource-wrap-seam`, `tool/agent-skills`.

## Which template to use

| Situation | Format |
|-----------|--------|
| Any commit on `master` | Full README template |
| First commit on a branch | Full README template |
| Follow-up commit on a branch | Short form: summary title plus itemized list |
| Squashing a branch to `master` or for a PR | Convert the squashed message to the full README template |

## Full template

```
<type>: <short-title>

**Reasoning:**

<explanation of why this change is necessary or beneficial>

**Additions and Changes:**

- <list of additions and changes made in this commit>

**Tests and Regression Protection:**

<tests or checks added to ensure future changes will not break this change>

**Verification:**

<tests, checks, or steps taken to verify this change>
```

`<type>` is `feat`, `fix`, or `tool`. In **Verification**, list the commands actually run and their results, using the checks in the `verify-source-changes` skill.

## Short form (follow-up commits only)

```
<type>: <short-title>

- <change 1>
- <change 2>
```

## Squash procedure

1. Gather the branch commits with `git log --oneline master..HEAD`.
2. Write one full-template message covering the whole branch: Reasoning from the branch purpose, Additions and Changes from the combined diff, and Verification from the checks run on the final tree.
3. Do not carry over the short-form bullets verbatim as the squashed body.

## Checks before committing

- Title uses `feat`, `fix`, or `tool` and a short imperative summary.
- Commit content is limited to one logical change.
- No generated output, caches, or local Pixi state are staged.
- Run the `verify-source-changes` checks first when source changed.
