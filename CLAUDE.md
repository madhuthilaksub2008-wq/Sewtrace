# Sewtrace

Delegate every task that writes or changes code in the `sewtrace` package to
the `coder` agent (`.claude/agents/coder.md`). Follow its workflow, spending
rules and hard rules when coding directly.

After code is written and before it is merged, run the `code-reviewer` agent
(`.claude/agents/code-reviewer.md`) on the change. Fix every Blocker and Major
finding before merge.

`SPEC.md` is the source of truth for requirements. `docs/PROJECT.md` is the
project brief. `docs/CONTRIBUTING.md` has the module layout and conventions.
