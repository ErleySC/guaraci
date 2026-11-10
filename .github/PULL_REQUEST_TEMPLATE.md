<!--
Pull requests in Portuguese are welcome — pode escrever em português.
-->

## What this changes

<!-- One paragraph. If the diff does more than one logical thing, split the PR. -->

## Why

<!-- The problem, not the solution. Link the issue if there is one. -->

## Does this change any number the software produces?

- [ ] **No** — refactor, docs, tooling, or interface only. Numerical outputs are
      byte-identical.
- [ ] **Yes** — and I explain below which numbers change, by how much, and why
      the new value is the correct one.

<!--
This is the question that matters most in this project. A change that silently
alters a computed result is the failure mode with the worst consequences: it
does not crash, it does not warn, it just produces a different number that ends
up in someone's thesis. If you checked "Yes", say what you compared against —
a reference implementation, a published value, or a property that must hold.
-->

## Checklist

- [ ] `pytest -q` — the number of passing tests did not go down
- [ ] `ruff check .` — clean
- [ ] Coverage did not drop (`pytest --cov=src/guaraci`)
- [ ] No new `print()` (use `log`)
- [ ] No new broad `except Exception` without `# noqa: BLE001` and a comment
      justifying it on the same line
- [ ] Any number written into documentation was actually produced by running
      the code, not estimated
- [ ] Any bibliographic reference added is real and verifiable
