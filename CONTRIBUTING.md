# Contributing to GUARACI

Thanks for your interest in improving GUARACI. Bug reports, feature requests
and pull requests are all welcome. Contributions in Portuguese are fine —
*pode escrever em português.*

Participation in this project is governed by the
[Code of Conduct](CODE_OF_CONDUCT.md).

## Reporting a bug

Open an [issue](https://github.com/ErleySC/guaraci/issues) — the bug report
template asks for what is needed. In short:

- What you ran (`guaraci`, `app_quimiometria.py`, or `python -m guaraci.pipeline`).
- The output of `guaraci doctor` (Python version, OS, optional dependencies).
- Your `config.yaml` (redact any private paths/data).
- The full error/traceback, or a screenshot for UI issues.

**If the problem is a wrong number rather than a crash, say so explicitly.** A
result that is silently wrong is the most serious class of bug here and is
triaged first.

## Proposing a feature

Open an issue describing the use case first — for chemometric methods,
please include a reference (paper/book) describing the technique. This
avoids duplicated or divergent implementation effort.

## Development setup

```bash
pip install -e .[all]
pytest -q            # run the test suite (550+ tests)
ruff check .          # lint
```

## Pull requests

- Keep changes focused; unrelated refactors belong in a separate PR.
- Add or update tests for any behavior change (`tests/`).
- Update `docs/MANUAL.md` if you change a user-facing workflow, menu, or
  output format.
- Make sure `pytest -q` and `ruff check .` pass before opening the PR.
- The project maintains parity between the CLI and the web app for every
  configurable field (`tests/test_interfaces_configuraveis.py`) — if you add
  a config option, wire it into both interfaces.
- If you add a classification or quantification method — a new entry in
  `guaraci.model_registry`, or a new model in `benchmark_regressao_por_especie`
  — it must pass `tests/test_contrato_validacao_agrupada.py` before merge.
  That test is what keeps "leakage-safe validation by default" (the
  project's subtitle) true: it runs your method through the real benchmark
  path and fails if any physical-replicate group ends up split across
  train/test. There is no flag to opt a method out of this.

## Property-based tests (Hypothesis)

`tests/test_propriedades_hypothesis.py` uses [Hypothesis](https://hypothesis.readthedocs.io/)
(a dev-only dependency, extra `[dev]` — never in `requirements.txt`) to
generate cases for the invariants that have proven fragile in this
project: grouped-validation leakage, config.yaml roundtrip, blind-mode
label handling.

**Random search alone is not enough — pin `@example`.** Measured directly
(2026-08-27): a config.yaml roundtrip property test ran 80 random examples
against code that had a real bug (`_fmt_yaml` writing YAML-ambiguous
strings unquoted) and never found it. The bug was only caught once the
specific adversarial values (`'010'`, `'1.50'`, `'0x1A'`, `?` inside a
list item) were pinned with `@hypothesis.example(...)`. **Convention: any
time you discover an adversarial case — via Hypothesis's own shrinker, a
bug report, or manual testing — pin it as an explicit `@example` on the
relevant property test, don't rely on the random search to keep finding
it.** Where no specific historical bug is known for a property yet, add
defensive `@example`s at the boundary conditions of the invariant (e.g.
the exact threshold where a code branch changes) and say so in a comment
— don't leave the impression of pinned coverage that isn't there.

**CI runs more examples than local.** `conftest.py` registers two
Hypothesis profiles — `dev` (50 examples, used locally) and `ci` (300
examples, auto-selected when the `CI` env var GitHub Actions sets is
present). Override with `HYPOTHESIS_PROFILE=ci` locally if you want to
run the wider search before pushing. Don't hardcode `max_examples=` on
individual tests — let the active profile control it, so the CI/local gap
stays uniform across the whole file.

`.hypothesis/` (Hypothesis's own example-replay database) is gitignored —
it's a local cache, not a source of truth. Cases worth keeping are pinned
as `@example` in the code, reviewed like any other test.

## License

By contributing, you agree that your contribution is licensed under the
project's [GPL-3.0-or-later](LICENSE), consistent with the rest of the
codebase.
