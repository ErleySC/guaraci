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

## License

By contributing, you agree that your contribution is licensed under the
project's [GPL-3.0-or-later](LICENSE), consistent with the rest of the
codebase.
