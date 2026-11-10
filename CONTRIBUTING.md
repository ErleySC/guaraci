# Contributing to GUARACI

**Público:** todos · **Estado:** manual · **Atualizado em:** ver `docs/qualidade/estado_do_projeto.md`

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
pip install -e ".[all,dev]"                # [dev] = pytest, xdist, cov, hypothesis, ruff, mypy, bandit (pinned ranges)
python scripts/camadas_de_teste.py rapido     # every commit: suite without `slow`, no coverage (target <= 10 min)
python scripts/camadas_de_teste.py cobertura  # once per cycle: the whole suite (with `slow`) with branch coverage, global and scientific-core summary
python scripts/camadas_de_teste.py completo   # final hash only: everything, including `slow` (target <= 30 min)
# test processes in the final verification: GUARACI_TESTES_WORKERS (Windows side, default 2, at most 4) and
# GUARACI_TESTES_WORKERS_WSL (WSL side, default 4, can only lower it); each side reads only its own variable
pytest -q                                   # the plain suite, serial
ruff check .                                # lint
mypy src/guaraci/ app_quimiometria.py       # types (same as the CI job)
bandit -r src -ll -q                        # static security scan (same as the CI job)
bash scripts/ci_local.sh                    # all CI gates, in order
```

## Pull requests

- Keep changes focused; unrelated refactors belong in a separate PR.
- Add or update tests for any behavior change (`tests/`).
- Update `MANUAL.md` if you change a user-facing workflow, menu, or
  output format.
- Make sure the `rapido` layer, `ruff check .`, `mypy` and `bandit` (commands above) pass before opening the PR.
- A test that validates statistical precision (family-wise error, interval coverage, Monte Carlo) keeps its size and is marked `slow`;
  its fast twin is a minimal, deterministic test that exercises the same branches. Never use `skip`/`xfail` to save time.
- Coverage exclusions (`# pragma: no cover`, and what coverage excludes on its own: `if TYPE_CHECKING:` and `...` stub bodies; plus
  `omit`, `exclude_lines`) need an entry in `tests/cobertura_excecoes.json` with reason and category; `tests/test_cobertura_excecoes.py`
  fails when their number grows without one. Coverage is measured on `src/guaraci` only (`app_quimiometria.py` and `scripts/` are out of scope).
- Before publishing, run `python scripts/auditar_exposicao.py`: it reports file and line (never the text) of any e-mail, user path, token or
  entry of the maintainer's private term list in the tree. Publish only when it prints 0.
- **`ruff format` is NOT adopted** (decision recorded 2026-10-02). `ruff format --check .` would reformat 327 files (+26,696/-14,555 lines of pure formatting); applying it is an author decision, not done. Do not run `ruff format` on the tree; CI only runs `ruff check`.
- Skipped tests are guarded: `tests/pulados_esperados.json` lists the allowed skip reasons and ceilings per environment (`local`, `ci`, `sem-checkout-git`; override with `GUARACI_PULADOS_AMBIENTE`). A new reason, a skip without reason or a count above the ceiling fails the session.
  Contributors without the maintainer's local data use `GUARACI_PULADOS_AMBIENTE=ci`.
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

## Reading the code comments

Comments cite their origin (`Passo N`, `M-xx`, `CLAUDE.md` section…). What each reference means and where to find it: [`docs/cientifico/glossario_referencias_no_codigo.md`](docs/cientifico/glossario_referencias_no_codigo.md).
