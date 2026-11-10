"""M-13 do parecer de 2026-09-25: uma versao so', em todo lugar."""
import re
from pathlib import Path

import guaraci
from guaraci.config import __version__

_R = Path(__file__).resolve().parents[1]


def _ler(p):
    return (_R / p).read_text(encoding="utf-8")


def test_pyproject_config_citation_e_readme_dizem_a_mesma_versao():
    v = __version__
    assert re.search(r'^version = "%s"' % re.escape(v), _ler("pyproject.toml"), re.M)
    assert re.search(r'^version: "%s"' % re.escape(v), _ler("CITATION.cff"), re.M)
    for readme in ("README.md", "README.pt-BR.md"):
        assert f"badge/version-{v}-" in _ler(readme)
    assert guaraci.__version__ == v if hasattr(guaraci, "__version__") else True


def test_changelog_nao_afirma_release_inexistente():
    assert f"## v{__version__}" in _ler("CHANGELOG.md")
