"""O nucleo (o que produz ou valida um numero) so' muda junto do manifesto, que e' regenerado ao repetir a verificacao final."""
from __future__ import annotations

import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "scripts"))
import congelar_nucleo as cn  # noqa: E402


def test_manifesto_do_nucleo_bate_com_os_arquivos():
    """Falhou? Arquivo do nucleo mudou. Se a mudanca era para mudar numero/validacao: repita `scripts/verificar_hash_final.py` e depois
    `python scripts/congelar_nucleo.py`. Se era so' apresentacao, o arquivo nao deveria estar no nucleo (ajuste PADROES)."""
    esperado = json.loads(cn.MANIFESTO.read_text(encoding="utf-8"))
    d = cn.comparar(esperado, cn.construir())
    assert not any(d.values()), d


def test_comparar_distingue_mudou_sumiu_e_entrou():
    base = {"arquivos": {"a": "1", "b": "2", "c": "3"}}
    novo = {"arquivos": {"a": "1", "b": "X", "d": "4"}}
    assert cn.comparar(base, novo) == {"mudaram": ["b"], "sumiram": ["c"], "entraram": ["d"]}
    assert cn.comparar(base, base) == {"mudaram": [], "sumiram": [], "entraram": []}


def test_hash_ignora_fim_de_linha(tmp_path):
    a, b = tmp_path / "a.py", tmp_path / "b.py"
    a.write_bytes(b"x = 1\r\ny = 2\r\n")
    b.write_bytes(b"x = 1\ny = 2\n")
    assert cn.sha256_normalizado(a) == cn.sha256_normalizado(b)
    b.write_bytes(b"x = 1\ny = 3\n")
    assert cn.sha256_normalizado(a) != cn.sha256_normalizado(b)


def test_o_nucleo_cobre_o_que_a_fase_de_design_proibe_mexer_sem_repetir():
    arqs = set(cn.arquivos_do_nucleo())
    for n in ("anova_grupos", "anova_cli", "preprocessamento", "validacao_estatistica", "selecao_variaveis", "vip_estabilidade",
              "classificadores", "pipeline", "dados_io", "hsi_io", "conformal", "transferencia_calibracao"):
        assert f"src/guaraci/{n}.py" in arqs, n


def test_apresentacao_nao_esta_no_nucleo():
    arqs = cn.arquivos_do_nucleo()
    for n in ("catalogo.py", "catalogo_dados.py", "guaraci_theme.py", "design_tokens.py", "paleta_cores.py", "app_nav.py",
              "assistente.py", "confianca.py", "exemplos.py"):
        assert f"src/guaraci/{n}" not in arqs, n
