"""`scripts/auditar_termos_matriz.py`: o produto e' por tecnica e tarefa; termo de matriz so' onde e' permitido."""
import importlib.util
import subprocess
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def aud():
    spec = importlib.util.spec_from_file_location("auditar_termos_matriz", RAIZ / "scripts" / "auditar_termos_matriz.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _repo(tmp_path, arquivos: dict) -> Path:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    for nome, texto in arquivos.items():
        p = tmp_path / nome
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(texto, encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    return tmp_path


@pytest.mark.parametrize("linha", [
    "classificar azeite por RMN", "espectros de milho em grao", "Óleo vegetal prensado", "oleos comestiveis", "mel de abelha",
    "carne suina", "leite em po", "vinho tinto", "solo agricola", "olive oil adulteration", "edible oils", "vegetable oils",
    "kaki e kiwi", "corn kernels", "wheat flour", "cafe torrado", "lavanda", "mandioca", "urina", "esgoto bruto",
    "Soja pura", "Arroz integral", "castanha", "pork", "honey", "wastewater", "cassava", "solos agricolas",
    # repositorio de dados: citar o repositorio e' citar o conjunto
    "dados no zenodo", "deposito no figshare", "mendeley data", "dryad", "kaggle",
])
def test_termos_de_matriz_sao_pegos_com_ou_sem_acento_e_caixa(aud, linha):
    assert aud._rx().search(aud._norm(linha)), linha


@pytest.mark.parametrize("linha", [
    "classificar amostras por PLS-DA", "espectros de 3 classes", "o melhor modelo", "somente leitura",
    "Perfil multielementar por tecnica e tarefa", "ruido branco", "a cornija", "milhoes de pontos", "melancia", "ouro",
    "oilfield? no", "caminho do arquivo",
])
def test_texto_neutro_nao_e_pego(aud, linha):
    assert not aud._rx().search(aud._norm(linha)), linha


def test_so_conta_fora_das_pastas_permitidas(aud, tmp_path):
    raiz = _repo(tmp_path, {
        "src/pacote/a.py": "# classifica azeite\n",
        "validacao/b.py": "# classifica azeite\n",
        "CITATION.cff": "title: olive oil\n",
        "docs/cientifico/referencias_verificadas.yaml": "titulo: edible oils\n",
        "README.md": "texto neutro\nmais um azeite\n",
    })
    fora, dentro = aud.varrer(raiz, dossie=tmp_path / "sem_dossie.md")
    # `validacao/` deixou de ser lugar permitido: a pasta saiu do repositorio (politica de dados de terceiros)
    assert fora == {"src/pacote/a.py": [1], "validacao/b.py": [1], "README.md": [2]}
    assert set(dentro) == {"CITATION.cff", "docs/cientifico/referencias_verificadas.yaml"}


def test_nomes_de_conjunto_vem_do_dossie_privado_com_caixa(aud, tmp_path):
    """Os nomes nunca ficam no repositorio: vem do dossie (fora do git), e casam com caixa (nome proprio)."""
    dossie = tmp_path / "fora" / "d.md"
    dossie.parent.mkdir()
    dossie.write_text("## Intro\n## Conjuntos sem licenca\n## Fulanoset (NIR)\n## Arquivos de terceiros\n", encoding="utf-8")
    assert aud.termos_do_dossie(dossie) == ["Fulanoset"]
    (tmp_path / "r").mkdir()
    raiz = _repo(tmp_path / "r", {"src/a.py": "# rodado no Fulanoset\n", "src/b.py": "# fulanoset em minusculas\n"})
    assert aud.varrer(raiz, dossie=dossie)[0] == {"src/a.py": [1]}
    assert aud.varrer(raiz, dossie=tmp_path / "sem_dossie.md")[0] == {}


def test_doi_de_repositorio_de_dados_nao_isenta_a_linha_vizinha(aud, tmp_path):
    """So' DOI de publicacao isenta o titulo citado ao lado; DOI de deposito de dados nao (citar o deposito e' citar o conjunto)."""
    # (DOIs de exemplo partidos em dois literais: o verificador de citacoes do repositorio nao os toma por citacao)
    raiz = _repo(tmp_path, {
        "docs/a.md": "artigo sobre azeite\ndoi:10." "1016/j.x.2020.1\n",
        "docs/b.md": "conjunto de azeite\nver 10." "17632/abcd.1\n",
        "docs/c.md": "deposito do milho: 10." "5281/zenodo.123\n",
    })
    fora, dentro = aud.varrer(raiz, dossie=tmp_path / "sem_dossie.md")
    assert fora == {"docs/b.md": [1], "docs/c.md": [1]} and set(dentro) == {"docs/a.md"}


def test_acknowledgments_nao_e_mais_lugar_permitido(aud, tmp_path):
    raiz = _repo(tmp_path, {"ACKNOWLEDGMENTS.md": "dados de azeite\n"})
    assert aud.varrer(raiz, dossie=tmp_path / "sem_dossie.md")[0] == {"ACKNOWLEDGMENTS.md": [1]}


def test_historico_e_testes_de_fixture_nao_entram(aud, tmp_path):
    raiz = _repo(tmp_path, {"docs/revisao_pares_x/a.md": "azeite\n", "CHANGELOG.md": "milho\n",
                            "tests/fixtures/y.txt": "carne\n"})
    assert aud.varrer(raiz) == ({}, {})


def test_main_sai_1_com_ocorrencia_e_0_sem(aud, tmp_path, monkeypatch, capsys):
    raiz = _repo(tmp_path, {"src/a.py": "x = 'azeite'\n"})
    monkeypatch.setattr(aud, "RAIZ", raiz)
    assert aud.main([]) == 1
    assert "1 linha(s)" in capsys.readouterr().out
    (raiz / "src" / "a.py").write_text("x = 'classe'\n", encoding="utf-8")
    assert aud.main([]) == 0


def test_a_saida_nunca_mostra_o_trecho(aud, tmp_path, monkeypatch, capsys):
    raiz = _repo(tmp_path, {"src/a.py": "segredo = 'azeite do fulano'\n"})
    monkeypatch.setattr(aud, "RAIZ", raiz)
    aud.main(["--por-arquivo"])
    saida = capsys.readouterr().out
    assert "src/a.py" in saida and "fulano" not in saida and "azeite" not in saida


def test_o_produto_esta_limpo(aud):
    """A meta da politica: nenhuma ocorrencia fora de validacao/ e das referencias. Atualizada para 0 ao fim da generalizacao."""
    fora, _ = aud.varrer()
    assert sum(len(v) for v in fora.values()) == 0, {k: len(v) for k, v in fora.items()}


@pytest.mark.parametrize("linha", ["queijo minas", "cacau em po", "peixe seco", "cerveja artesanal", "tomato paste", "biodiesel B20", "propolis verde", "caffeine content"])
def test_outras_matrizes_comuns_tambem_sao_pegas(aud, linha):
    assert aud._rx().search(aud._norm(linha)), linha


def test_termo_colado_por_sublinhado_conta_em_prosa_mas_nao_em_identificador_de_codigo(aud, tmp_path):
    """`\b` nao separa em `_`: `oleos_essenciais` escapava do auditor no manual (o termo composto so' conta em prosa e metadado)."""
    composto = "o nivel oleos" "_essenciais\n"          # partido em dois literais: este arquivo e' permitido, mas o exemplo fica explicito
    raiz = _repo(tmp_path, {"MANUAL.md": composto, "config.example.yaml": "chave: " + composto, "src/a.py": "x = 1  # " + composto})
    fora, _ = aud.varrer(raiz, dossie=tmp_path / "sem_dossie.md")
    assert fora == {"MANUAL.md": [1], "config.example.yaml": [1]}
    assert not aud._rx().search(aud._norm(composto)) and aud._rx(composto=True).search(aud._norm(composto))


def test_main_sem_git_falha_alto_em_vez_de_dizer_zero(aud, tmp_path, monkeypatch, capsys):
    """Arvore sem `.git` (zip, sdist): a varredura nao olhou nada, entao NAO pode sair 'zero ocorrencias' com codigo 0."""
    (tmp_path / "README.md").write_text("texto neutro\n", encoding="utf-8")
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    monkeypatch.setattr(aud, "RAIZ", tmp_path)
    assert aud.main([]) == 2
    assert "nada foi verificado" in capsys.readouterr().err
