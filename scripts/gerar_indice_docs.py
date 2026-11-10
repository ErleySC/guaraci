#!/usr/bin/env python3
"""gerar_indice_docs.py -- gera `docs/projeto/INDICE.md`: todo documento versionado, por destino, com uma linha de descricao.

    python scripts/gerar_indice_docs.py             # grava o indice
    python scripts/gerar_indice_docs.py --verificar # sai 1 se o indice estiver desatualizado ou faltar descricao

A lista de arquivos vem do `git ls-files` (nunca a mao); a descricao vem de `DESCRICOES`. Documento versionado sem descricao falha.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

RAIZ = Path(__file__).resolve().parents[1]
SAIDA = "docs/projeto/INDICE.md"

#: (titulo da secao, prefixo ou caminho exato) em ordem de leitura.
SECOES: List[Tuple[str, Tuple[str, ...]]] = [
    ("Comece aqui", ("README.md", "README.pt-BR.md", "MANUAL.md", "docs/index.md", "CHANGELOG.md")),
    ("Usuário", ("docs/usuario/",)),
    ("Científico", ("docs/cientifico/",)),
    ("Contribuir", ("CONTRIBUTING.md", "CODE_OF_CONDUCT.md", "SECURITY.md", "ACKNOWLEDGMENTS.md", "SUPPORT.md", "GOVERNANCE.md", "docs/contribuir/", "docs/api/")),
    ("Projeto", ("docs/projeto/", "docs/POLITICA_DE_DOCUMENTACAO.md")),
    ("Qualidade (gerado por script)", ("docs/qualidade/",)),
]

DESCRICOES: Dict[str, str] = {
    "README.md": "Visão geral em inglês: o que é, instalação, primeiros passos e limitações.",
    "README.pt-BR.md": "Versão completa em português do README.",
    "MANUAL.md": "Manual de uso: telas, comandos, técnicas, opções e limitações de cada uma.",
    "docs/index.md": "Página inicial do site de documentação.",
    "CHANGELOG.md": "Histórico de mudanças por versão.",
    "CONTRIBUTING.md": "Como relatar erro, propor mudança e rodar os testes.",
    "CODE_OF_CONDUCT.md": "Código de conduta da comunidade.",
    "SECURITY.md": "Política de segurança: carregamento de modelo, modo servidor e canal para relato de vulnerabilidade.",
    "ACKNOWLEDGMENTS.md": "Atribuição do software e da literatura em que o projeto se apoia.",
    "SUPPORT.md": "Onde tirar dúvida, relatar erro ou falar com o mantenedor, e o prazo de resposta esperado.",
    "GOVERNANCE.md": "Projeto mantido por uma pessoa: quem decide e como alguém vira colaborador.",
    "docs/usuario/expectativas_teste_usabilidade.md": "O que se espera aprender com um teste de usabilidade e como interpretar os resultados.",
    "docs/usuario/guia_do_testador_externo.md": "Guia para quem testa o programa sem conhecer o projeto: o que fazer e o que relatar.",
    "docs/usuario/instalacao.md": "Instalação passo a passo no Windows, Linux e WSL, com requisitos mínimos e problemas comuns.",
    "docs/usuario/privacidade.md": "Rascunho do aviso de privacidade do demonstrativo hospedado: o que recebe, onde fica, por quanto tempo e o que não enviar.",
    "docs/usuario/termos_de_uso.md": "Rascunho dos termos de uso do demonstrativo hospedado: não é laudo, sem garantia, o que não enviar.",
    "docs/usuario/roteiro_teste_usabilidade.md": "Roteiro de tarefas para um teste de usabilidade.",
    "docs/usuario/template_feedback_teste.md": "Modelo de formulário para registrar o resultado de um teste.",
    "docs/cientifico/glossario_referencias_no_codigo.md": "Glossário único dos rótulos que aparecem em comentários e docstrings (Passo, Bloco, achados).",
    "docs/cientifico/limitacoes.md": "O que o projeto não garante, com a evidência de cada limitação.",
    "docs/cientifico/numeros_publicos.yaml": "Contagens próprias do repositório citadas nos READMEs (sem número de conjunto de dados).",
    "docs/cientifico/referencias.md": "Tabela única de referências (fonte, DOI, data de conferência, onde é citada), gerada do registro.",
    "docs/cientifico/referencias_verificadas.yaml": "Tabela única de referências: fonte, DOI, data de conferência e onde é citada.",
    "docs/cientifico/validacao_numerica.md": "Conferência dos métodos contra as fórmulas que os definem, com tolerâncias.",
    "docs/contribuir/arquitetura.md": "Mapa dos módulos do pacote e de como se encaixam.",
    "docs/contribuir/compatibilidade_da_api.md": "Contrato da API pública e política de compatibilidade entre versões.",
    "docs/api/chemometric_stats.md": "Referência da API: estatística quimiométrica.",
    "docs/api/classificadores.md": "Referência da API: classificadores.",
    "docs/api/predicao.md": "Referência da API: predição.",
    "docs/api/preprocessamento.md": "Referência da API: pré-processamento.",
    "docs/api/selecao_variaveis.md": "Referência da API: seleção de variáveis.",
    "docs/api/validacao_estatistica.md": "Referência da API: validação estatística.",
    "docs/projeto/INDICE.md": "Este índice (gerado).",
    "docs/projeto/FASE_DESIGN.md": "Fase de design depois do núcleo congelado: o que entra, o que não entra e como verificar.",
    "docs/projeto/ROADMAP.md": "Roteiro por fase, da v1.0.0 à v1.1+: o que entra, o que não entra, critério de pronto e dependência.",
    "docs/projeto/estudo_novas_tecnicas.md": "Estudo de viabilidade de seis técnicas candidatas (UV-Vis, ICP-OES, MIP-OES, FAAS, GFAAS, DMA), sem código.",
    "docs/projeto/escopo_fusao_multibloco_e_mspc.md": "Escopo da fusão multibloco e do controle estatístico multivariado de processo.",
    "docs/projeto/identidade_visual.md": "Identidade visual e padrões de interface (terminal e web).",
    "docs/projeto/roteiro_v1_1.md": "Roteiro de técnicas e capacidades previstas para a v1.1 e depois.",
    "docs/projeto/MAPA_DE_MIGRACAO.md": "De onde cada documento veio: caminho antigo, caminho novo.",
    "docs/POLITICA_DE_DOCUMENTACAO.md": "O que vive onde, o que nunca entra em arquivo versionado e como isso é verificado.",
    "docs/qualidade/mapa_tecnica_tarefa.md": "Mapa técnica × tarefa × comando × exemplo sintético × teste (gerado do catálogo).",
    "docs/qualidade/dependencias_de_maquina.md": "Testes que usam RAM, CPU, relógio, caminho, ambiente ou rede, com a decisão de cada um (gerado por varredura).",
    "docs/qualidade/cobertura_lacunas.md": "Linhas e ramos que a suíte não cobre, por camada, com a categoria de cada lacuna e as exclusões registradas (gerado da cobertura medida).",
    "docs/qualidade/mutacao_linhas_alteradas.md": "Teste de mutação nas linhas alteradas desde a última verificação completa: mortos, sobreviventes classificados e módulos não executados (gerado).",
    "docs/qualidade/estresse_testes.md": "Testes sensíveis a tempo e memória rodados sob carga de CPU e com memória limitada (gerado).",
    "docs/qualidade/nucleo_congelado.json": "Manifesto do núcleo que produz ou valida um número, com o SHA-256 de cada arquivo.",
    "docs/qualidade/estado_do_projeto.md": "Estado do projeto medido por script: documentos por destino, auditores, links e referências.",
    "docs/qualidade/anova_blindagem.md": "Relatório de blindagem do ANOVA: oráculos independentes, mutação e cobertura.",
    "docs/qualidade/matriz_cenarios.md": "Matriz de cenários sintéticos rodados pela CLI, com resultado e tempo.",
    "docs/qualidade/selecao_aninhada_rotulos_aleatorios.json": "Evidência de que a seleção aninhada fica no acaso com rótulos sorteados.",
    "docs/qualidade/politicas_anova_simulacao.md": "Simulação das políticas de comparação múltipla do ANOVA: erro tipo I e poder.",
    "docs/qualidade/politicas_anova_simulacao.json": "Dados da simulação das políticas do ANOVA.",
    "docs/qualidade/anova/cobertura.json": "Cobertura de ramos dos módulos de ANOVA e VIP.",
    "docs/qualidade/anova/equivalentes_amostra.md": "Amostra dos mutantes classificados como equivalentes, com a verificação.",
    "docs/qualidade/anova/mutantes_classificados.json": "Classificação dos mutantes sobreviventes.",
    "docs/qualidade/anova/recomendacao_simulacao.md": "Simulação da recomendação automática de método do ANOVA.",
    "docs/qualidade/anova/recomendacao_simulacao.json": "Dados da simulação da recomendação do ANOVA.",
}
for _m in ("anova_cli", "anova_grupos", "relatorio_blocos", "vip_cli", "vip_estabilidade", "pulados"):
    DESCRICOES[f"docs/qualidade/anova/mutacao_{_m}.json"] = f"Resultado do teste de mutação de `{_m}.py` (para `pulados`: `tests/_pulados.py`, a guarda de testes pulados)."
for _m in ("vip_cli_py310_linha214", "vip_cli_py311_linha214"):
    DESCRICOES[f"docs/qualidade/anova/mutacao_{_m}.json"] = "Resultado do teste de mutação de `vip_cli.py` numa linha, em outra versão de Python."

for _arq in sorted(Path(__file__).resolve().parents[1].joinpath("docs", "qualidade", "mutacao").glob("*.json")):
    _mod, _, _faixa = _arq.stem.partition("__")
    DESCRICOES[f"docs/qualidade/mutacao/{_arq.name}"] = f"Resultado do teste de mutação de `{_mod}.py`, linhas {_faixa.replace('_', ' ')} (para `mutacao_linhas_alteradas.md`)."

DESCRICOES["docs/qualidade/rodada_local_formatos_reais.json"] = "Registro (sem arquivos) da rodada local dos leitores contra arquivos reais de fabricante: contagens e hash."
DESCRICOES["docs/qualidade/cobertura/mutacao_cobertura_excecoes.json"] = "Teste de mutação da guarda de exclusões de cobertura (`scripts/cobertura_excecoes.py`), primeira rodada."
DESCRICOES["docs/qualidade/cobertura/mutacao_cobertura_excecoes_2.json"] = "Teste de mutação da guarda de exclusões de cobertura, nova rodada só dos sobreviventes."

DOC_EXT = (".md", ".yaml", ".json")


def descricao(doc: str) -> str:
    """Descricao de um documento; `resultado_<hash>.json` (verificacao final) segue um padrao."""
    if doc in DESCRICOES:
        return DESCRICOES[doc]
    return ""


def versionados(raiz: Path = RAIZ) -> List[str]:
    r = subprocess.run(["git", "ls-files"], cwd=raiz, capture_output=True, text=True, encoding="utf-8")
    return r.stdout.split()


def documentos(arquivos: List[str]) -> List[str]:
    """Documentos que o indice cobre: raiz (md) e tudo em docs/."""
    raiz_md = {"README.md", "README.pt-BR.md", "MANUAL.md", "CHANGELOG.md", "CONTRIBUTING.md", "CODE_OF_CONDUCT.md", "SECURITY.md", "ACKNOWLEDGMENTS.md",
              "SUPPORT.md", "GOVERNANCE.md"}
    return sorted(f for f in arquivos if f in raiz_md or (f.startswith("docs/") and f.endswith(DOC_EXT) and not e_resultado_de_verificacao(f)))


def e_resultado_de_verificacao(doc: str) -> bool:
    """`docs/qualidade/resultado_<hash>.json`: um por verificacao final, criado DEPOIS do hash verificado. Fica fora do indice para que
    publica-lo nao exija mudar um arquivo fora de docs/qualidade/ (a regra e' que depois do hash so' mudam arquivos gerados la)."""
    return doc.startswith("docs/qualidade/resultado_") and doc.endswith(".json")


def gerar(arquivos: List[str]) -> Tuple[str, List[str]]:
    docs = documentos(arquivos)
    faltam = [d for d in docs if not descricao(d)]
    usados = set()
    L = ["# Índice dos documentos\n",
         "**Público:** todos · **Estado:** gerado por `scripts/gerar_indice_docs.py` (não editar à mão) · **Atualizado em:** ver `docs/qualidade/estado_do_projeto.md`\n",
         "Todo documento versionado aparece uma vez, no destino a que pertence (menos os `docs/qualidade/resultado_<hash>.json`, um por verificação final). A política (o que vive onde, o que nunca entra) está em "
         "[`../POLITICA_DE_DOCUMENTACAO.md`](../POLITICA_DE_DOCUMENTACAO.md).\n"]
    for titulo, prefixos in SECOES:
        itens = [d for d in docs if d not in usados and any(d == p or (p.endswith("/") and d.startswith(p)) for p in prefixos)]
        if not itens:
            continue
        L.append(f"## {titulo}\n")
        L.append("| documento | o que é |")
        L.append("|---|---|")
        for d in itens:
            usados.add(d)
            alvo = _relativo(d)
            L.append(f"| [`{d}`]({alvo}) | {descricao(d) or '(sem descrição)'} |")
        L.append("")
    sobra = [d for d in docs if d not in usados]
    if sobra:
        L.append("## Outros\n")
        L.append("| documento | o que é |")
        L.append("|---|---|")
        for d in sobra:
            L.append(f"| [`{d}`]({_relativo(d)}) | {descricao(d) or '(sem descrição)'} |")
        L.append("")
    return "\n".join(L), faltam


def _relativo(doc: str) -> str:
    """Link de `docs/projeto/INDICE.md` para `doc`."""
    return "../../" + doc if not doc.startswith("docs/") else "../" + doc[len("docs/"):]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--verificar", action="store_true")
    a = ap.parse_args(argv)
    texto, faltam = gerar(versionados())
    for d in faltam:
        print(f"sem descricao: {d}")
    destino = RAIZ / SAIDA
    atual = destino.read_bytes().decode("utf-8").replace("\r\n", "\n") if destino.exists() else ""
    if a.verificar:
        if atual != texto:
            print(f"desatualizado: {SAIDA}")
        return 1 if (faltam or atual != texto) else 0
    crlf = destino.exists() and b"\r\n" in destino.read_bytes()
    destino.write_bytes((texto.replace("\n", "\r\n") if crlf else texto).encode("utf-8"))
    print(f"gravado: {SAIDA} ({len(documentos(versionados()))} documentos)")
    return 1 if faltam else 0


if __name__ == "__main__":
    sys.exit(main())
