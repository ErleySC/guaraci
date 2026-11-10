#!/usr/bin/env python3
"""varrer_dependencias_de_maquina.py -- teste que depende de RAM, CPU, relogio, caminho, ambiente ou rede sem controle.

    python scripts/varrer_dependencias_de_maquina.py              # grava docs/qualidade/dependencias_de_maquina.md
    python scripts/varrer_dependencias_de_maquina.py --verificar  # sai 1 se houver uso NAO decidido, decisao obsoleta ou tabela velha

Varre `tests/` (a validacao do mantenedor fica fora do repositorio e nao entra) atras de usos de memoria livre, contagem de CPU, relogio, caminho de maquina, variavel de ambiente
e rede. Cada (arquivo, categoria) achado precisa estar em `DECISOES` com o estado (`injetado`, `bloqueado`, `isolado`, `corrigido`,
`legitimo`) e o motivo: uso novo sem decisao reprova (guarda estatica), e decisao sem uso correspondente tambem (a tabela nao envelhece).
A tabela `teste | dependencia | corrigido?` sai daqui.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

RAIZ = Path(__file__).resolve().parents[1]
SAIDA = RAIZ / "docs" / "qualidade" / "dependencias_de_maquina.md"
PASTAS = ("tests",)

#: categoria -> regex de uma linha de codigo de teste
CATEGORIAS: Dict[str, str] = {
    "ram": r"\bpsutil\b|virtual_memory|ram_livre_gb|_verificar_ram",
    "cpu": r"cpu_count|sched_getaffinity|n_jobs\s*=\s*-1",
    "relogio": r"time\.time\(|perf_counter|datetime\.now|date\.today|time\.sleep|monotonic\(",
    "caminho": r"[\"' ][A-Z]:(?:\\\\|/)[A-Za-z]|expanduser|Path\.home|gettempdir|[\"']/(?:home|tmp|Users)/",
    "ambiente": r"os\.environ|getenv\(",
    "rede": r"\bsocket\b|urllib\.request|\brequests\.|http\.client|urlopen",
}

#: (arquivo de teste, categoria) -> (estado, motivo). `corrigido` = tinha dependencia real e foi trocada nesta rodada.
DECISOES: Dict[Tuple[str, str], Tuple[str, str]] = {
    ("tests/test_simulador_espectral.py", "ambiente"): ("legitimo", "le e apaga (monkeypatch) GUARACI_SINTETICO_CENARIO para provar que o padrao do usuario e' 'nir' e que a suite fixa 'classico'; a variavel e' o proprio objeto do teste"),
    ("tests/test_hardware.py", "ram"): ("injetado", "psutil e' bloqueado ou substituido por uma memoria falsa; as fontes sem psutil (/proc/meminfo por caminho injetavel, sysconf, ctypes) sao trocadas por arquivo/funcao falsos; hardware.py le o ambiente por natureza e a chamada real so' confere as chaves"),
    ("tests/test_hardware.py", "cpu"): ("injetado", "o fallback sem psutil e' exercitado bloqueando o import; nenhum valor de CPU e' afirmado"),
    ("tests/test_pipeline_core.py", "ram"): ("injetado", "o ajuste por hardware recebe um dicionario com a RAM livre; a sonda real so' confere as chaves"),
    ("tests/test_avaliacao_modelos.py", "ram"): ("corrigido", "o benchmark de regressao dependia da RAM livre do instante (< 0,6 GB pulava e o teste falhava); "
                                                 "agora a verificacao e' injetada nos dois ramos"),
    ("tests/test_tarefas_cli.py", "relogio"): ("corrigido", "o teto de 20 s por tarefa foi removido: o tempo e' medido na documentacao, nao afirmado"),
    ("tests/test_web_seguranca.py", "relogio"): ("legitimo", "fabrica um arquivo com data antiga relativa a agora; nao afirma duracao"),
    ("tests/test_doctor_instalar.py", "ram"): ("legitimo", "so' o NOME do modulo psutil como dado do plano de instalacao (find_spec trocado por funcao fixa); nenhuma memoria e' lida"),
    ("tests/test_fachada_reexport.py", "ram"): ("legitimo", "so' o NOME da funcao na lista de reexports; nenhuma memoria e' lida"),
    ("tests/test_guaraci_cli.py", "caminho"): ("legitimo", "texto de entrada de um teste de truncamento de saida (caminho ficticio)"),
    ("tests/test_web_fluxo_completo.py", "caminho"): ("legitimo", "valor de campo de formulario; o arquivo nunca e' aberto"),
    ("tests/test_verificar_hash_final.py", "caminho"): ("legitimo", "conversao Windows->WSL de um caminho ficticio (usuario `alguem`)"),
    ("tests/test_assistente_duvidas.py", "rede"): ("bloqueado", "o socket e' substituido por uma funcao que falha: prova que o assistente nao usa rede"),
    ("tests/test_hsi_offline_prova.py", "rede"): ("bloqueado", "o socket e' bloqueado durante o pipeline HSI: prova de operacao offline"),
    ("tests/test_app_logic.py", "caminho"): ("legitimo", "confere que o upload usa a pasta temporaria do sistema, qualquer que seja ela"),
    ("tests/test_config_io.py", "caminho"): ("legitimo", "string de entrada de formatacao YAML, nao um caminho acessado"),
    ("tests/test_gerar_blindagem_anova.py", "caminho"): ("legitimo", "texto de entrada do limpador de caminhos (usuario ficticio)"),
    ("tests/test_sanitizar_saidas.py", "caminho"): ("legitimo", "textos de entrada do sanitizador (usuario ficticio)"),
    ("tests/test_checar_design.py", "caminho"): ("legitimo", "texto de entrada do normalizador de instantaneo (usuario ficticio); nenhum caminho e' acessado"),
    ("tests/test_web_seguranca.py", "caminho"): ("legitimo", "afirma que a mensagem de erro NAO traz caminho de unidade"),
}
#: Ambiente: arquivos que herdam `os.environ` e o sobrescrevem explicitamente (HOME/USERPROFILE isolados, MPLBACKEND, PYTHONPATH...) ou
#: leem uma chave opt-in (GUARACI_REGRAVAR_GOLDEN).
for _arq in ("test_amostragem_ativa_cli", "test_contrato_api_publica", "test_contrato_saida_tabular", "test_correcoes_locais_opt_in",
             "test_golden_valores", "test_graficos_motor", "test_monitorar_cli", "test_notebook_5_minutos",
             "test_parafac_cow_relatorio_cli", "test_perfis_objetivo_rigor", "test_pipeline_import_sem_efeito_global",
             "test_revisao_cega_r5", "test_saida_sem_caminho_da_maquina", "test_tarefas_cli",
             "test_tempo_ate_o_menu", "test_transferir_linearidade_robustez", "test_web_fluxo_completo"):
    DECISOES[(f"tests/{_arq}.py", "ambiente")] = ("isolado", "herda o ambiente e sobrescreve o que importa (HOME, PYTHONPATH, MPLBACKEND...) ou le uma chave opt-in")
for _arq in ("test_contrato_api_publica", "test_contrato_saida_tabular", "test_golden_valores"):
    DECISOES[(f"tests/{_arq}.py", "ambiente")] = ("legitimo", "chave opt-in GUARACI_REGRAVAR_GOLDEN: regravar o golden e' decisao explicita")


DECISOES.update({
    ("tests/test_lacunas_avaliacao_modelos.py", "ram"): ("injetado", "_verificar_ram e' trocado por uma funcao fixa nos dois ramos (com e sem memoria)"),
    ("tests/test_idioma_relatorios.py", "ambiente"): ("injetado", "GUARACI_IDIOMA e' fixada em pt e depois en para gerar os relatorios nos dois idiomas; o valor anterior e' restaurado"),
    ("tests/test_lacunas_predicao.py", "ambiente"): ("isolado", "GUARACI_MODEL_KEY_FILE aponta para uma pasta temporaria: a chave real do usuario nunca e' tocada"),
    ("tests/test_medir_vault.py", "ambiente"): ("injetado", "GUARACI_VAULT, GUARACI_VAULT_GERADOR e GUARACI_LIVRO_RAZAO apontam para um vault sintetico temporario"),
    ("tests/test_registrar_rodada_local.py", "caminho"): ("legitimo", "o 'caminho' e' texto de exemplo dentro do XML sintetico, para provar que o registro nao o guarda"),
    ("tests/test_verificar_hash_final.py", "cpu"): ("injetado", "os.cpu_count e' substituido por valores fixos para provar a regra 'metade dos nucleos, no maximo 4'"),
})
_RAM_INJETADA = "a memoria e' sempre uma memoria falsa (psutil ou hardware_probe substituidos); nenhum valor da maquina e' afirmado"
for _arq in ("test_lacunas_fechamento", "test_lacunas_guaraci_ajuda_checklist", "test_lacunas_guaraci_comandos", "test_lacunas_guaraci_predicao_hardware",
             "test_lacunas_guaraci_resto", "test_lacunas_modulos_pequenos_final", "test_lacunas_pipeline_executar",
             "test_lacunas_web_abas_projeto_inicio", "test_lacunas_web_modelo"):
    DECISOES[(f"tests/{_arq}.py", "ram")] = ("injetado", _RAM_INJETADA)
DECISOES.update({
    ("tests/test_lacunas_modulos_pequenos_final.py", "cpu"): ("injetado", "os.cpu_count e' substituido por uma funcao que falha: prova o valor padrao quando a contagem nao existe"),
    ("tests/test_lacunas_web_modelo.py", "relogio"): ("injetado", "o tempo maximo da execucao e' reduzido a zero por constante de modulo; nenhuma duracao real e' afirmada"),
})
# Lote 12 da vistoria 2026-10 (ambiente de teste robusto)
DECISOES.update({
    ("tests/test_basetemp_curto.py", "caminho"): ("legitimo", "caminhos ficticios de entrada e gettempdir substituido; prova a troca do basetemp longo no Windows"),
    ("tests/test_basetemp_curto.py", "relogio"): ("legitimo", "pausa curta so' para ordenar pastas por data e data antiga fabricada; nenhuma duracao e' afirmada"),
    ("tests/test_basetemp_curto.py", "ambiente"): ("legitimo", "le PYTEST_XDIST_WORKER para saber se o basetemp ja' vem com popen-gwN; nada e' alterado"),
    ("tests/test_pulados_esperados.py", "caminho"): ("legitimo", "texto de exemplo de um motivo de pulo (caminho do wsl.exe); o arquivo nunca e' aberto"),
    ("tests/test_suite_sem_tk.py", "ambiente"): ("isolado", "herda o ambiente e sobrescreve TCL_LIBRARY/TK_LIBRARY com pastas inexistentes, para quebrar o Tk de proposito"),
    ("tests/test_verificar_hash_final.py", "ambiente"): ("legitimo", "GUARACI_WSL_DISTRO escolhe a distro do WSL, como no verificador de producao; o padrao e' Ubuntu"),
})
# Lote 2.2 da vistoria 2026-10 (guarda de dados de terceiros)
DECISOES.update({
    ("tests/test_guarda_dados_de_terceiros.py", "caminho"): ("legitimo", "le o dossie privado de calibracao (fora do git) so' se existir, para os nomes de conjuntos nunca ficarem no repositorio; sem ele a guarda usa so' os padroes publicos"),
})


def decisao(chave: Tuple[str, str]) -> Optional[Tuple[str, str]]:
    """Decisao explicita registrada em `DECISOES` (nenhuma regra implicita por pasta)."""
    return DECISOES.get(chave)


IGNORAR = ("tests/test_dependencias_de_maquina.py",)       # carrega as linhas de exemplo da propria guarda


def _arquivos(raiz: Path) -> List[Path]:
    out: List[Path] = []
    for pasta in PASTAS:
        out += [p for p in sorted((raiz / pasta).glob("test_*.py")) if p.relative_to(raiz).as_posix() not in IGNORAR]
    return out


def achados(raiz: Path = RAIZ) -> Dict[Tuple[str, str], List[int]]:
    """{(arquivo, categoria): [linhas]}."""
    rx = {k: re.compile(v) for k, v in CATEGORIAS.items()}
    out: Dict[Tuple[str, str], List[int]] = {}
    for p in _arquivos(raiz):
        rel = p.relative_to(raiz).as_posix()
        for n, linha in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if linha.lstrip().startswith("#"):
                continue
            for cat, r in rx.items():
                if r.search(linha):
                    out.setdefault((rel, cat), []).append(n)
    return out


def comparar(a: Dict[Tuple[str, str], List[int]], decisoes: Dict[Tuple[str, str], Tuple[str, str]] = DECISOES) -> Dict[str, List[str]]:
    return {"sem_decisao": sorted(f"{k[0]} [{k[1]}] linhas {v[:5]}" for k, v in a.items() if k not in decisoes and decisao(k) is None),
            "decisao_obsoleta": sorted(f"{k[0]} [{k[1]}]" for k in decisoes if k not in a)}


def tabela(a: Dict[Tuple[str, str], List[int]]) -> str:
    L = ["# Dependências de máquina nos testes", "",
         "**Público:** quem mantém a suíte · **Estado:** gerado por `scripts/varrer_dependencias_de_maquina.py` (não editar à mão) · "
         "**Atualizado em:** ver `docs/qualidade/estado_do_projeto.md`", "",
         "Todo uso de memória livre, CPU, relógio, caminho de máquina, variável de ambiente ou rede em teste precisa de uma decisão: "
         "`injetado` (valor controlado pelo teste), `bloqueado` (a dependência é desligada de propósito), `isolado` (ambiente próprio), "
         "`corrigido` (tinha dependência real e foi trocada) ou `legitimo` (não afeta o resultado). Uso novo sem decisão reprova.", "",
         "| teste | dependência | estado | corrigido? | por quê |", "|---|---|---|---|---|"]
    for (arq, cat) in sorted(a):
        estado, motivo = decisao((arq, cat)) or ("?", "?")
        L.append(f"| `{arq}` | {cat} | {estado} | {'sim' if estado == 'corrigido' else 'não era necessário'} | {motivo} |")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--verificar", action="store_true")
    a = ap.parse_args(argv)
    ach = achados()
    d = comparar(ach)
    for k, v in d.items():
        for x in v:
            print(f"{k}: {x}")
    texto = tabela(ach)
    atual = SAIDA.read_bytes().decode("utf-8").replace("\r\n", "\n") if SAIDA.exists() else ""
    if a.verificar:
        if atual != texto:
            print("desatualizada: docs/qualidade/dependencias_de_maquina.md")
        return 1 if (any(d.values()) or atual != texto) else 0
    crlf = SAIDA.exists() and b"\r\n" in SAIDA.read_bytes()
    SAIDA.write_bytes((texto.replace("\n", "\r\n") if crlf else texto).encode("utf-8"))
    print("gravado: docs/qualidade/dependencias_de_maquina.md")
    return 1 if any(d.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
