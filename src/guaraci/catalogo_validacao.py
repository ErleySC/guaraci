"""catalogo_validacao.py -- registro de evidencias e selos (fonte do assistente
de confianca).

EVIDENCIAS: so' evidencia que vai junto com o pacote e o usuario pode reproduzir
(simulacao e testes automatizados com verdade conhecida). SELOS: o selo de cada
tecnica, apontando para as evidencias. O texto da evidencia vive SO' aqui:
descricoes de opcao e de tecnica nao relatam validacao.

Redacao do selo: variante (b) da decisao D8 do autor, PROVISORIA ate' a decisao
final -- o selo se apoia so' no que vai junto; sobre os valores padrao, uma frase
generica (CALIBRACAO_PADROES), sem citar dado de terceiros.

Estados do selo:
  validado               reservado: exige evidencia de dado real que va' junto com o pacote (hoje nenhuma)
  em_validacao           mecanismo verificado em testes/simulacao com verdade conhecida
  limitado               usado com limite declarado (poucas unidades, sinal fraco)
  sem_validacao_publica  verificacao parcial: o limite principal da tecnica nao e' verificavel nos testes
  nao_aplicavel          configuracao informativa, sem calculo proprio
ASCII por convencao do CLI.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

ESTADOS = ("validado", "em_validacao", "limitado", "sem_validacao_publica", "nao_aplicavel")
# Tipos de evidencia aceitos: so' o que vai junto com o pacote.
TIPOS_EVIDENCIA = ("simulacao", "testes")
ROTULO_ESTADO = {
    "validado": ("validado", "validated"),
    "em_validacao": ("em validacao", "under validation"),
    "limitado": ("limitado", "limited"),
    "sem_validacao_publica": ("verificacao parcial", "partial verification"),
    "nao_aplicavel": ("informativo", "informative"),
}
# (unicode, ascii)
SIMBOLO_ESTADO = {
    "validado": ("●", "[V]"), "em_validacao": ("◐", "[E]"), "limitado": ("○", "[L]"),
    "sem_validacao_publica": ("◌", "[-]"), "nao_aplicavel": ("·", "[.]"),
}

# Frase sobre os valores padrao: so' afirma o que o pacote sustenta (ponto de partida, ajustavel), sem citar dado que nao acompanha o programa.
CALIBRACAO_PADROES: Tuple[str, str] = (
    "Valores padrao: ponto de partida, nao universais; ajustaveis pelas opcoes da tecnica. "
    "O selo se apoia so' na evidencia acima, que vai junto com o pacote e pode ser reproduzida.",
    "Default values: starting points, not universal; adjustable through the technique options. "
    "The seal rests only on the evidence above, which ships with the package and can be reproduced.")

# tipo: um de TIPOS_EVIDENCIA
# Campos: rotulo, ref (arquivo de teste), data, resumo, ressalva, testes (arquivos em tests/), secao (legado, vazio)
EVIDENCIAS: Dict[str, Dict[str, Any]] = {
    "anova_pseudo_replicacao": dict(tipo="simulacao", rotulo=('Simulacao de pseudo-replicacao (2000 execucoes sob H0)', 'Pseudo-replication simulation (2000 runs under H0)'), ref="tests/test_anova_grupos.py", data="2026-10-02",
        resumo=('Tratando replicas como independentes: 57,0% de falsos positivos; agregando por unidade: 4,8% (IC95 de 3,9 a 5,8%). Poder com 6 unidades por grupo e efeito de 1 desvio-padrao: 0,35.', 'Treating replicates as independent: 57.0% false positives; aggregating by unit: 4.8% (95% CI 3.9 to 5.8%). Power with 6 units per group and an effect of 1 standard deviation: 0.35.'),
        ressalva=('Dados sinteticos.', 'Synthetic data.'), testes=("test_anova_grupos.py",), secao=""),
    "anova_blindagem": dict(tipo="testes", rotulo=('Testes automatizados do ANOVA: oraculos independentes, casos de borda e propriedades', 'Automated ANOVA tests: independent oracles, edge cases and properties'), ref="tests/test_anova_blindagem.py", data="2026-10-06",
        resumo=('F, p, Tukey-Kramer, Games-Howell, Levene, eta2, omega2 e FDR conferidos contra SciPy e formulas ingenuas; invariancia a escala, deslocamento e ordem.', 'F, p, Tukey-Kramer, Games-Howell, Levene, eta2, omega2 and FDR checked against SciPy and naive formulas; invariance to scale, shift and order.'), ressalva=('Dados sinteticos.', 'Synthetic data.'), testes=("test_anova_blindagem.py", "test_anova_grupos.py", "test_anova_cli_e2e.py"), secao=""),
    "kruskal_testes": dict(tipo="testes", rotulo=('Testes automatizados de Kruskal-Wallis, Dunn e do diagnostico de premissas', 'Automated tests of Kruskal-Wallis, Dunn and the assumption diagnostics'), ref="tests/test_anova_nao_parametrico.py", data="2026-10-06",
        resumo=('Kruskal-Wallis conferido contra SciPy e Dunn contra implementacao ingenua; invariancia a transformacao monotona; limiares da recomendacao nas bordas.', 'Kruskal-Wallis checked against SciPy and Dunn against a naive implementation; invariance to monotone transforms; recommendation thresholds at the edges.'), ressalva=('Dados sinteticos; sem exemplo publicado conferido na fonte.', 'Synthetic data; no published example checked at the source.'), testes=("test_anova_nao_parametrico.py", "test_anova_cli_e2e.py"), secao=""),
    "vip_testes": dict(tipo="testes", rotulo=('Testes automatizados do VIP com estabilidade', 'Automated tests of the VIP with stability'), ref="tests/test_vip_estabilidade.py", data="2026-10-06",
        resumo=('VIP conferido contra PLS1 por NIPALS feito a mao; soma de VIP ao quadrado igual ao numero de variaveis; variaveis informativas estaveis em dado sintetico.', 'VIP checked against hand-written NIPALS PLS1; sum of squared VIP equals the number of variables; informative variables stable on synthetic data.'), ressalva=('Dados sinteticos.', 'Synthetic data.'), testes=("test_vip_estabilidade.py", "test_vip_cli.py"), secao=""),
    "conformal_testes": dict(tipo="testes", rotulo=('Testes automatizados de predicao conformal e identificacao', 'Automated tests of conformal prediction and identification'), ref="tests/test_conformal.py", data="2026-09-20",
        resumo=('Cobertura por grupo e limite alpha minimo verificados em dados sinteticos.', 'Per-group coverage and minimum alpha limit checked on synthetic data.'), ressalva=('Dados sinteticos.', 'Synthetic data.'), testes=("test_conformal.py", "test_identificacao_cobertura_sessao.py"), secao=""),
    "classificadores_testes": dict(tipo="testes", rotulo=('Testes automatizados de classificadores (DD-SIMCA, PLS-DA)', 'Automated tests of classifiers (DD-SIMCA, PLS-DA)'), ref="tests/test_classificadores.py", data="2026-09-20",
        resumo=('Comportamento do DD-SIMCA e do PLS-DA verificado em dados sinteticos.', 'DD-SIMCA and PLS-DA behaviour checked on synthetic data.'), ressalva=('Dados sinteticos; a sensibilidade por grupo depende do seu delineamento.', 'Synthetic data; per-group sensitivity depends on your design.'), testes=("test_classificadores.py",), secao=""),
    "selecao_amostras_testes": dict(tipo="testes", rotulo=('Testes automatizados de selecao de amostras', 'Automated tests of sample selection'), ref="tests/test_selecao_amostras.py", data="2026-09-20",
        resumo=('Kennard-Stone, Duplex e SPXY verificados em dados sinteticos.', 'Kennard-Stone, Duplex and SPXY checked on synthetic data.'), ressalva=('', ''), testes=("test_selecao_amostras.py",), secao=""),
    "amostragem_ativa_testes": dict(tipo="testes", rotulo=('Testes automatizados da priorizacao por amostragem ativa', 'Automated tests of active-sampling prioritization'), ref="tests/test_amostragem_ativa.py; tests/test_tarefas_cli.py", data="2026-10-04",
        resumo=('Ordenacao e contagem de sessoes faltantes conferidas em dado sintetico; prova pelo CLI (tela J) com mae_id sintetico no formato do modo dx.', 'Ordering and missing-session counts checked on synthetic data; CLI proof (screen J) with synthetic mae_id in the dx-mode format.'), ressalva=('Dados sinteticos; exige especie, adulterante e sessao codificados no identificador.', 'Synthetic data; requires species, adulterant and session encoded in the identifier.'), testes=("test_amostragem_ativa.py", "test_tarefas_cli.py"), secao="5"),
    "pooled_local_testes": dict(tipo="testes", rotulo=('Testes automatizados da decisao modelo unico x modelo por especie', 'Automated tests of the pooled x per-species decision'), ref="tests/test_politica_pooled_local.py", data="2026-09-04",
        resumo=('Recomendacao fica no modelo unico sem poder suficiente; troca so com ganho pareado significativo, em dado sintetico.', 'The recommendation stays pooled without enough power; it switches only with a significant paired gain, on synthetic data.'), ressalva=('Dados sinteticos.', 'Synthetic data.'), testes=("test_politica_pooled_local.py", "test_portao_pooled_cli.py"), secao="5"),
    "figuras_merito_testes": dict(tipo="testes", rotulo=('Testes automatizados de figuras de merito e RPD/RER', 'Automated tests of figures of merit and RPD/RER'), ref="tests/test_pipeline_core.py", data="2026-09-20",
        resumo=('LOD, LOQ, sensibilidade e RPD/RER verificados em dados sinteticos.', 'LOD, LOQ, sensitivity and RPD/RER checked on synthetic data.'), ressalva=('A faixa de interpretacao do RPD/RER e generica da literatura.', 'The RPD/RER interpretation range is generic from the literature.'), testes=("test_pipeline_core.py",), secao="5"),
    "linearidade_testes": dict(tipo="testes", rotulo=('Testes automatizados de falta de ajuste', 'Automated tests of lack of fit'), ref="tests/test_linearity.py", data="2026-09-20",
        resumo=('Teste de falta de ajuste e caso nao computavel verificados em dados sinteticos.', 'Lack-of-fit test and the non-computable case checked on synthetic data.'), ressalva=('', ''), testes=("test_linearity.py",), secao=""),
    "robustez_testes": dict(tipo="testes", rotulo=('Testes automatizados do protocolo de robustez', 'Automated tests of the robustness protocol'), ref="tests/test_robustness.py", data="2026-09-20",
        resumo=('Perturbacoes e relatorio min/mediana/max verificados em dados sinteticos.', 'Perturbations and the min/median/max report checked on synthetic data.'), ressalva=('', ''), testes=("test_robustness.py",), secao=""),
    "mcr_testes": dict(tipo="testes", rotulo=('Testes automatizados de MCR-ALS', 'Automated tests of MCR-ALS'), ref="tests/test_mcr_als.py", data="2026-09-20",
        resumo=('Decomposicao verificada em misturas sinteticas.', 'Decomposition checked on synthetic mixtures.'), ressalva=('A solucao nao e unica; o resultado e de interpretacao e nao substitui a quantificacao por PLS.', 'The solution is not unique; the result is for interpretation and does not replace PLS quantification.'), testes=("test_mcr_als.py",), secao=""),
    "perfil_matriz_testes": dict(tipo="testes", rotulo=('Testes automatizados dos perfis de tecnica', 'Automated tests of technique profiles'), ref="tests/test_perfil_matriz.py", data="2026-09-20",
        resumo=('Leitura, combinacao e vocabulario dos perfis verificados.', 'Reading, combination and vocabulary of the profiles checked.'), ressalva=('', ''), testes=("test_perfil_matriz.py",), secao=""),
    "mspc_simulacao": dict(tipo="simulacao", rotulo=('Deriva espectral sintetica injetada por sessao', 'Synthetic spectral drift injected by session'), ref="tests/test_mspc_validacao_deriva.py", data="2026-09-25",
        resumo=('Dominio real (PCA + T2/Q) e teste de deriva ponta a ponta; alarma com deriva e fica quieto sem ela.', 'Real domain (PCA + T2/Q) and drift test end to end; flags with drift and stays quiet without it.'),
        ressalva=('Dados sinteticos.', 'Synthetic data.'), testes=("test_mspc_validacao_deriva.py", "test_monitorar_cli.py"), secao=""),
    "pls_r_testes": dict(tipo="testes", rotulo=('Testes automatizados da regressao PLS', 'Automated tests of PLS regression'), ref="tests/test_pipeline_core.py; tests/test_figuras_regressao.py", data="2026-10-11",
        resumo=('Ajuste PLS e metricas de regressao (R2, erro) verificados em dados sinteticos com sinal conhecido; y constante tratado sem quebrar.', 'PLS fit and regression metrics (R2, error) checked on synthetic data with a known signal; constant y handled without breaking.'), ressalva=('Dados sinteticos.', 'Synthetic data.'), testes=("test_pipeline_core.py", "test_figuras_regressao.py"), secao=""),
    "transferencia_testes": dict(tipo="testes", rotulo=('Testes automatizados de transferencia de calibracao (DS e PDS)', 'Automated tests of calibration transfer (DS and PDS)'), ref="tests/test_transferencia_calibracao.py", data="2026-10-11",
        resumo=('DS e PDS reduzem o erro entre instrumentos com deslocamento conhecido por construcao, em dados sinteticos.', 'DS and PDS reduce the error between instruments with a shift known by construction, on synthetic data.'), ressalva=('O ganho depende do par de instrumentos; confira no seu par.', 'The gain depends on the instrument pair; check it on your pair.'), testes=("test_transferencia_calibracao.py",), secao=""),
    "icp_testes": dict(tipo="testes", rotulo=('Testes automatizados do perfil multielementar (ICP)', 'Automated tests of the multi-element profile (ICP)'), ref="tests/test_icp_multielementar.py", data="2026-10-11",
        resumo=('Leitura, censura abaixo do limite de deteccao, padrao interno e QC verificados em fixtures sinteticas.', 'Reading, censoring below the detection limit, internal standard and QC checked on synthetic fixtures.'), ressalva=('Dados sinteticos.', 'Synthetic data.'), testes=("test_icp_multielementar.py", "test_lacunas_icp_multielementar.py", "test_multielementar_pipeline.py"), secao=""),
    "parafac_testes": dict(tipo="testes", rotulo=('Testes automatizados do PARAFAC para EEM', 'Automated tests of PARAFAC for EEM'), ref="tests/test_eem_multiway.py", data="2026-10-11",
        resumo=('PARAFAC recupera as proporcoes de uma mistura sintetica de componentes puros conhecidos.', 'PARAFAC recovers the proportions of a synthetic mixture of known pure components.'), ressalva=('Dados sinteticos.', 'Synthetic data.'), testes=("test_eem_multiway.py", "test_parafac_cow_relatorio_cli.py"), secao=""),
    "cow_testes": dict(tipo="testes", rotulo=('Testes automatizados do alinhamento COW', 'Automated tests of COW alignment'), ref="tests/test_alinhamento_retencao.py", data="2026-10-11",
        resumo=('COW recupera um deslocamento nao linear conhecido em cromatograma sintetico (correlacao sobe apos o alinhamento).', 'COW recovers a known non-linear warp in a synthetic chromatogram (correlation rises after alignment).'), ressalva=('Dados sinteticos.', 'Synthetic data.'), testes=("test_alinhamento_retencao.py", "test_parafac_cow_relatorio_cli.py"), secao=""),
    "portao_testes": dict(tipo="testes", rotulo=('Testes automatizados do portao de aceite de correcao de sinal', 'Automated tests of the signal-correction acceptance gate'), ref="tests/test_portao_correcao_sinal.py", data="2026-10-11",
        resumo=('Metodo inutil ou prejudicial e rejeitado; metodo com ganho sintetico conhecido e aprovado.', 'A useless or harmful method is rejected; a method with a known synthetic gain is approved.'), ressalva=('Dados sinteticos.', 'Synthetic data.'), testes=("test_portao_correcao_sinal.py",), secao=""),
}

# id -> estado, evid (ids de EVIDENCIAS), nota (curta, opcional)
SELOS: Dict[str, Dict[str, Any]] = {
    "pls_da": dict(estado="em_validacao", evid=("classificadores_testes",), nota=('Com poucas amostras por classe o sinal pode ser fraco; use validacao agrupada.', 'With few samples per class the signal may be weak; use grouped validation.')),
    "pls_r": dict(estado="em_validacao", evid=("pls_r_testes", "figuras_merito_testes"), nota=('', '')),
    "pls_r_especie": dict(estado="sem_validacao_publica", evid=("pooled_local_testes",), nota=('Exige n minimo por especie.', 'Requires a minimum n per species.')),
    "ddsimca": dict(estado="sem_validacao_publica", evid=("classificadores_testes",), nota=('Sensibilidade por grupo nao e validavel com um grupo so por classe.', 'Per-group sensitivity cannot be validated with a single group per class.')),
    "conformal": dict(estado="em_validacao", evid=("conformal_testes",), nota=('Alpha minimo 1/(n+1).', 'Minimum alpha 1/(n+1).')),
    "identificacao": dict(estado="sem_validacao_publica", evid=("conformal_testes",), nota=('Poucas combinacoes tem sessoes independentes suficientes.', 'Few combinations have enough independent sessions.')),
    "selecao_amostras": dict(estado="em_validacao", evid=("selecao_amostras_testes",), nota=('', '')),
    "transferencia": dict(estado="em_validacao", evid=("transferencia_testes",), nota=('', '')),
    "figuras_merito": dict(estado="em_validacao", evid=("figuras_merito_testes",), nota=('', '')),
    "linearidade": dict(estado="em_validacao", evid=("linearidade_testes",), nota=('', '')),
    "robustez": dict(estado="em_validacao", evid=("robustez_testes",), nota=('', '')),
    "mcr_als": dict(estado="limitado", evid=("mcr_testes",), nota=('Ferramenta de interpretacao, nao de quantificacao.', 'Interpretation tool, not a quantification tool.')),
    "anova": dict(estado="em_validacao", evid=("anova_pseudo_replicacao", "anova_blindagem"), nota=('Poder baixo com menos de 6 unidades por grupo.', 'Low power with fewer than 6 units per group.')),
    "kruskal": dict(estado="em_validacao", evid=("anova_blindagem", "kruskal_testes"), nota=('Compara distribuicoes; poder baixo com poucas unidades por grupo.', 'Compares distributions; low power with few units per group.')),
    "vip_estab": dict(estado="em_validacao", evid=("vip_testes",), nota=('VIP > 1 e regra pratica; selecionar e validar nos mesmos dados e otimista.', 'VIP > 1 is a rule of thumb; selecting and validating on the same data is optimistic.')),
    "multielementar": dict(estado="em_validacao", evid=("icp_testes",), nota=('Poucas unidades independentes por grupo reduzem o poder; declare a unidade das concentracoes.', 'Few independent units per group reduce power; declare the concentration unit.')),
    "parafac": dict(estado="em_validacao", evid=("parafac_testes",), nota=("Escolha do numero de componentes e do usuario.", "The number of components is the user's choice.")),
    "cow": dict(estado="em_validacao", evid=("cow_testes",), nota=("Verificado so como mecanismo (sem rotulo).", "Checked only as a mechanism (no labels).")),
    "monitoramento": dict(estado="em_validacao", evid=("mspc_simulacao",), nota=('Exige 19 amostras por lote para testar com alpha 0,05.', 'Requires 19 samples per batch to test with alpha 0.05.')),
    "portao_correcao": dict(estado="em_validacao", evid=("portao_testes",), nota=('Sem poder suficiente o veredito nao conclui nada; o veredito vale para o dado em que o portao roda.', 'Without enough power the verdict concludes nothing; the verdict holds for the data the gate runs on.')),
    "amostragem_ativa": dict(estado="limitado", evid=("amostragem_ativa_testes",), nota=('So prioriza com mae_id no formato que codifica sessao e adulterante (modo dx).', 'Only prioritizes with mae_id in the format that encodes session and adulterant (dx mode).')),
    "pooled_local": dict(estado="sem_validacao_publica", evid=("pooled_local_testes",), nota=('Exige grupo (mae_id) e ao menos 6 amostras por especie.', 'Requires a group (mae_id) and at least 6 samples per species.')),
    "informativo": dict(estado="nao_aplicavel", evid=("perfil_matriz_testes",), nota=('', '')),
}

# opcao -> selo (so' as que tem selo proprio)
OPCAO_SELO = {
    "ddsimca": "ddsimca", "modo_ddsimca": "ddsimca", "ddsimca_pcv": "ddsimca",
    "icp_unidade": "multielementar", "icp_coluna_amostra": "multielementar", "icp_coluna_tipo": "multielementar",
    "icp_coluna_sessao": "multielementar", "icp_padrao_interno": "multielementar",
    "icp_politica_censura": "multielementar", "icp_limiar_censura": "multielementar", "icp_log": "multielementar",
    "perfil_matriz": "informativo", "perfil_tecnica": "informativo",
    "eixo_unidade": "informativo", "tecnica_analitica": "informativo",
}

