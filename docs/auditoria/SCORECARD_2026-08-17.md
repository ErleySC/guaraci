# Scorecard — auditoria mestre de 2026-08-17

Pontuação por item: **0** ausente · **0,5** parcial · **1** completo ·
**N/V** não verificável (fora do denominador). Cada linha tem a evidência
que a sustenta. O relatório completo **não é versionado**: descrevia o
acervo de terceiro (inventário de espécies, janela de aquisição, métricas
do dataset não publicado) e foi retirado do repositório em 2026-08-19.

Estado marcado **✅** foi corrigido nesta rodada; a nota reflete o estado
**depois** da correção, e a coluna de evidência diz qual era antes.

---

## 1. Metodologia científica e validação — peso 25 %

| # | Item | Nota | Evidência |
|---|---|---|---|
| 1.1 | Rótulo não entra no caminho de quantificação | **0** | `pipeline.py:785,868` — calibração é por espécie×adulterante conhecidos |
| 1.2 | Sem vazamento indireto (pré-proc dentro do CV) | **1** | `preprocessamento.py:128-141`; MSC/SG/centragem dentro do `Pipeline` |
| 1.3 | Modo cego é o padrão da API/CLI | **0** | `grep -rni "cego\|modo_controle\|--modo" src/` → 0 |
| 1.4 | Teste que falha se o rótulo vazar | **0** | não existe |
| 1.5 | Responde corretamente a amostra não adulterada / adulterante fora do treino | **0** | `predicao.py:245` — `argmax` sempre força uma classe |
| 1.6 | Detecção de fora-de-domínio no caminho padrão | **1** ✅ | colunas `AD_*` em `predicao.py`; **antes** aceitava 0,14–0,57 da própria calibração (`medir_ad_vies_insample.py`) |
| 1.7 | Arquitetura detectar→identificar→quantificar | **0** | não existe |
| 1.8 | Ordem de leitura descorrelacionada do teor | **0** | ρ de Spearman próximo de +1 em **todos** os blocos medidos (`medir_ordem_leitura.py`; valores em doc. local) |
| 1.9 | Puro e adulterado na mesma sessão (declaração do autor) | **1** | confirmado em 13/14 espécies |
| 1.10 | Brancos/releituras para estimar deriva | **0** | quase nenhum `mae_id` lido em mais de um dia (contagem em doc. local) |
| 1.11 | Split por grupo físico, não por espectro | **1** | `GroupKFold`/`GroupShuffleSplit` por `mae_id` |
| 1.12 | Sessão única declarada como limite de robustez | **0,5** | limitações documentadas no MANUAL, mas sem esta |
| 1.13 | CV aninhada em todos os métodos comparados | **1** | `_avaliar_busca_nested_cv`, `selecao_variaveis.py` |
| 1.14 | RMSEC/RMSECV/RMSEP/R²pred/bias | **1** | `pipeline.py:1036,2473` |
| 1.15 | RPD e RER | **0** | ausentes |
| 1.16 | Matriz de confusão + métricas por classe | **1** | `modos_analise.py:140` |
| 1.17 | LOD/LOQ com método declarado | **0,5** | implementados (Valderrama 2009, `chemometric_stats.py:415`), mas saem `N/A` sem réplicas |
| 1.18 | Incerteza entregue como intervalo | **1** | BCa, `validacao_estatistica.py:204` |
| 1.19 | Linearidade / faixa de trabalho / robustez | **0** | não reportados |
| 1.20 | Aderência a AOCS/AOAC/ASTM/ISO declarada | **0** | não declarada |
| 1.21 | `Nh`/`Nq` conforme a referência primária | **1** | medido: truncagem muda a cobertura em ≤0,003 — **retratado**, não é achado |
| 1.22 | Modelo generaliza para matriz não-óleo | **1** | Corn: RMSEP 0,171 %m/m, dentro da literatura |
| 1.23 | Configuração por matriz externalizada | **0,5** | faixa espectral e pré-proc sim; `CODIGO_ESPECIE`/`ADULTERANTE_NOME`/`BANDAS_NIR` hardcoded |
| 1.24 | Matriz sem calibração falha em vez de predizer | **0** | não há conceito de matriz; prediz sempre |
| 1.25 | Saída não afirma química da matriz errada | **0,5** ✅ | model card de-hardcoded nesta rodada; vocabulário "adulterados" ainda vaza (`pipeline.py:2389`) |

**Subtotal: 12,5 / 25 → 50,0 %** (0 itens N/V)

---

## 2. Código e testes — peso 20 %

| # | Item | Nota | Evidência |
|---|---|---|---|
| 2.1 | Roda ponta a ponta em ambiente limpo | **1** | execução completa no Corn |
| 2.2 | Determinismo / seeds | **1** ✅ | 2 `default_rng(42)` fixos trocados por `cfg.seed` |
| 2.3 | Lockfile + versão de Python declarada | **1** | `requirements-lock.txt`; `requires-python=">=3.10"` |
| 2.4 | Testes unitários nas funções numéricas | **1** | núcleo científico ≥95 % |
| 2.5 | Cobertura real | **0,5** | 70 % total; `guaraci.py` em 36 % |
| 2.6 | `except` genéricos sob controle | **1** | 53, 100 % com `noqa` justificado; zero `except: pass` |
| 2.7 | Sem falha silenciosa no caminho de produção | **1** ✅ | fallback circular de `q_ucl` agora recusa (`predicao.py`) |
| 2.8 | Regra de decisão única (sem 4ª cópia) | **1** ✅ | `predicao.py` passa a usar a distância combinada |
| 2.9 | Duplicação consolidada | **1** ✅ | `q_residuos_loo` unificada (2 cópias → 1) |
| 2.10 | Código morto removido | **1** | verificado; `dominio_aplicabilidade` é API pública, não órfã |
| 2.11 | Complexidade sob controle | **0** | `executar()` ~1.500 linhas; `guaraci.py` 3.975 |
| 2.12 | Backend/API | **N/V** | não há |

**Subtotal: 10,5 / 11 → 95,5 %** (1 N/V)

---

## 3. Dados: limpeza, proveniência, vazamento — peso 15 %

| # | Item | Nota | Evidência |
|---|---|---|---|
| 3.1 | Proveniência documentada por arquivo | **0,5** | instrumento/data/operador estão nos `.dx`, mas não há dicionário externo |
| 3.2 | Duplicatas, vazios, unidades, encoding | **1** | `prescan_dx`, `validar_entrada` |
| 3.3 | Brutos separados dos processados | **1** | dados fora do repo; processamento é script |
| 3.4 | Dicionário de dados | **0** | não existe |
| 3.5 | Nenhum dado pessoal versionado | **1** | verificado |
| 3.6 | Nenhum espectro no repo ou histórico | **1** | `git log --all --diff-filter=A --name-only` |
| 3.6b | Nenhum **identificador de amostra** em arquivo versionado | **0** | **3 identificadores reais** versionados (ver §2 do relatório) — item que eu tinha dado como completo e **retratei** |
| 3.7 | Metadados dos brutos sanitizados | **0** | nome de instituição + nome de operador em **todos** os `.dx` do acervo |
| 3.8 | Licença de cada dataset público citada | **1** | Corn: fonte e termos citados em `datasets_publicos.py` |
| 3.9 | Validação externa executada | **1** | Corn, reproduzindo a literatura |
| 3.10 | Comparação com resultados publicados | **1** | RMSEP 0,171 vs faixa 0,1–0,2 |
| 3.11 | Segundo/terceiro dataset público (mel, Mendeley) | **N/V** | Mendeley retorna HTTP 403 sem credencial |

**Subtotal: 8,5 / 11 → 77,3 %** (1 N/V)

---

## 4. Jurídico, titularidade, licenciamento — peso 15 %

| # | Item | Nota | Evidência |
|---|---|---|---|
| 4.1 | Autoria única coerente no histórico | **1** | 91/96 commits do autor; 0 coautores humanos |
| 4.2 | Histórico serve como prova temporal | **0** | reescrito em 2026-08-16; janela 07-04→08-17 não é real |
| 4.3 | Nenhum código de terceiro incorporado | **1** | `git log --format='%an <%ae>'` |
| 4.4 | Mapa de fatos para a Lei 9.609/98 | **1** | instrumento institucional + operadores terceiros, medidos |
| 4.5 | Proveniência dos dados documentada, não apagada | **0,5** | os fatos estão registrados aqui; falta `ACKNOWLEDGMENTS` |
| 4.6 | `LICENSE` é licença padrão identificável | **1** | GPL-3.0-or-later, texto íntegro |
| 4.7 | Sem cláusula não comercial | **1** | ausente — o P0 previsto no prompt não se aplica |
| 4.8 | Duplo licenciamento coerente | **1** | `COMMERCIAL.md` + `CITATION.cff` + `pyproject.toml` alinhados |
| 4.9 | Licença de cada dependência compatível | **1** | todas BSD/MIT/Apache/PSF |
| 4.10 | Nenhuma referência institucional versionada | **1** | verificado; só Lattes do autor (balde AFILIAÇÃO) |
| 4.11 | CLA para contribuidor futuro | **0** | não existe |
| 4.12 | Licença cobre uso em rede (SaaS) | **0,5** | GPL-3.0 não cobre; AGPL cobriria — é decisão, não defeito |

**Subtotal: 9 / 12 → 75,0 %** (0 N/V)

---

## 5. Segurança — peso 10 %

| # | Item | Nota | Evidência |
|---|---|---|---|
| 5.1 | Sem segredos no HEAD | **1** | varredura por padrões de token/chave |
| 5.2 | Sem segredos no histórico | **1** | 96 commits varridos |
| 5.3 | `pickle`/`joblib` protegido | **1** | `confiar=True` + SHA-256 antes do `load` |
| 5.4 | Sem `eval`/`exec`; `subprocess` sem entrada do usuário | **1** ✅ | 3 usos, args constantes; `shell=True` documentado |
| 5.5 | YAML com `safe_load` | **1** | `config_io.py:320` |
| 5.6 | Dependências sem CVE | **1** | `pip-audit --vulnerability-service osv` |
| 5.7 | Permissões de CI mínimas | **1** ✅ | `contents: read` adicionado aos 2 workflows que faltavam |
| 5.8 | Path traversal | **1** | `app_logic.py:223` |
| 5.9 | App público em produção reflete o código corrigido | **N/V** | não verificável sem login (CLAUDE.md, pendência nº 1) |

**Subtotal: 8 / 8 → 100 %** (1 N/V)

---

## 6. Documentação, CLI, referências — peso 10 %

| # | Item | Nota | Evidência |
|---|---|---|---|
| 6.1 | `--help` completo, flags consistentes | **0,5** | existe, mas só `demo\|doctor\|--version` |
| 6.2 | Modo cego padrão na CLI | **0** | não existe |
| 6.3 | Saída declara valor + incerteza + domínio + versão | **0,5** | falta a matriz usada |
| 6.4 | README com instalação testada e limitações | **1** | verificado |
| 6.5 | Model card | **1** ✅ | gerado automaticamente; matriz de-hardcoded nesta rodada |
| 6.6 | Referências verificadas por DOI | **1** ✅ | 16 DOIs; 2 estavam errados, **corrigidos** e conferidos no Crossref |
| 6.7 | Nenhuma referência inventada | **1** | nenhuma encontrada |
| 6.8 | Docstrings nas funções públicas | **1** | verificado |

**Subtotal: 6 / 8 → 75,0 %** (0 N/V)

---

## 7. Repositório e CI — peso 5 %

| # | Item | Nota | Evidência |
|---|---|---|---|
| 7.1 | Estrutura coerente | **1** | — |
| 7.2 | `.gitignore` cobre dados/artefatos/credenciais | **1** | — |
| 7.3 | Histórico legível | **1** | mensagens informativas |
| 7.4 | CI roda testes e lint a cada push, nos SOs declarados | **1** | Ubuntu/Windows/macOS, Py 3.10–3.13 |
| 7.5 | SemVer + CHANGELOG | **1** | `docs/CHANGELOG.md` |
| 7.6 | Issues/PRs triadas | **0** | 5 PRs do Dependabot sem revisão; 11 commits locais não enviados |

**Subtotal: 5 / 6 → 83,3 %** (0 N/V)

---

## Consolidado

| Dimensão | Peso | Pontuação | Contribuição |
|---|---:|---:|---:|
| Metodologia científica e validação | 25 % | 50,0 % | 12,50 |
| Código e testes | 20 % | 95,5 % | 19,10 |
| Dados (limpeza, proveniência, vazamento) | 15 % | 77,3 % | 11,60 |
| Jurídico, titularidade, licenciamento | 15 % | 75,0 % | 11,25 |
| Segurança | 10 % | 100 % | 10,00 |
| Documentação, CLI, referências | 10 % | 75,0 % | 7,50 |
| Repositório e CI | 5 % | 83,3 % | 4,17 |
| **TOTAL** | **100 %** | | **76,1 % completo → 23,9 % a corrigir** |

**Itens `N/V`: 3 de 75 (4,0 %)** — abaixo do limiar de 20 %, então o
percentual é confiável no sentido do prompt.

**Mas leia o número com cuidado.** Ele é uma média ponderada de
checklists, e a dimensão que pesa mais é justamente a que pontua pior. Os
77 % descrevem um software bem construído; os 50 % da primeira dimensão
descrevem o que impede o resultado de ser citável. Um agregado alto não
compensa um P0 metodológico — e dois dos três P0 abertos não são
corrigíveis por código.
