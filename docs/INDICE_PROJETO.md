# Índice do projeto

Ponto de entrada único para toda a documentação do repositório. Cada
linha abaixo é o caminho, o nome e uma frase objetiva do conteúdo —
para o detalhe completo, abra o arquivo.

Dados de amostra de terceiro, material de TCC privado e anotações de
auditoria pessoais ficam deliberadamente **fora** deste repositório, em
`~/.guaraci_local/` (política de isolamento já estabelecida, não
detalhada aqui). Isso não inclui rascunhos de correspondência sobre
licenciamento de dataset público (ex. `docs/RASCUNHOS_CONTATO.md`
abaixo) — esses não contêm dado pessoal e ficam versionados por
transparência de processo.

---

## Visão geral

| Arquivo | Conteúdo |
|---|---|
| [`README.md`](../README.md) | Página de entrada em inglês: o que o Guaraci faz, instalação, comparativo com ferramentas comerciais, licença, citação. |
| [`README.pt-br.md`](../README.pt-br.md) | Mesmo conteúdo do README em português. |
| [`docs/index.md`](index.md) | Página inicial do site de documentação (MkDocs) — resumo de uma tela do projeto. |
| [`CITATION.cff`](../CITATION.cff) | Metadado de citação acadêmica (formato Citation File Format), lido por GitHub/Zenodo. |
| [`paper/paper.md`](../paper/paper.md) | Manuscrito no formato JOSS (Journal of Open Source Software) descrevendo o software para publicação. |
| [`ACKNOWLEDGMENTS.md`](../ACKNOWLEDGMENTS.md) | Agradecimentos — proposital e parcialmente incompleto até autorização das pessoas citadas. |
| [`CODE_OF_CONDUCT.md`](../CODE_OF_CONDUCT.md) | Código de conduta padrão (Contributor Covenant) para a comunidade do projeto. |
| [`CONTRIBUTING.md`](../CONTRIBUTING.md) | Como abrir issues/PRs, convenções de contribuição (aceita português). |
| [`.github/PULL_REQUEST_TEMPLATE.md`](../.github/PULL_REQUEST_TEMPLATE.md) | Modelo preenchido automaticamente ao abrir um pull request no GitHub. |

## Estado de desenvolvimento

| Arquivo | Conteúdo |
|---|---|
| [`docs/PROGRESSO.md`](PROGRESSO.md) | Diário de bordo do desenvolvimento — um registro por passo/instrução, mais recente primeiro, com o que foi investigado, decidido e medido. Não é um resumo polido: é o histórico de investigação em si. |
| [`docs/CHANGELOG.md`](CHANGELOG.md) | Histórico de versões do pipeline, extraído do cabeçalho de `pipeline.py`. |
| [`docs/DESIGN.md`](DESIGN.md) | Proposta de identidade visual/navegação da interface — **não implementada**, exige aprovação explícita antes de qualquer mudança de UI. |
| [`docs/COMMERCIAL.md`](COMMERCIAL.md) | Termos de licenciamento comercial (dual-license) do software. |

## Validação científica

| Arquivo | Conteúdo |
|---|---|
| [`docs/VALIDACAO_PUBLICA.md`](VALIDACAO_PUBLICA.md) | Base de evidência do projeto: validação contra dataset PÚBLICO real para cada uma das 11 técnicas analíticas suportadas (Corn, Mendeley, RMN, HPLC, GC-IMS etc.), com métrica medida, referência da literatura e retratações documentadas quando um achado foi corrigido depois. |
| [`docs/VALIDATION.md`](VALIDATION.md) | Validação NUMÉRICA interna — cada fórmula/estatística do Guaraci comparada contra uma implementação ou valor de referência conhecido (não usa dataset público externo). |
| [`docs/BENCHMARK_TECATOR.md`](BENCHMARK_TECATOR.md) | Um benchmark específico e antigo (dataset Tecator, NIR de gordura em carne) — primeira prova externa do motor antes da política de `docs/VALIDACAO_PUBLICA.md` existir. |

## Compatibilidade e contrato

| Arquivo | Conteúdo |
|---|---|
| [`docs/COMPATIBILITY.md`](COMPATIBILITY.md) | Política de compatibilidade da API pública — o que pode mudar entre versões, o que é coberto pelo teste de contrato automático (`tests/test_contrato_api_publica.py`), exceções documentadas. |
| [`docs/api/`](api/) | Referência de API autogerada (mkdocstrings) para os módulos de cálculo puro: `chemometric_stats`, `classificadores`, `predicao`, `preprocessamento`, `selecao_variaveis`, `validacao_estatistica`. |

## Uso

| Arquivo | Conteúdo |
|---|---|
| [`docs/MANUAL.md`](MANUAL.md) | Manual de instruções de uso — cada funcionalidade, tela e fluxo do CLI/app, mantido atualizado a cada mudança de interface (diferente do README, que é a porta de entrada). |
| [`scripts/gerar_vault_obsidian.py`](../scripts/gerar_vault_obsidian.py) | Gera um vault Obsidian navegável (técnicas, módulos, conceitos, validações, decisões, achados) a partir das fontes deste índice — o vault em si fica fora do repositório (`~/GuaraciVault/` por padrão, ou `GUARACI_VAULT_DIR`); rodar de novo regenera do zero. |

## Dados

| Arquivo | Conteúdo |
|---|---|
| [`datasets/README.md`](../datasets/README.md) | Política de dados públicos: nenhum dado de terceiro é versionado; como cada dataset público usado em `docs/VALIDACAO_PUBLICA.md` é baixado (script + licença + checksum). |

## Segurança

| Arquivo | Conteúdo |
|---|---|
| [`SECURITY.md`](../SECURITY.md) | Política de segurança — principalmente o risco de execução de código arbitrário ao carregar um `.joblib` de terceiro (pickle), e como relatar vulnerabilidades. |

## Correspondência preparada (não enviada)

| Arquivo | Conteúdo |
|---|---|
| [`docs/RASCUNHOS_CONTATO.md`](RASCUNHOS_CONTATO.md) | Rascunhos de e-mail para autores de dataset (ex. dúvida de licença), prontos para revisão e envio manual — nunca disparados por automação. |
