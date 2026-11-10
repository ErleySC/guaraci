# GUARACI — Plataforma quimiométrica com validação anti-vazamento por padrão

**Público:** todos · **Estado:** manual · **Atualizado em:** ver `docs/qualidade/estado_do_projeto.md`

Plataforma Python de quimiometria multitécnica, aberta e reprodutível, para
classificação, autenticação, exploração e quantificação
(FT-NIR, NIR, MIR, Raman, UV-Vis, fluorescência, HPLC, GC-MS, RMN, IMS e outras técnicas, por perfil).

**Diferencial metodológico:** validação *group-aware* (`GroupKFold`/
`StratifiedGroupKFold` por `mae_id`) — réplicas físicas da mesma amostra nunca
caem em treino e teste ao mesmo tempo, um vazamento comum e sub-relatado na
literatura de espectroscopia.

- Código-fonte: <https://github.com/ErleySC/guaraci>
- Demo online (Streamlit): <https://guaraci.streamlit.app/>
- Notebook "Guaraci em 5 minutos" (Colab): [abrir](https://colab.research.google.com/github/ErleySC/guaraci/blob/master/notebooks/guaraci_5_minutos.ipynb)

## Por onde começar

| Se você é... | Comece por... |
|---|---|
| Curioso, decidindo se vale testar | [README completo (GitHub)](https://github.com/ErleySC/guaraci#readme) |
| Vai usar de verdade | [Manual](MANUAL.md) |
| Cético/revisor, quer saber se os números são confiáveis | [Validação científica](cientifico/validacao_numerica.md) |
| Quer saber o que o projeto NÃO garante | [Limitações conhecidas](cientifico/limitacoes.md) |
| Vai carregar um modelo `.joblib` de terceiro | [Segurança](SECURITY.md) |
| Desenvolvedor, quer a assinatura das funções do núcleo | [Referência da API](api/chemometric_stats.md) |

## Três interfaces, um motor

CLI (Rich), aplicativo web (Streamlit) e uso direto via `python -m
guaraci.pipeline` chamam a mesma função `executar()` — nenhuma lógica
duplicada entre interfaces.

## Comandos que explicam a si mesmos

`guaraci --help` lista os comandos por objetivo; `guaraci ajuda <tópico>` explica qualquer opção,
técnica, gráfico ou tela no mesmo formato; `guaraci assistente "pergunta"` responde offline a partir
do catálogo; `guaraci confianca <técnica>` mostra o selo e a evidência; `guaraci exemplo` monta um
cenário guiado com dado sintético. Detalhes em [Manual](MANUAL.md), seção 5.11.

## Comparar grupos e importância de variáveis

`guaraci anova` compara grupos sobre a unidade independente (padrão Games-Howell, escolhido por simulação; Tukey a pedido; Kruskal-Wallis com Dunn; `--metodo recomendar` mostra o diagnóstico de premissas e a sugestão, sem trocar o método em silêncio). `guaraci vip` recalcula o VIP do PLS nas dobras de uma validação agrupada e mostra a estabilidade de cada variável. Detalhes no [Manual](MANUAL.md), seções 2.8b e 2.8c.
