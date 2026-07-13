# GUARACI

Plataforma Python de quimiometria multitécnica, aberta e reprodutível, para
classificação, autenticação, exploração e quantificação de matrizes complexas
(FT-NIR, NIR, MIR, Raman, UV-Vis, fluorescência, HPLC, GC-MS, RMN, IMS).

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
| Cético/revisor, quer saber se os números são confiáveis | [Validação científica](VALIDATION.md) |
| Vai carregar um modelo `.joblib` de terceiro | [Segurança](SECURITY.md) |
| Quer ver o motor rodando fora dos dados do autor | [Benchmark externo (Tecator)](BENCHMARK_TECATOR.md) |
| Desenvolvedor, quer a assinatura das funções do núcleo | [Referência da API](api/chemometric_stats.md) |

## Três interfaces, um motor

CLI (Rich), aplicativo web (Streamlit) e uso direto via `python -m
guaraci.pipeline` chamam a mesma função `executar()` — nenhuma lógica
duplicada entre interfaces.
