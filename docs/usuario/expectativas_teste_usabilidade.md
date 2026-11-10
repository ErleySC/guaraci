# O que esperar do teste de usabilidade (para quem aplica o teste)

**Público:** quem usa ou testa o programa · **Estado:** manual · **Atualizado em:** ver `docs/qualidade/estado_do_projeto.md`

**Este documento é para o autor/observador, não para quem faz o teste.**

## O objetivo

Descobrir **onde uma pessoa que nunca viu o projeto tropeça** — o que
nenhuma auditoria feita por quem conhece o código consegue revelar. Por
isso, **os pontos em que a pessoa trava são o dado mais importante do
teste, não uma falha dela nem do teste.** Um teste em que ninguém trava
teria pouco a ensinar.

## Regras para quem observa

- Não explique, não dê dica, não corrija. Se a pessoa pedir ajuda, lembre
  que a regra é anotar onde travou e seguir.
- Se ela desistir de uma tarefa, isso é um resultado válido; peça só o
  "onde" e o "por quê".
- A lentidão em si também é dado; não interprete silêncio como problema.
- Guarde qualquer mensagem de erro **exatamente como copiada**.

## O que é um resultado razoável

| Tarefa | Sucesso razoável | Trava provável (e é informação útil) |
|---|---|---|
| 1 Instalar | `pip install ".[reports]"` e `guaraci --version` em até ~10 min (medido em ambiente virgem: 43 s de instalação) | versão do Python; `pip` não encontrado; `doctor` falhando na primeira vez por "Controle de Aplicativo" no Windows (rodar de novo resolve; medido) |
| 2 Classificar | `exemplo classificar` + `run` em ~1 min; acha `resumo_modelo.txt` e a acurácia balanceada | não perceber que a pasta `resultados` fica dentro da pasta do exemplo; termos "acurácia balanceada" e "matriz de confusão" |
| 3 Quantificar | `exemplo quantificar` + `run` em segundos; acha o RMSEP | o que é RMSEP; diferença entre erro de calibração e de predição |
| 4 Comparar grupos | `exemplo comparar` + `anova`; lê a conclusão e entende a recusa sem `--unidade` | "p-valor", "unidade independente", "Tukey/Games-Howell" |
| 5 Pedir ajuda | acha, só pelo programa, o que fazer com Q2 baixo, o selo do PLS-DA e o que faz `max_lvs`; entende o "não sei" do assistente | achar o comando certo sem o manual; esperar que o assistente responda qualquer coisa |
| Repetição | `exemplo explorar` em metade do tempo da tarefa 2 | — |

**Tempo total razoável:** ~50 min. Mais de 80 min sugere que o programa não se explicou sozinho.

## Como ler "curva de aprendizado curta"

A tabela de alvos está em `docs/usuario/roteiro_teste_usabilidade.md`. Os alvos são propostas: com uma pessoa só, o resultado é um achado, não uma estatística. O sinal mais forte é **consulta ao manual nas tarefas 2 a 5** (o objetivo é zero) e **pedido de ajuda ao observador** (idem).

## Conta como sucesso do PRODUTO

- Tarefas 1–3 concluídas **sem** ajuda externa e sem abrir o manual depois da tarefa 1.
- Na tarefa 4 a pessoa explica por que o programa **recusou** o comando sem `--unidade`, e na
  tarefa 5 entende que o "não sei" do assistente é intencional (ele não inventa resposta).

## Conta como achado (a corrigir), não como fracasso

- Qualquer erro cru (traceback) em vez de mensagem em português.
- Qualquer passo em que o manual manda fazer algo que o programa não faz.
- Qualquer termo que a pessoa não interpretou mesmo lendo o manual.
- Qualquer ponto em que ela fez algo razoável e o programa bloqueou.

## Histórico: achados da simulação prévia (feita pelo autor, 2026-09-23)

O autor simulou um usuário novo (pasta pessoal vazia) antes de entregar o
kit; isso já revelou e corrigiu 3 problemas que teriam derrubado o
primeiro contato: (1) `max_lvs=40` quebrava com dataset pequeno (60
amostras); (2) o checklist da CLI bloqueava quem só tinha um CSV (exigia a
pasta `dados`, só usada no modo `dx`); (3) o resumo/auditoria dizia
"`mae_id` confiável" em execução sem nenhum `mae_id`. Além disso, os
números dos campos do menu dependem do modo iniciante/avançado (por isso o
roteiro descreve os campos pelo nome). Só o teste com pessoa real mostra o
que **esta** simulação não viu.
