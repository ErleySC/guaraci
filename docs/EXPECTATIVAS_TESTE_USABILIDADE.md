# O que esperar do teste de usabilidade (para quem aplica o teste)

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
| 1 Instalar | Instala e roda `guaraci doctor` em até ~15 min | Versão do Python, `pip` não encontrado; extras `[web,reports]` (o manual só os cita como opcionais) |
| 2 Demo | Roda (~4 min), acha a pasta `GUARACI_Demo/` e abre uma figura | Não perceber que a pasta é criada onde o comando foi rodado; janela do explorador que abre sozinha (Windows) |
| 3 Rodar CSV | Configura modo `csv` + arquivo e roda em 1–2 min | Os números dos campos do menu mudam (modo iniciante esconde opções avançadas); confirmações "s/n" e "[Enter para continuar]"; achar o campo do arquivo CSV |
| 4 Entender | Diz nº de classes e acurácia; lê "não validado" sem pânico | Termos técnicos ("acurácia balanceada", "grupos"); aviso de ausência de `mae_id` no resumo/auditoria |
| 5 Web (opc.) | Repete a tarefa 3 na interface visual | Extras `[web]` não instalados; upload de CSV vs. caminho local |
| 6 ASCA (opc.) | Executa e lê "% da variância" | Conceito de "fator"/"significativo" — esperado que a pessoa não entenda |

**Tempo total razoável:** ~1 hora. Mais de 90 min sugere que o manual não
bastou.

## Conta como sucesso do PRODUTO

- Tarefas 1–3 concluídas **sem** ajuda externa.
- Na tarefa 4, a pessoa consegue dizer o que o programa **recusou**
  afirmar ("não validado"/"n/a") e que isso é intencional.

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
