# Roteiro de teste de usabilidade — GUARACI

**Leia isto primeiro.**

- **Não peça ajuda ao autor durante o teste.** Se travar, anote **onde** e
  **por quê** (e a mensagem de erro exata, se houver) e passe para a
  próxima tarefa. Travar **não é falha sua** — é justamente o que este
  teste quer descobrir.
- Você **não precisa saber quimiometria** nem conhecer o GUARACI. O
  programa faz análise de dados de espectros (curvas medidas em amostras)
  para reconhecer o tipo de amostra. Tudo o que precisa está no
  `docs/MANUAL.md`, seção 0 "Primeiros passos" — **use só o manual**.
- Anote o **tempo aproximado** de cada tarefa e preencha, ao final,
  `docs/TEMPLATE_FEEDBACK_TESTE.md`.

Você precisa de: um computador com Python 3.10+ e internet para instalar,
e a pasta do projeto. Reserve cerca de 1 hora.

---

## Tarefa 1 — Instalar e conferir o ambiente

1. Instale o programa seguindo o manual (seção 0, passos 1 e 2).
2. Rode o comando que confere o ambiente.

**Você terminou quando:** o comando de conferência rodou e você sabe dizer
se está tudo certo ou se falta algo.

## Tarefa 2 — Ver o programa funcionando com dados de teste

1. Siga o passo 3 da seção 0 do manual (leva alguns minutos).
2. Encontre a pasta de resultados. Abra **uma** figura e o arquivo de
   resumo numérico.

**Você terminou quando:** consegue dizer, com suas palavras, **o que a
figura mostra** (mesmo que seja "não entendi") e **onde achou** a
"acurácia balanceada" no resumo.

## Tarefa 3 — Rodar com um arquivo de dados

Você vai usar um arquivo de exemplo (dados **inventados**, só para teste).

1. Gere o arquivo: `python scripts/gerar_dados_exemplo_teste.py`
   (cria `dados_exemplo.csv` na pasta atual — uma tabela com uma coluna
   `classe` e 200 colunas numéricas).
2. Abra o assistente (`guaraci`) e, **seguindo o manual (passo 4 da seção
   0)**, configure o programa para ler esse arquivo e rode a análise.
   Nota: o programa pede confirmação ao mudar um campo; leia as mensagens.

**Você terminou quando:** a análise concluiu e você sabe qual pasta de
resultados foi criada. (Deve levar 1–2 minutos.)

## Tarefa 4 — Entender o resultado

Com a pasta da tarefa 3 (ou da tarefa 2), responda **sem ajuda**:

1. Quantas classes (tipos de amostra) havia nos dados?
2. Qual foi a acurácia balanceada?
3. O resumo traz algum aviso ou "não validado"/"n/a"? Você entendeu por
   quê? (Se não entendeu, escreva isso.)
4. Onde estaria o arquivo do modelo treinado?

## Tarefa 5 (opcional) — Versão visual

Abra a versão web (manual, seção 0 passo 4 / seção 1) e refaça a tarefa 3
por lá. Anote se foi mais fácil ou mais difícil que o terminal.

## Tarefa 6 (opcional) — Uma técnica avançada

Na mesma sessão do assistente, depois de configurar o arquivo (tarefa 3),
abra o item **[T] Técnicas Avançadas**, escolha **ASCA**, escolha o fator
`especie` e leia o resultado. Diga, com suas palavras, o que o número
"% da variância total" parece indicar. (Não é preciso acertar.)

---

Ao terminar (ou ao desistir de uma tarefa), preencha
`docs/TEMPLATE_FEEDBACK_TESTE.md` e devolva ao autor junto com **qualquer
mensagem de erro copiada exatamente como apareceu**.
