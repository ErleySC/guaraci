# Roteiro de teste de usabilidade — GUARACI

**Público:** quem usa ou testa o programa · **Estado:** manual · **Atualizado em:** ver `docs/qualidade/estado_do_projeto.md`

**Leia isto primeiro.**

- **Não peça ajuda ao observador durante o teste.** Se travar, anote **onde**, **por quê** e a mensagem de erro **copiada exatamente**, e passe para a próxima tarefa. Travar **não é falha sua**: é o que este teste quer descobrir.
- Você **não precisa saber quimiometria** nem conhecer o GUARACI. O programa analisa curvas medidas em amostras (espectros) para reconhecer o tipo de amostra, estimar um teor ou comparar grupos.
- **O manual só serve para instalar (tarefa 1).** Nas tarefas 2 a 5 o programa tem de se explicar sozinho: use o que aparece na tela, o `?` e os comandos `guaraci ajuda`, `guaraci assistente` e `guaraci confianca`. Se precisar abrir o manual, **anote em qual tarefa e por quê**: isso conta como achado.
- Anote o horário de início e de fim de cada tarefa na folha de registro (no fim deste arquivo). **Não escreva seu nome**: use só o código P1, P2…
- Você precisa de: um computador com Python 3.11 a 3.13, internet para instalar e a pasta do projeto (um arquivo `.zip`). Reserve cerca de 50 minutos.

## Entrega do pacote (para quem aplica o teste)

Gere o pacote só com o que o git versiona (nenhum dado seu, nenhuma pasta de trabalho):

```bash
git archive --format=zip -o guaraci_teste.zip HEAD
```

A pessoa extrai o `.zip`, abre um terminal **dentro da pasta extraída** e segue a tarefa 1. Os comandos abaixo foram executados em um ambiente virgem (Windows 11, Python 3.12.10, ambiente virtual novo, pasta e HOME vazios) em 2026-10-06: instalação em 43 s; as tarefas 2 a 5 terminaram com código 0.

---

## Tarefa 1 — Instalar e conferir o ambiente  (tempo-alvo: 10 min)

Windows (PowerShell):

```
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install ".[reports]"
guaraci --version
guaraci doctor
```

macOS ou Linux: troque a segunda linha por `source .venv/bin/activate`.

- `[ausente]` em um extra opcional (como `streamlit`) **não é erro**: só quer dizer que aquele recurso não foi instalado.
- No Windows, se `guaraci doctor` terminar com a frase "Controle de Aplicativo bloqueou este arquivo", **rode o mesmo comando de novo**: o Windows libera a biblioteca depois da primeira checagem (medido: falhou na primeira vez, 24 s, e passou na segunda).

**Sucesso:** `guaraci --version` mostra `GUARACI v1.0.0` e você sabe dizer, em uma frase, se o `doctor` aponta algo faltando.

## Tarefa 2 — Classificar amostras  (tempo-alvo: 8 min; o programa leva cerca de 1 min)

```
guaraci exemplo classificar minha_pasta
guaraci run minha_pasta/config.yaml
```

O primeiro comando cria dados inventados e a configuração; o segundo roda a análise. Ao terminar, procure dentro de `minha_pasta/resultados` o arquivo `resumo_modelo.txt` e a figura da matriz de confusão.

**Sucesso:** você diz (a) qual foi a acurácia balanceada e onde a achou, e (b) com suas palavras, o que a matriz de confusão mostra (vale "não entendi").

## Tarefa 3 — Estimar um teor (quantificar)  (tempo-alvo: 6 min)

```
guaraci exemplo quantificar outra_pasta
guaraci run outra_pasta/config.yaml
```

**Sucesso:** você acha o erro de predição (RMSEP) no `resumo_modelo.txt` e diz o que ele quer dizer para quem vai usar o modelo.

## Tarefa 4 — Comparar grupos  (tempo-alvo: 8 min)

```
guaraci exemplo comparar terceira_pasta
guaraci anova terceira_pasta/tabela.csv --resposta teor --grupo grupo --unidade amostra --saida terceira_pasta/res --sem-bootstrap
```

Depois repita o segundo comando **sem** `--unidade amostra` e leia a resposta do programa.

**Sucesso:** você diz se os grupos diferem e entre quais pares, e explica, com suas palavras, **por que o programa recusou** o comando sem `--unidade`.

## Tarefa 5 — Pedir ajuda ao próprio programa  (tempo-alvo: 8 min; sem abrir o manual)

```
guaraci assistente "por que meu Q2 é baixo"
guaraci confianca pls_da
guaraci ajuda max_lvs
guaraci assistente "qual a capital da França"
```

**Sucesso:** sem abrir o manual, você diz (a) o que o programa recomenda fazer quando o Q2 é baixo e em que tela isso está, (b) qual é o selo do PLS-DA e uma limitação dele, (c) o que a opção `max_lvs` faz e (d) o que o assistente respondeu à pergunta fora do assunto e se isso foi útil.

## Repetição (tempo-alvo: metade do tempo da tarefa 2)

Rode a tarefa 2 de novo com `guaraci exemplo explorar quarta_pasta` e `guaraci run quarta_pasta/config.yaml`. O que se mede aqui é o quanto você aprendeu na primeira vez.

## Extras (opcionais)

Abra o menu (`guaraci`) e encontre **pelo nome**, sem manual, as telas *Perfis prontos*, *Monitorar lotes* e *Relatórios*. Se não achar em 1 minuto, aperte `?`. Para a interface visual, `pip install ".[web]"` e `guaraci web`.

---

## O que este teste mede ("curva de aprendizado curta")

Alvos **propostos**, ainda não validados por nenhuma pessoa real; o resultado do teste os confirma ou os derruba.

| medida | alvo | onde se lê |
|---|---|---|
| tarefas concluídas sozinha (S) | pelo menos 4 de 5 | coluna "Completou" |
| tempo até a primeira análise concluída (tarefa 1 + tarefa 2) | até 20 min | horários |
| consultas ao manual nas tarefas 2 a 5 | 0 | coluna "Consultou fora do programa" |
| pedidos de ajuda ao observador | 0 | coluna "Pediu ajuda" |
| facilidade percebida (1 a 7) | mediana 5 ou mais | coluna "Facilidade" |
| tempo da repetição ÷ tempo da tarefa 2 | 0,5 ou menos | horários |
| mensagem de erro sem a ação a tomar | 0 | coluna "Mensagem" |

## Folha de registro (sem dado pessoal)

Pessoa: P__   Data: ________   Sistema (Windows/Mac/Linux): ______   Já conhecia quimiometria? ( ) não ( ) um pouco ( ) sim

| Tarefa | Início | Fim | Completou (S sozinha / D com dificuldade / N não) | Consultou fora do programa (manual, busca) | Pediu ajuda | Facilidade (1 difícil – 7 fácil) | Onde travou | Mensagem de erro exata |
|---|---|---|---|---|---|---|---|---|
| 1 Instalar | | | | | | | | |
| 2 Classificar | | | | | | | | |
| 3 Quantificar | | | | | | | | |
| 4 Comparar grupos | | | | | | | | |
| 5 Pedir ajuda | | | | | | | | |
| Repetição | | | | | | | | |

Depois preencha `docs/usuario/template_feedback_teste.md` e devolva as duas folhas ao observador.
