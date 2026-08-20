# Agradecimentos / Acknowledgments

> **Este arquivo está incompleto de propósito.** Os espaços marcados
> `[A PREENCHER]` exigem **autorização das pessoas nomeadas** antes de
> serem preenchidos. Nomear alguém publicamente sem consentimento — mesmo
> para agradecer — não é uma cortesia, é uma decisão sobre a exposição de
> outra pessoa.
>
> *This file is deliberately incomplete. The `[A PREENCHER]` slots require
> permission from the people named. Publicly naming someone without
> consent — even to thank them — is a decision about that person's
> exposure, not a courtesy.*

---

## Por que este arquivo existe

O GUARACI é um software de autoria individual (ver `CITATION.cff` e o
histórico do repositório: 100 % dos commits de código vêm de um único
autor). Mas o
software foi **desenvolvido e depurado contra dados reais**, e esses dados
não se produzem sozinhos.

A distinção que este arquivo mantém:

| | Vai para |
|---|---|
| Quem escreveu o software | `CITATION.cff` |
| Quem forneceu amostras, preparou material ou operou instrumento | **este arquivo** |
| Quem cedeu acesso a equipamento ou infraestrutura | **este arquivo** |

Contribuição para os **dados** não é coautoria do **software** — e não
registrá-la também não é opção. A proveniência é documentada, nunca apagada:
remover o registro de que os dados vieram de determinada origem não é
limpeza, é misatribuição.

---

## Aquisição de dados

Os dados espectrais usados durante o desenvolvimento e a depuração do
GUARACI (FT-NIR de óleos vegetais fixos) foram adquiridos em instrumento
**institucional**, operado por **terceiros** — não pelo autor do software.
A janela de aquisição não é publicada aqui: é metadado do acervo, e este
arquivo não existe para descrever o acervo.

- **Operação do instrumento:** `[A PREENCHER — requer autorização]`
- **Preparo de amostras e fortificação:** `[A PREENCHER — requer autorização]`
- **Fornecimento das amostras / acervo:** `[A PREENCHER — requer autorização]`
- **Instituição detentora do equipamento:** `[A PREENCHER — decisão do autor,
  ver `DESACOPLAMENTO_2026-08-18.md`]`

Esses dados **não são distribuídos** com este software, e **nenhuma
métrica de desempenho do GUARACI é sustentada por eles**: a validação
publicada vem exclusivamente de datasets públicos — ver
[`docs/VALIDACAO_PUBLICA.md`](docs/VALIDACAO_PUBLICA.md).

> **Verificado em 2026-08-20.** Uma versão anterior desta seção afirmava
> que esses dados não produziam métrica em nenhum artefato público, e a
> medição contradizia: relatórios e scripts de auditoria versionados
> registravam contagens do acervo. Esses documentos saíram do repositório;
> as contagens que restavam em `docs/CHANGELOG.md` e no código foram
> substituídas por descrição genérica. A afirmação acima foi remedida
> depois disso, no escopo em que está escrita.

> **Antes de publicar qualquer espectro bruto**, sanitize o cabeçalho: os
> arquivos JCAMP-DX gravam operador e local em `##AUDIT TRAIL`, e essa
> informação viaja junto com o arquivo. Use
> `python scripts/sanitizar_dx.py <entrada> <saída> --conferir`.
>
> **`--conferir` não é um veredito.** Ele reporta os campos de
> proveniência que conhece e **lista para inspeção humana** toda linha
> fora da lista de rótulos esperados — um campo novo do gravador cai
> nessa lista, e nenhuma lista de proibição o pegaria. Abra alguns dos
> arquivos sanitizados à mão antes de publicar. Uma versão anterior
> deste script podia deixar operador e local para trás e ainda assim
> reportar limpo (corrigido em 2026-08-20).

---

## Datasets públicos

O GUARACI é validado contra dados publicamente disponíveis. Os autores e
mantenedores desses conjuntos tornaram possível validar este software sem
depender de dados de terceiros:

| Dataset | Fonte | Uso aqui |
|---|---|---|
| **Corn** | [Eigenvector Research](https://eigenvector.com/data/Corn/) | teste de integração permanente no CI; referência de RMSEP para regressão NIR |
| **Tecator** | StatLib / domínio público | benchmark de pré-processamento |

---

## Software

O GUARACI é construído sobre trabalho de outras pessoas. Em particular:
**NumPy**, **SciPy**, **scikit-learn**, **pandas**, **matplotlib**,
**Rich** e **Streamlit** — e, indiretamente, sobre a literatura de
quimiometria citada em `docs/VALIDATION.md` e nas docstrings dos módulos de
cálculo, cada método atribuído à publicação que o define.

---

## Como preencher

1. Pergunte a cada pessoa se ela **quer** ser nomeada, e **como** (nome
   completo, forma de citação, afiliação — a escolha é dela).
2. Se alguém preferir não ser nomeada, registre a contribuição sem o nome
   (ex.: *"aquisição espectral realizada por técnico do laboratório X"*) —
   a contribuição continua registrada, a pessoa não fica exposta.
3. Remova o aviso do topo deste arquivo quando não restar nenhum
   `[A PREENCHER]`.
