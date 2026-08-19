# Exposição de metadado no repositório público — resumo para a conversa

Preparado em 2026-08-17. **Sem ação técnica unilateral** — este documento
existe para a conversa sobre autoria e distribuição de dados, não para
justificar uma reescrita de histórico.

---

## O que está exposto

Identificadores de amostra do dataset de óleos em arquivos versionados e
já enviados ao GitHub público. São **metadados**, não espectros: código de
espécie, data de coleta e, em alguns casos, adulterante e teor.

| Arquivo | Ocorrências |
|---|---:|
| `src/guaraci/dados_io.py` | 4 |
| `src/guaraci/guaraci.py` | 2 |
| `src/guaraci/cli_assistente.py` | 2 |
| `tests/test_dados_io_jcamp.py` | 2 |
| `tests/test_heatmap_especie_adulterante.py` | 2 |
| `tests/test_dados_io_parsing.py` | 1 |

Formato do que aparece — exemplos reais do repositório remoto:

```
CAP-04-11-2099            (espécie + data de coleta)
CAP-04-11-2099-A1.03      (+ adulterante e teor)
```

Aparecem como exemplos de documentação de regex, fixtures de teste e
strings de ajuda do CLI.

## Desde quando

O commit mais antigo que os contém é **`e6384db`, de 2026-07-04**. Estão
públicos há **~6 semanas** no momento desta redação, ao longo de **80
commits** no remoto.

Ressalva importante: o histórico do repositório foi reescrito em
2026-07-11 (`338c45f`, "checkpoint pós-reescrita de histórico"). Não é
possível determinar, a partir do repositório, se identificadores
circularam em commits anteriores a essa data.

## O que NÃO está exposto

- **Nenhum espectro.** A purga de 2026-08-15 (achado S4) removeu os 48
  espectros reais do histórico, com verificação em todas as branches e
  tags. Isso continua válido.
- Os identificadores das 6 amostras com `##TITLE=` malformado, e o alias
  de `mae_id`, **não foram enviados** — saíram do código nesta rodada
  (BLOCO B) e agora vivem em `~/.guaraci_local/`, fora da árvore
  versionada. Os 11 commits locais que os continham foram reconstruídos
  antes de qualquer push.

## Por que não resolvo isso sozinho

**Reescrever o histórico remoto não despublica o que já foi indexado ou
clonado.** Um repositório público por 6 semanas pode ter sido clonado,
indexado por buscadores, arquivado por serviços de terceiros ou incluído
em datasets de treinamento. A reescrita:

- quebra clones existentes e qualquer referência a SHA;
- **não** remove cópias fora do seu controle;
- sinaliza publicamente que houve algo a esconder, o que pode ser pior que
  a exposição em si, dependendo do que se decidir sobre autoria.

Ou seja: a decisão de reescrever ou não **não é técnica**. Depende de o que
se acorde sobre titularidade e distribuição do dataset.

## Perguntas para levar à conversa

1. Metadado de amostra (espécie + data + teor de adulterante), **sem
   espectro**, é considerado dado do projeto sob titularidade compartilhada,
   ou é informação de baixo risco?
2. Se for considerado sensível: aceita-se que a remoção do histórico
   remoto é **parcial por natureza** (não alcança clones/índices), ou o
   objetivo passa a ser tornar o repositório privado?
3. Há restrição sobre a *estrutura de codificação* em si (o formato
   `COD-DD-MM-AAAA-AD-X-N%`) ficar pública, mesmo com identificadores
   sintéticos? Ela está documentada no regex de `dados_io.py` e é
   necessária para o software funcionar.
4. O que já foi decidido para o dataset vale também para os metadados
   derivados que aparecem em figuras e nomes de arquivo de saída?

## Estado atual do push

**Nenhum commit foi enviado.** Há 11 commits locais aguardando esta
conversa e a decisão de escopo. Os identificadores novos já foram
removidos deles.
