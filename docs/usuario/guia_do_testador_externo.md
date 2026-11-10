# Guia para quem vai testar o GUARACI (teste externo)

**Público:** quem usa ou testa o programa · **Estado:** manual · **Atualizado em:** ver `docs/qualidade/estado_do_projeto.md`

Este guia é para a pessoa que **usa** o GUARACI pela primeira vez, com os
próprios dados. (Quem observa um teste de usabilidade usa
[`expectativas_teste_usabilidade.md`](expectativas_teste_usabilidade.md).)

## 1. O que você precisa ter

- **Computador** com Windows, Linux ou macOS; 8 GB de RAM recomendados
  (o programa reduz sozinho alguns extras com menos memória e avisa).
- **Python 3.11, 3.12 ou 3.13.** Com Python 3.14 tudo funciona, exceto
  a leitura de cromatogramas de fabricante (extra `[hplc]`).
- **Seus dados**, em um destes formatos:
  - uma pasta por classe com arquivos JCAMP-DX (`.dx`, `.jdx`); ou
  - um CSV com uma linha por espectro, as colunas espectrais e **uma coluna
    de classe** (por exemplo `classe`).
- **Recomendado:** uma coluna (ou arquivo) que diga qual espectro veio de qual
  **amostra física**. Sem ela, réplicas da mesma amostra podem cair dos dois
  lados da validação e os números saem otimistas — o programa avisa.
- **Mínimo útil:** pelo menos 2 amostras físicas (de preferência 5 ou mais)
  por classe. Com menos, a validação cruzada é recusada com explicação.

Você **não** precisa enviar seus dados a ninguém: tudo roda no seu computador.

## 2. Tempo esperado

Medido em 2026-10-02, numa instalação limpa (Linux, 16 núcleos lógicos, 7,6 GB):

| Etapa | Tempo medido |
|---|---|
| Instalar (`pip install` com extras `web,reports,benchmark`) | ~2 min |
| `guaraci doctor` (diagnóstico do ambiente) | segundos |
| `guaraci run` com o exemplo (`config.example.yaml`, dados sintéticos) | ~3 min |
| `guaraci demo` (fluxo completo com dados sintéticos) | ~14 min |
| Seu dataset | depende do tamanho: alguns milhares de espectros podem levar dezenas de minutos por análise |

Em computadores mais modestos, conte com o dobro.

## 3. Roteiro sugerido (cerca de 1 hora)

1. Instale (ver `README.md`, seção de instalação) e rode `guaraci doctor`.
2. Rode `guaraci demo` e abra `Relatorios/resumo_modelo.txt` e a pasta `Graficos/`.
3. Copie `config.example.yaml` para `config.yaml`, aponte para os seus dados
   (`modo_entrada`, `pasta_dados` ou `arquivo_csv`, `coluna_classe`) e rode
   `guaraci run config.yaml`. Ou use o assistente: `guaraci` (sem argumentos).
4. Leia o **"Leia primeiro"** do model card antes dos números.

## 4. O que é esperado (e não é defeito)

- Avisos de **"n insuficiente"**, **"cobertura não validada"** ou
  **"confundimento classe × sessão"**: o programa está dizendo o que os seus
  dados permitem afirmar. Não é erro.
- Técnica opcional **"indisponível"** (por exemplo XGBoost ou SHAP bloqueados
  pelo sistema): o resto da análise continua; a mensagem diz o que fazer.
- Os números do **holdout externo** são os que valem como desempenho fora da
  amostra; os de validação cruzada podem ser otimistas.

## 5. O que reportar

Para cada problema ou dúvida:

1. o que você tentou fazer e o comando (ou tela) usado;
2. o que aconteceu e o que você esperava;
3. a saída de `guaraci doctor` (arquivo `guaraci_doctor.txt`);
4. se houver erro, as últimas ~30 linhas do terminal;
5. se puder, o `resumo_modelo.txt` — **apague dele nomes de amostra ou de
   arquivo que você não queira tornar públicos**.

Também vale reportar: textos confusos, passos que você não entendeu e o tempo
que cada etapa levou.

**Não envie** seus espectros, nomes de amostras ou caminhos da sua máquina.

## 6. Onde reportar

- **Issues do repositório no GitHub** (`ErleySC/guaraci`), usando o modelo de
  relato de erro — é público; não inclua dados.
- Para o que não puder ser público: o canal privado do GitHub descrito no `SECURITY.md` (vulnerabilidade) ou as GitHub Discussions do repositório (sem dados).
