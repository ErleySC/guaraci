# Política de Segurança — GUARACI

**Público:** todos · **Estado:** manual · **Atualizado em:** ver `docs/qualidade/estado_do_projeto.md`

*English summary.* GUARACI is maintained by one person. Report a vulnerability privately through GitHub (repository *Security* tab, *Report a vulnerability*); never in a public issue. The goal (not a guarantee) is a first reply within 7 business days. Supported versions: the latest published release and the `master` branch. Loading a `.joblib` model runs arbitrary code (it is a pickle); the web app's default (server) mode never loads one and never reads local paths. Details below, in Portuguese.

## Como relatar uma vulnerabilidade

**Canal privado:** use o relato privado de vulnerabilidade do GitHub: na página do repositório,
aba *Security* → *Report a vulnerability* (<https://github.com/ErleySC/guaraci/security/advisories/new>). O relato só é visto por quem mantém o
repositório. **Não abra uma *issue* pública com detalhes** de uma vulnerabilidade ainda não corrigida.
O projeto não publica endereço de e-mail. Dúvidas gerais (que não sejam vulnerabilidade) vão para as
GitHub Discussions, como descrito em [`SUPPORT.md`](SUPPORT.md).

**Prazo:** a meta é uma primeira resposta em até 7 dias úteis, sem garantia de prazo nem
de correção (o projeto é mantido por uma pessoa; ver [`SUPPORT.md`](SUPPORT.md)). Se possível,
inclua um passo a passo para reproduzir, a versão (`guaraci --version`) e o sistema operacional.

**Versões suportadas:** a última versão publicada e o ramo `master`. Enquanto não houver
versão publicada em um repositório de pacotes, vale só o ramo `master`.

## Modelo de ameaça: dois modos do app web

O app web (`app_quimiometria.py`) roda em um de dois modos, escolhidos por
variável de ambiente. **O padrão é o modo seguro.**

| | **Modo servidor (padrão)** | **Modo local** (`GUARACI_LOCAL_MODE=1`) |
|---|---|---|
| Quem é o visitante | Qualquer pessoa, não confiável | Só você, na sua máquina |
| Dados de entrada | Somente upload de tabela (`.csv`, `.txt` ou `.tsv`, até 50 MB) ou demo sintético | Caminhos livres, pastas, CSV, HSI |
| Onde ficam arquivos | Pasta de sessão isolada, id aleatório, **apagada ao fim da sessão** (e varrida após 6 h) | Onde você escolher |
| `config.yaml` | **Nunca lido nem gravado**; a configuração é da sessão (dá para baixar o YAML) | Lido/gravado na pasta do app |
| Modelo para predição | **Só o modelo portátil `.portatil.json`** (não executa código) | `.joblib` com manifesto assinado |
| Erros | Mensagem curta + código de erro; o traceback vai só para o log do servidor (`showErrorDetails = "none"`) | Traceback completo |

Para uso local: `guaraci web` (sobe o app do pacote instalado, liga o modo local e
prende o servidor a `127.0.0.1`) ou `GUARACI_LOCAL_MODE=1 streamlit run app_quimiometria.py`.

### Endurecimento do modo servidor

No modo servidor o app aplica, por sessão:

- **Upload:** só `.csv`, `.txt` e `.tsv`, no máximo 50 MB (`maxUploadSize` do
  `.streamlit/config.toml` e do `Dockerfile`), e a tabela é recusada antes de ser lida por
  inteiro se passar de 100 mil linhas, 50 mil colunas ou 20 milhões de células. O nome do
  arquivo é saneado (só o nome final, caracteres seguros, tamanho limitado; nomes
  reservados do Windows e vazios são recusados) e a mensagem de recusa não mostra caminho.
- **Tetos de custo:** permutações e permutações de Wold limitadas a 1000; *bootstrap* BCa e
  VIP limitados a 2000. O teto só reduz o valor pedido, e a tela avisa quando reduziu.
- **Erros:** o visitante vê só uma mensagem curta e um código; o detalhe fica no log do servidor.
- **Modelo:** só o modelo portátil `.json`; nenhum `.joblib` é carregado.
- **Chave de assinatura local** (`~/.guaraci/model_signing.key`): criada já com permissão
  restrita ao dono (`0o600` em sistemas POSIX).

**Atenção a quem hospeda.** Esses limites vêm do `.streamlit/config.toml` e do `Dockerfile` do
repositório. Quem instala o pacote e roda `streamlit run` direto num servidor **não recebe** o
`config.toml`, então o limite de upload e a ocultação de erros não valem. Em rede, use o
`Dockerfile` do repositório ou repasse essas opções na linha de comando. O comando
`guaraci web` é para uso local, na sua máquina, e não deve ser exposto na rede.

**Deploy em rede (Docker, Streamlit Cloud, etc.): não defina
`GUARACI_LOCAL_MODE`.** Com a variável ausente, nenhum campo permite apontar
para arquivos do servidor e nenhum pickle é carregado.

## Carregar um modelo `.joblib` é execução de código arbitrário

O GUARACI grava modelos treinados em `.joblib`, que usa **pickle**. Um pickle
pode conter instruções que **executam código no momento em que o arquivo é
carregado**, antes de qualquer validação de conteúdo. Isso é próprio do pickle
(vale para scikit-learn, joblib, pandas…), não uma falha do GUARACI — mas
significa que carregar um `.joblib` de origem desconhecida equivale a rodar um
programa desconhecido.

### O que o GUARACI faz

1. **`guaraci.predicao.load_model(caminho, confiar=False)`** é o único ponto de
   carregamento de `.joblib`. Recusa por padrão; `confiar=True` é a confirmação
   humana explícita (na CLI, uma pergunta s/n; no app web local, uma caixa
   obrigatória).
2. **Manifesto assinado, fail-closed.** Todo `modelo_plsda.joblib` gravado pelo
   pipeline vem com `<modelo>.manifest.json` contendo o SHA-256 do arquivo e
   uma **assinatura HMAC-SHA256 feita com uma chave secreta local**
   (`~/.guaraci/model_signing.key`, criada no primeiro treino, nunca enviada a
   lugar nenhum). `load_model` **só carrega se**: o manifesto existe, é
   legível, traz `sha256` igual ao do arquivo **e** uma assinatura válida para a
   chave desta instalação. O arquivo é lido uma única vez: os mesmos bytes cujo
   hash foi conferido são os que o `joblib` carrega (não há uma segunda leitura
   que um atacante possa trocar no intervalo). Sem
   manifesto, com manifesto ilegível, sem hash, com hash divergente ou sem
   assinatura válida: o modelo **não carrega**.
   - O hash sozinho é só *integridade*; quem consegue trocar o `.joblib` também
     consegue trocar o manifesto na mesma pasta. A assinatura é o que
     *autentica* a origem: sem a chave local, o atacante não a produz.
   - Um modelo recebido de outra máquina não carrega até você **assiná-lo de
     forma explícita**, depois de conferir a origem
     (`guaraci.predicao.assinar_modelo(caminho)`; a CLI oferece essa pergunta).
     Falha de integridade (hash divergente) **nunca** é oferecida para
     assinatura.
3. **Modelo portátil em JSON** (`modelo_plsda.portatil.json`, gravado ao lado do
   `.joblib`): é dado puro, não executa código ao ser lido. É o único formato
   aceito pelo app web em modo servidor.
4. **Web em modo servidor**: ver a tabela acima. Além disso, o rótulo de classe
   e demais dados do usuário são escapados antes de qualquer HTML, e caminhos
   absolutos são ocultados das mensagens de erro.

### O que isso NÃO resolve

- A assinatura prova que **esta instalação** liberou o arquivo, não que o
  conteúdo é seguro. Se você assinar um `.joblib` malicioso, ele executa.
- Não existe verificação automática de "isto é seguro" para pickle.
- Quem tem acesso de escrita à sua pasta pessoal (`~/.guaraci`) e à sua chave
  local pode forjar assinaturas: o modelo de ameaça é "arquivo vindo de fora",
  não "malware já rodando como você".
- O modo local **não** é para expor na rede.

### Recomendações

- Só carregue `.joblib` que você mesmo treinou ou que veio de fonte que você
  controla. Prefira compartilhar **dados** (espectros/CSV) ou o **modelo
  portátil `.json`**, e deixar cada pessoa treinar/usar o seu.
- Nunca assine um `.joblib` recebido por e-mail ou link público.
- Em deploy público, deixe `GUARACI_LOCAL_MODE` **desligado**.

---
*Ver também `docs/cientifico/validacao_numerica.md` (validação numérica) e `MANUAL.md`.*
