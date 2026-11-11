# Instalação do GUARACI

**Público:** quem usa ou testa o programa · **Estado:** manual · **Atualizado em:** ver `docs/qualidade/estado_do_projeto.md`

Este guia cobre Windows, Linux e WSL. **macOS: só verificado pela CI** (a suíte de testes passa em Python 3.11 e 3.13, mas ninguém usou o programa de verdade lá; relate o que acontecer). O GUARACI precisa de **Python 3.11 a 3.13**. Ainda não há pacote publicado no PyPI (o nome `guaraci` no PyPI pertence a outro projeto, sem relação com este: não o instale): a instalação parte da pasta do projeto (o `.zip` do GitHub ou um clone).

## Requisitos mínimos

- **Memória e processador:** 4 GB de RAM e 2 núcleos bastam para o cenário típico (dezenas de amostras e centenas de variáveis); isso foi medido pelo autor em dados sintéticos, numa máquina limitada a esses recursos. O cenário típico usa bem menos que 4 GB de memória.
- **Arquivos grandes:** a leitura de uma tabela CSV de centenas de MB consome muita memória só para abrir o arquivo, antes de qualquer análise. Com 4 GB de RAM, prefira tabelas menores ou uma máquina com mais memória.
- **Tempo de instalação:** alguns minutos com boa conexão (o `pip` baixa numpy, scipy, scikit-learn, matplotlib e outras bibliotecas). O extra `.[all]` baixa bem mais que o núcleo.
- **Espaço em disco:** o núcleo ocupa bem menos que `.[all]`; reserve folga para as bibliotecas e para os resultados de cada análise (cada execução grava de poucos MB, com gráficos e tabelas). Confira o tamanho no seu ambiente com o gerenciador de arquivos ou `du -sh .venv`.
- **Conexão:** só para instalar. Depois disso o programa funciona offline.

## Windows

1. Instale o **Python 3.12** em <https://www.python.org/downloads/> e marque *"Add Python to PATH"*.
2. Baixe o GUARACI (botão **Code → Download ZIP** no GitHub), descompacte em uma pasta de nome curto (por exemplo, `C:\guaraci`) e **evite pastas sincronizadas** com a nuvem (serviços de sincronização): a sincronização pode travar arquivos durante a instalação e corromper o ambiente virtual.
3. Abra o PowerShell **dentro** dessa pasta e rode, uma linha por vez:

   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install ".[web,reports]"
   guaraci doctor --instalar
   guaraci demo
   ```

**Se o PowerShell recusar o `Activate.ps1`** (mensagem sobre "execução de scripts desabilitada"), libere só para esta janela e repita o comando de ativação:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

Isso vale apenas até fechar a janela. Outra saída é não ativar nada e chamar o programa direto: `.venv\Scripts\guaraci.exe doctor`. No Prompt de Comando (`cmd`), use `.venv\Scripts\activate.bat`.

**Se o `guaraci doctor` terminar dizendo que o Controle de Aplicativo bloqueou um arquivo**, rode o mesmo comando de novo: o Windows costuma liberar a biblioteca depois da primeira checagem. Se continuar bloqueando, o Python usado provavelmente não é o instalador oficial do python.org (por exemplo, um Python que vem embutido em outra ferramenta); recrie o ambiente com o oficial.

## Linux (Ubuntu e Debian)

No Debian e no Ubuntu o módulo `venv` vem em um pacote separado; sem ele, o `python3 -m venv` falha com a mensagem "ensurepip is not available". Instale o pacote da versão do Python que você usa (aqui, 3.12):

```bash
sudo apt update
sudo apt install python3.12 python3.12-venv
cd guaraci                      # a pasta descompactada ou clonada
python3.12 -m venv .venv
source .venv/bin/activate
pip install ".[web,reports]"
guaraci doctor --instalar
guaraci demo
```

Outras distribuições: use o pacote equivalente do seu gerenciador (o nome varia). As figuras são gravadas em arquivo sem abrir janela, então o **Tk não é necessário** para rodar o GUARACI; só instale `python3-tk` se os seus próprios scripts quiserem janelas do matplotlib.

No `zsh` (padrão em alguns sistemas), mantenha as **aspas** em `".[web,reports]"`: sem elas o shell interpreta os colchetes e o `pip` falha.

## WSL (Windows com Linux por dentro)

O WSL com Ubuntu segue exatamente os passos do Linux acima, dentro do terminal do Ubuntu. Dois cuidados:

- Trabalhe numa pasta **dentro** do Linux (por exemplo, `~/guaraci`), não em `/mnt/c/...`: o disco do Windows visto pelo WSL é bem mais lento e mistura tipos de fim de linha.
- `guaraci web` abre o servidor em `127.0.0.1`; o navegador do Windows costuma alcançá-lo pelo endereço `http://localhost:...` que o Streamlit imprime no terminal. Se não alcançar, confira esse endereço e a porta.

## O que instalar (extras)

| Comando | Para que serve |
|---|---|
| `pip install .` | Núcleo: terminal (`guaraci`), análises e gráficos. |
| `pip install ".[web]"` | Núcleo + app no navegador (`guaraci web`). |
| `pip install ".[reports]"` | Relatórios em PDF, Word, Excel e PowerPoint. |
| `pip install ".[all]"` | Tudo isso e os leitores de instrumentos opcionais (mais demorado e maior). |
| `pip install ".[opus]"`, `".[spc]"`, `".[rmn]"`, `".[hplc]"` | Só o leitor de um formato: Bruker OPUS, Thermo/Galactic SPC, pasta Bruker de RMN, pasta Agilent/Waters de HPLC/GC. |
| `pip install ".[multiway]"`, `".[imagem]"` | PARAFAC (fluorescência 3D e HSI) e textura de imagem. |

**Não sabe o que falta? Deixe o GUARACI instalar.** `guaraci doctor --instalar` confere o ambiente, mostra o que falta (navegador, relatórios e os leitores de OPUS, SPC, RMN e HPLC/GC, PARAFAC, textura de imagem e PCV disponíveis para o seu Python), pergunta se pode instalar e roda o `pip` do mesmo Python que executa o GUARACI. Com `--sim` ele não pergunta. As versões vêm do próprio pacote instalado, então funciona também a partir de um clone. Fica de fora o extra `benchmark` (`xgboost` e `shap`, grandes, só para comparar com outros algoritmos): `pip install ".[benchmark]"`. Se o `pip` falhar (sem conexão, por exemplo), o comando sai com erro e `guaraci doctor` mostra o que continua faltando. O `rainbow-api` (HPLC/GC) não tem versão para o Python 3.14 em diante; o `doctor` avisa e sugere exportar os dados para CSV.

Sem um desses extras o programa roda, mas o formato correspondente não abre: a tela `[2.4]` marca o formato com `(extra: pip install ...)` e avisa quando você o escolhe, e `guaraci doctor` e `guaraci checar` listam o que falta e quais técnicas isso afeta.

Para trabalhar no código do projeto (e não só usá-lo), veja [`CONTRIBUTING.md`](../../CONTRIBUTING.md).

## Conferir que deu certo

```bash
guaraci --version      # mostra a versão instalada
guaraci doctor         # confere Python, memória, processador e bibliotecas
guaraci demo           # roda a análise completa em dados sintéticos
```

Um `[ausente]` em um extra opcional (como `streamlit`) **não é erro**: só quer dizer que aquele recurso não foi instalado.

**Terminal de fundo claro?** A paleta foi pensada para fundo escuro. Em fundo branco, defina `GUARACI_TERMINAL=claro` antes de rodar (no PowerShell: `$env:GUARACI_TERMINAL = "claro"`; no Linux: `export GUARACI_TERMINAL=claro`).

## Quando algo dá errado

- Anote a mensagem **exatamente como apareceu**, o sistema operacional, a versão (`guaraci --version`) e o comando usado.
- Veja [`SUPPORT.md`](../../SUPPORT.md) para saber onde perguntar e como relatar o erro.
- O roteiro [`roteiro_teste_usabilidade.md`](roteiro_teste_usabilidade.md) mostra uma instalação completa, passo a passo, em um ambiente virgem.
