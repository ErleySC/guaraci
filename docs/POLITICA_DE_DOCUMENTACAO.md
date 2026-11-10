# Política de documentação

**Público.** Estado: manual. Esta política vale para todo arquivo versionado e para todo commit do histórico que será publicado.

## 1. O que vive onde

| destino | o que é | exemplos |
|---|---|---|
| **Público** (versionado, em `docs/{usuario,cientifico,contribuir,projeto,qualidade}`, `README*.md`, `MANUAL.md`) | o que o programa faz, como usar, o que foi medido e com que limite | manual, tutoriais, referência da CLI, métodos, validação, limitações, contribuição, segurança, licença, changelog, roteiro `v1.1+` genérico, referências, citação |
| **Privado-local** (fora do git, em `~/.guaraci_local/`) | o que é de quem mantém: negociação, decisões e pendências do autor, rascunhos, relatórios brutos de rodada | contato com instituição ou núcleo de inovação, planos e valores de licenciamento, ações de governança do GitHub, rascunho de declaração de uso de IA, livro-razão bruto, `decisoes_adotadas`, diário de passos |
| **Removido** | duplicatas, obsoletos, scripts de revisão fora do produto, texto que narra a história dos testes em vez de dizer "isto faz aquilo" | relatórios de sessão antigos, auditorias já incorporadas ao código e ao manual |

Só `docs/qualidade/` guarda saída **gerada** por script (estado do projeto, matriz técnica × tarefa, limitações medidas, "Correções registradas" em voz neutra). Um arquivo rastreado em `docs/auditoria/` fora da lista branca falha um teste.

## 2. O que nunca entra em arquivo versionado

1. Nome de pessoa ou de instituição em contexto **administrativo, de negociação, de crítica ou de erro**. Ficam a citação científica (autoria, DOI) e a atribuição que uma licença exige.
2. Frase que culpa alguém identificável. Descrever pela **causa técnica e pelo estado** ("a leitura x falhava quando y; corrigido em z").
3. Comparação com software ou grupo concorrente **sem fonte** reproduzível.
4. Alegação sem evidência reproduzível (cada número tem script e dado sintético, ou referência com DOI conferido).
5. Detalhe de infraestrutura e de segurança do aplicativo hospedado.
6. Plano de licenciamento comercial, valores, contatos.
7. Menção a pessoas do círculo do autor.
8. Dados e códigos de amostra; dado, linha, espectro ou figura de amostra individual de terceiros.
9. Caminhos locais (usuário, pastas sincronizadas, pastas de máquina).
10. E-mail, telefone, chave, token, URL interna. O projeto não publica e-mail: o contato é o das GitHub Discussions do repositório e, para vulnerabilidade, o canal privado do GitHub (`SECURITY.md`).

## 3. Como é verificado

- **Guarda genérica (na CI):** `scripts/auditar_exposicao.py` — e-mail, telefone, caminho de usuário ou de pasta sincronizada, chave ou token, URL interna e marcas de texto preparado para envio a terceiros. Imprime arquivo, linha e regra; **nunca o trecho**.
- **Guarda local (na máquina do autor):** a mesma ferramenta lê `~/.guaraci_local/termos_sensiveis.txt` (um termo por linha: pessoa, instituição, código de amostra), arquivo que **não é versionado** e que o agente cria vazio. Nenhum nome real de terceiro é escrito em lugar versionado.
- **Histórico:** `scripts/auditar_exposicao.py --historico ORIGEM..HEAD` varre o `git log -p` do intervalo a publicar. Termo da lista local ou dado de terceiros em qualquer commit do intervalo reprova a publicação.
- **Termos de matriz:** `scripts/auditar_termos_matriz.py` — o produto é por técnica e tarefa; nomes de matriz de amostra só aparecem nas referências bibliográficas e nos metadados de citação.
- **Links e comandos:** todo bloco de comando dos documentos de entrada é executado por teste (`scripts/executar_docs.py`); todo link é conferido antes e depois de qualquer movimentação.

## 4. Padrão de cada documento

Cabeçalho `Título | Público | Estado (gerado/manual) | Atualizado em <hash>`; um glossário único; os documentos de entrada em PT e EN espelhados; versão, contagens, lista de comandos e de dependências **geradas por script** (uma fonte por fato).

## 5. Movimentação

Mover com `git mv`, registrar `antigo → novo` em `docs/projeto/MAPA_DE_MIGRACAO.md` e corrigir todo link (README, manual, mkdocs, CI, testes, ajuda, vault). Nada é apagado antes de existir cópia fora do repositório.
