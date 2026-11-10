# Aviso de privacidade do demonstrativo hospedado

**Público:** quem usa o demonstrativo hospedado do GUARACI · **Estado:** rascunho, aguardando aprovação do mantenedor; ainda não é citado no README nem no aplicativo · **Atualizado em:** ver `docs/qualidade/estado_do_projeto.md`

Este aviso descreve o que o **demonstrativo hospedado** (a versão do GUARACI que roda num servidor, em modo servidor) faz com o que você envia. Ele descreve o comportamento do código; **não é uma declaração de conformidade com nenhuma lei** nem substitui orientação jurídica. Ao usar o GUARACI instalado no seu próprio computador (`guaraci web` ou a linha de comando), nada é enviado a servidor do projeto: os arquivos ficam na sua máquina.

## 1. O que o demonstrativo recebe

- Os arquivos que você envia: tabelas de texto (`.csv`, `.txt`, `.tsv`) e, na tela de predição, o modelo portátil (`.json`).
- Os parâmetros que você escolhe nas telas (técnica, pré-processamento, validação).
- O que o programa gera a partir deles: tabelas, figuras e relatórios da análise.
- Dados técnicos de conexão que a própria hospedagem registra (por exemplo, o endereço de rede de quem acessa). Isso é feito pela plataforma de hospedagem, não pelo GUARACI (ver seção 5).

O GUARACI não pede nome, e-mail nem cadastro, não usa contas de usuário e não inclui código de análise de uso ou de rastreamento. A coleta de estatísticas de uso do Streamlit está desligada na configuração do projeto.

## 2. Para que os dados são usados

Só para executar a análise que você pediu e mostrar o resultado na sua sessão. O projeto não usa o que você envia para treinar modelos, compor conjuntos de dados nem publicar nada.

## 3. Onde ficam e por quanto tempo

- **Pasta da sessão.** Cada visita recebe uma pasta própria, com nome aleatório, que guarda os seus envios e os resultados gerados. Outro visitante não tem como abrir a sua pasta pelo aplicativo.
- **Quando é apagada.** A pasta é apagada quando a sua sessão termina, isto é, quando o servidor descarta o estado da sessão (por exemplo, ao fechar a aba e a conexão expirar) ou quando o processo do servidor é encerrado normalmente.
- **Rede de segurança de 6 horas.** Sempre que uma nova sessão começa, o servidor apaga as pastas de sessão sem nenhuma modificação há mais de 6 horas. **Limitação:** essa varredura só roda quando alguém abre uma sessão nova. Se o servidor for interrompido de forma abrupta (reinício, queda) e ninguém acessar depois, a pasta pode permanecer além de 6 horas até a próxima visita ou até a hospedagem descartar o disco temporário.
- **Memória.** Enquanto a sessão está aberta, o conteúdo enviado também fica na memória do servidor.
- O programa não faz cópia de segurança do que você envia. Cópias e registros que a hospedagem eventualmente mantenha não são controlados pelo GUARACI.

## 4. Registros de erro

Quando algo falha, a tela mostra só uma mensagem curta e um código de erro de 8 caracteres. O detalhe técnico completo (a descrição do erro e o caminho dentro do código) vai para o **registro do servidor**, com o mesmo código. Esse registro pode conter trechos de mensagens de erro, como o nome de uma coluna ou de um arquivo que você enviou. Quem opera a hospedagem pode ter acesso a ele, e o tempo que ele é mantido é definido pela hospedagem, não pelo programa.

## 5. Hospedagem por terceiro

O demonstrativo roda na infraestrutura de uma empresa de hospedagem, que não é o projeto. Os servidores dela podem ficar **fora do Brasil**, e a plataforma aplica as próprias regras de privacidade e de registro, que você deve consultar no site dela. O projeto não controla essa infraestrutura.

## 6. O que você deve e não deve enviar

- **Não envie dados pessoais** (nomes de pessoas, documentos, dados de saúde ou de contato) **nem dados sigilosos, confidenciais ou sob contrato**. Use o demonstrativo com dados sintéticos ou dados que já sejam públicos.
- Limites de envio: até 50 MB por arquivo; tabelas de até 100 000 linhas, 50 000 colunas e 20 milhões de células. O que passar disso é recusado, e a análise de grande volume deve ser feita localmente.
- Para dados que não podem sair do seu computador, instale o programa e rode localmente (`docs/usuario/instalacao.md`).

## 7. Seus direitos e contato

Como os envios são apagados ao fim da sessão e o programa não guarda identificação sua, não há cadastro a corrigir ou excluir pelo projeto. Para dúvidas ou pedidos sobre este aviso: abra uma conversa nas GitHub Discussions do repositório; se o assunto não puder ser público, use o canal privado do GitHub (aba *Security* → *Report a vulnerability*).

Termos de uso do demonstrativo: `docs/usuario/termos_de_uso.md`.
