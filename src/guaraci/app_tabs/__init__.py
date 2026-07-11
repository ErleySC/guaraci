"""app_tabs — um módulo por aba do app web (item 18 da auditoria: quebrar o
monólito app_quimiometria.py por aba/serviço).

Cada submódulo expõe uma função `render(...)` chamada de dentro do
`with tab_x:` correspondente em app_quimiometria.py, recebendo como
parâmetros explícitos tudo que a aba precisa (pipeline carregado, Config
base, specs, tradução/tema, e os poucos helpers específicos que a aba usa).
Estado que atravessa abas (resultado da última execução, projeto, predição)
continua fluindo por `st.session_state`, como já acontecia antes da quebra —
nenhuma aba lê/escreve variável local de outra aba diretamente.
"""
