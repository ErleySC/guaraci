# Dependências de máquina nos testes

**Público:** quem mantém a suíte · **Estado:** gerado por `scripts/varrer_dependencias_de_maquina.py` (não editar à mão) · **Atualizado em:** ver `docs/qualidade/estado_do_projeto.md`

Todo uso de memória livre, CPU, relógio, caminho de máquina, variável de ambiente ou rede em teste precisa de uma decisão: `injetado` (valor controlado pelo teste), `bloqueado` (a dependência é desligada de propósito), `isolado` (ambiente próprio), `corrigido` (tinha dependência real e foi trocada) ou `legitimo` (não afeta o resultado). Uso novo sem decisão reprova.

| teste | dependência | estado | corrigido? | por quê |
|---|---|---|---|---|
| `tests/test_amostragem_ativa_cli.py` | ambiente | isolado | não era necessário | herda o ambiente e sobrescreve o que importa (HOME, PYTHONPATH, MPLBACKEND...) ou le uma chave opt-in |
| `tests/test_app_logic.py` | caminho | legitimo | não era necessário | confere que o upload usa a pasta temporaria do sistema, qualquer que seja ela |
| `tests/test_assistente_duvidas.py` | rede | bloqueado | não era necessário | o socket e' substituido por uma funcao que falha: prova que o assistente nao usa rede |
| `tests/test_avaliacao_modelos.py` | ram | corrigido | sim | o benchmark de regressao dependia da RAM livre do instante (< 0,6 GB pulava e o teste falhava); agora a verificacao e' injetada nos dois ramos |
| `tests/test_basetemp_curto.py` | ambiente | legitimo | não era necessário | le PYTEST_XDIST_WORKER para saber se o basetemp ja' vem com popen-gwN; nada e' alterado |
| `tests/test_basetemp_curto.py` | caminho | legitimo | não era necessário | caminhos ficticios de entrada e gettempdir substituido; prova a troca do basetemp longo no Windows |
| `tests/test_basetemp_curto.py` | relogio | legitimo | não era necessário | pausa curta so' para ordenar pastas por data e data antiga fabricada; nenhuma duracao e' afirmada |
| `tests/test_checar_design.py` | caminho | legitimo | não era necessário | texto de entrada do normalizador de instantaneo (usuario ficticio); nenhum caminho e' acessado |
| `tests/test_config_io.py` | caminho | legitimo | não era necessário | string de entrada de formatacao YAML, nao um caminho acessado |
| `tests/test_contrato_api_publica.py` | ambiente | legitimo | não era necessário | chave opt-in GUARACI_REGRAVAR_GOLDEN: regravar o golden e' decisao explicita |
| `tests/test_contrato_saida_tabular.py` | ambiente | legitimo | não era necessário | chave opt-in GUARACI_REGRAVAR_GOLDEN: regravar o golden e' decisao explicita |
| `tests/test_correcoes_locais_opt_in.py` | ambiente | isolado | não era necessário | herda o ambiente e sobrescreve o que importa (HOME, PYTHONPATH, MPLBACKEND...) ou le uma chave opt-in |
| `tests/test_doctor_instalar.py` | ram | legitimo | não era necessário | so' o NOME do modulo psutil como dado do plano de instalacao (find_spec trocado por funcao fixa); nenhuma memoria e' lida |
| `tests/test_fachada_reexport.py` | ram | legitimo | não era necessário | so' o NOME da funcao na lista de reexports; nenhuma memoria e' lida |
| `tests/test_gerar_blindagem_anova.py` | caminho | legitimo | não era necessário | texto de entrada do limpador de caminhos (usuario ficticio) |
| `tests/test_golden_valores.py` | ambiente | legitimo | não era necessário | chave opt-in GUARACI_REGRAVAR_GOLDEN: regravar o golden e' decisao explicita |
| `tests/test_graficos_motor.py` | ambiente | isolado | não era necessário | herda o ambiente e sobrescreve o que importa (HOME, PYTHONPATH, MPLBACKEND...) ou le uma chave opt-in |
| `tests/test_guaraci_cli.py` | caminho | legitimo | não era necessário | texto de entrada de um teste de truncamento de saida (caminho ficticio) |
| `tests/test_guarda_dados_de_terceiros.py` | caminho | legitimo | não era necessário | le o dossie privado de calibracao (fora do git) so' se existir, para os nomes de conjuntos nunca ficarem no repositorio; sem ele a guarda usa so' os padroes publicos |
| `tests/test_hardware.py` | cpu | injetado | não era necessário | o fallback sem psutil e' exercitado bloqueando o import; nenhum valor de CPU e' afirmado |
| `tests/test_hardware.py` | ram | injetado | não era necessário | psutil e' bloqueado ou substituido por uma memoria falsa; as fontes sem psutil (/proc/meminfo por caminho injetavel, sysconf, ctypes) sao trocadas por arquivo/funcao falsos; hardware.py le o ambiente por natureza e a chamada real so' confere as chaves |
| `tests/test_hsi_offline_prova.py` | rede | bloqueado | não era necessário | o socket e' bloqueado durante o pipeline HSI: prova de operacao offline |
| `tests/test_idioma_relatorios.py` | ambiente | injetado | não era necessário | GUARACI_IDIOMA e' fixada em pt e depois en para gerar os relatorios nos dois idiomas; o valor anterior e' restaurado |
| `tests/test_lacunas_avaliacao_modelos.py` | ram | injetado | não era necessário | _verificar_ram e' trocado por uma funcao fixa nos dois ramos (com e sem memoria) |
| `tests/test_lacunas_fechamento.py` | ram | injetado | não era necessário | a memoria e' sempre uma memoria falsa (psutil ou hardware_probe substituidos); nenhum valor da maquina e' afirmado |
| `tests/test_lacunas_guaraci_ajuda_checklist.py` | ram | injetado | não era necessário | a memoria e' sempre uma memoria falsa (psutil ou hardware_probe substituidos); nenhum valor da maquina e' afirmado |
| `tests/test_lacunas_guaraci_comandos.py` | ram | injetado | não era necessário | a memoria e' sempre uma memoria falsa (psutil ou hardware_probe substituidos); nenhum valor da maquina e' afirmado |
| `tests/test_lacunas_guaraci_predicao_hardware.py` | ram | injetado | não era necessário | a memoria e' sempre uma memoria falsa (psutil ou hardware_probe substituidos); nenhum valor da maquina e' afirmado |
| `tests/test_lacunas_guaraci_resto.py` | ram | injetado | não era necessário | a memoria e' sempre uma memoria falsa (psutil ou hardware_probe substituidos); nenhum valor da maquina e' afirmado |
| `tests/test_lacunas_modulos_pequenos_final.py` | cpu | injetado | não era necessário | os.cpu_count e' substituido por uma funcao que falha: prova o valor padrao quando a contagem nao existe |
| `tests/test_lacunas_modulos_pequenos_final.py` | ram | injetado | não era necessário | a memoria e' sempre uma memoria falsa (psutil ou hardware_probe substituidos); nenhum valor da maquina e' afirmado |
| `tests/test_lacunas_pipeline_executar.py` | ram | injetado | não era necessário | a memoria e' sempre uma memoria falsa (psutil ou hardware_probe substituidos); nenhum valor da maquina e' afirmado |
| `tests/test_lacunas_predicao.py` | ambiente | isolado | não era necessário | GUARACI_MODEL_KEY_FILE aponta para uma pasta temporaria: a chave real do usuario nunca e' tocada |
| `tests/test_lacunas_web_abas_projeto_inicio.py` | ram | injetado | não era necessário | a memoria e' sempre uma memoria falsa (psutil ou hardware_probe substituidos); nenhum valor da maquina e' afirmado |
| `tests/test_lacunas_web_modelo.py` | ram | injetado | não era necessário | a memoria e' sempre uma memoria falsa (psutil ou hardware_probe substituidos); nenhum valor da maquina e' afirmado |
| `tests/test_lacunas_web_modelo.py` | relogio | injetado | não era necessário | o tempo maximo da execucao e' reduzido a zero por constante de modulo; nenhuma duracao real e' afirmada |
| `tests/test_medir_vault.py` | ambiente | injetado | não era necessário | GUARACI_VAULT, GUARACI_VAULT_GERADOR e GUARACI_LIVRO_RAZAO apontam para um vault sintetico temporario |
| `tests/test_monitorar_cli.py` | ambiente | isolado | não era necessário | herda o ambiente e sobrescreve o que importa (HOME, PYTHONPATH, MPLBACKEND...) ou le uma chave opt-in |
| `tests/test_notebook_5_minutos.py` | ambiente | isolado | não era necessário | herda o ambiente e sobrescreve o que importa (HOME, PYTHONPATH, MPLBACKEND...) ou le uma chave opt-in |
| `tests/test_parafac_cow_relatorio_cli.py` | ambiente | isolado | não era necessário | herda o ambiente e sobrescreve o que importa (HOME, PYTHONPATH, MPLBACKEND...) ou le uma chave opt-in |
| `tests/test_perfis_objetivo_rigor.py` | ambiente | isolado | não era necessário | herda o ambiente e sobrescreve o que importa (HOME, PYTHONPATH, MPLBACKEND...) ou le uma chave opt-in |
| `tests/test_pipeline_core.py` | ram | injetado | não era necessário | o ajuste por hardware recebe um dicionario com a RAM livre; a sonda real so' confere as chaves |
| `tests/test_pipeline_import_sem_efeito_global.py` | ambiente | isolado | não era necessário | herda o ambiente e sobrescreve o que importa (HOME, PYTHONPATH, MPLBACKEND...) ou le uma chave opt-in |
| `tests/test_pulados_esperados.py` | caminho | legitimo | não era necessário | texto de exemplo de um motivo de pulo (caminho do wsl.exe); o arquivo nunca e' aberto |
| `tests/test_registrar_rodada_local.py` | caminho | legitimo | não era necessário | o 'caminho' e' texto de exemplo dentro do XML sintetico, para provar que o registro nao o guarda |
| `tests/test_revisao_cega_r5.py` | ambiente | isolado | não era necessário | herda o ambiente e sobrescreve o que importa (HOME, PYTHONPATH, MPLBACKEND...) ou le uma chave opt-in |
| `tests/test_saida_sem_caminho_da_maquina.py` | ambiente | isolado | não era necessário | herda o ambiente e sobrescreve o que importa (HOME, PYTHONPATH, MPLBACKEND...) ou le uma chave opt-in |
| `tests/test_sanitizar_saidas.py` | caminho | legitimo | não era necessário | textos de entrada do sanitizador (usuario ficticio) |
| `tests/test_simulador_espectral.py` | ambiente | legitimo | não era necessário | le e apaga (monkeypatch) GUARACI_SINTETICO_CENARIO para provar que o padrao do usuario e' 'nir' e que a suite fixa 'classico'; a variavel e' o proprio objeto do teste |
| `tests/test_suite_sem_tk.py` | ambiente | isolado | não era necessário | herda o ambiente e sobrescreve TCL_LIBRARY/TK_LIBRARY com pastas inexistentes, para quebrar o Tk de proposito |
| `tests/test_tarefas_cli.py` | ambiente | isolado | não era necessário | herda o ambiente e sobrescreve o que importa (HOME, PYTHONPATH, MPLBACKEND...) ou le uma chave opt-in |
| `tests/test_tarefas_cli.py` | relogio | corrigido | sim | o teto de 20 s por tarefa foi removido: o tempo e' medido na documentacao, nao afirmado |
| `tests/test_tempo_ate_o_menu.py` | ambiente | isolado | não era necessário | herda o ambiente e sobrescreve o que importa (HOME, PYTHONPATH, MPLBACKEND...) ou le uma chave opt-in |
| `tests/test_transferir_linearidade_robustez.py` | ambiente | isolado | não era necessário | herda o ambiente e sobrescreve o que importa (HOME, PYTHONPATH, MPLBACKEND...) ou le uma chave opt-in |
| `tests/test_verificar_hash_final.py` | ambiente | legitimo | não era necessário | GUARACI_WSL_DISTRO escolhe a distro do WSL, como no verificador de producao; o padrao e' Ubuntu |
| `tests/test_verificar_hash_final.py` | caminho | legitimo | não era necessário | conversao Windows->WSL de um caminho ficticio (usuario `alguem`) |
| `tests/test_verificar_hash_final.py` | cpu | injetado | não era necessário | os.cpu_count e' substituido por valores fixos para provar a regra 'metade dos nucleos, no maximo 4' |
| `tests/test_web_fluxo_completo.py` | ambiente | isolado | não era necessário | herda o ambiente e sobrescreve o que importa (HOME, PYTHONPATH, MPLBACKEND...) ou le uma chave opt-in |
| `tests/test_web_fluxo_completo.py` | caminho | legitimo | não era necessário | valor de campo de formulario; o arquivo nunca e' aberto |
| `tests/test_web_seguranca.py` | caminho | legitimo | não era necessário | afirma que a mensagem de erro NAO traz caminho de unidade |
| `tests/test_web_seguranca.py` | relogio | legitimo | não era necessário | fabrica um arquivo com data antiga relativa a agora; nao afirma duracao |
