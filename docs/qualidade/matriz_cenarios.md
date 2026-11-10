# Matriz de cenários sintéticos pela CLI

**Público:** quem avalia o que a CLI cobre · **Estado:** gerado por `scripts/matriz_cenarios.py` (não editar à mão) · **Atualizado em:** ver `docs/qualidade/estado_do_projeto.md`

Cada cenário roda em subprocesso (HOME isolado, rede bloqueada) sobre dado sintético gerado por `guaraci exemplo`; nenhum dado de terceiros.

| cenario | grupo | dataset | caminho no CLI | resultado | tempo / nota |
|---|---|---|---|---|---|
| explorar_sintetico | explorar | sintetico | guaraci exemplo explorar + guaraci run {ex}/config.yaml | passou | 9.0 s; rc=0; achou resumo_modelo.txt |
| classificar_sintetico | classificar | sintetico | guaraci exemplo classificar + guaraci run {ex}/config.yaml | passou | 13.5 s; rc=0; achou resumo_modelo.txt |
| quantificar_sintetico | quantificar | sintetico | guaraci exemplo quantificar + guaraci run {ex}/config.yaml | passou | 8.6 s; rc=0; achou resumo_modelo.txt |
| multielementar_sintetico | comparar | sintetico | guaraci exemplo multielementar + guaraci run {ex}/config.yaml | passou | 14.5 s; rc=0; achou multielementar.json |
| comparar_sintetico | comparar | sintetico | guaraci exemplo comparar + guaraci anova {ex}/tabela.csv | passou | 4.0 s; rc=0; achou anova_por_resposta.csv |
| anova_nao_parametrico | comparar | sintetico | guaraci exemplo comparar + guaraci anova {ex}/tabela.csv | passou | 3.7 s; rc=0; achou anova_relatorio.md |
| pls_vip | modelar | sintetico | guaraci exemplo vip + guaraci vip {ex}/tabela.csv | passou | 3.9 s; rc=0; achou vip_estabilidade.csv |
| fluorescencia_eem | classificar | sintetico | guaraci exemplo fluorescencia_eem + guaraci run {ex}/config.yaml | passou | 12.7 s; rc=0; achou resumo_modelo.txt |
| parafac_eem | explorar | sintetico | guaraci exemplo fluorescencia_eem + teclas E fluorescencia_eem | passou | 5.2 s; rc=0; achou parafac_fator_amostra.csv |
| cromatograma_gcms | classificar | sintetico | guaraci exemplo cromatograma_gcms + guaraci run {ex}/config.yaml | passou | 12.4 s; rc=0; achou resumo_modelo.txt |
| alinhamento_cromatogramas | explorar | sintetico | guaraci exemplo cromatograma_gcms + teclas E cromatograma_gcms | passou | 22.3 s; rc=0; achou cromatogramas_alinhados.csv |
| imagens_hsi | imagens | sintetico | guaraci exemplo hsi + guaraci run {ex}/config.yaml | passou | 4.9 s; rc=0; achou hsi_relatorio.txt |
| monitorar_processos | monitorar | sintetico | guaraci exemplo monitorar | passou | 3.9 s; rc=0; achou monitor_lotes.csv |
| fusao_multibloco | modelar | sintetico | guaraci exemplo quantificar + teclas T 4 | passou | 5.1 s; rc=0; achou stdout:Fusao multibloco |
