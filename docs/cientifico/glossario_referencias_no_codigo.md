# Glossário: referências que aparecem nos comentários do código

**Público:** quem lê o código · **Estado:** manual · **Atualizado em:** ver `docs/qualidade/estado_do_projeto.md`

Os comentários do GUARACI registram **por que** cada decisão foi tomada. Alguns citam de onde ela veio com um rótulo curto. Este glossário diz o que cada rótulo significa. O que vale é o conteúdo do próprio comentário; o rótulo é só o rastro histórico, e as fontes dele (diário de desenvolvimento, pareceres, relatórios de rodada) não são distribuídas com o código.

| Rótulo no código | Significado |
|---|---|
| `Passo N` (ex.: "Passo 213") | Etapa numerada do diário de desenvolvimento do mantenedor |
| `Bloco N` (ex.: "Bloco 11") | Bloco de funcionalidade de uma rodada de trabalho (ex.: Bloco 11 = auditoria de delineamento) |
| `M-xx`, `m-xx`, `B-x`, `S-x` | Achado de uma revisão por pares (maior, menor, bloqueante, sugestão) |
| `R2-xx`, `R3-xx`, `R5-xx` | Achado de um revisor específico (funcional, metodologia, usabilidade) |
| `CLAUDE.md seção X`, `CLAUDE.md P1` | Notas internas de roadmap do mantenedor (P1..P9), fora do repositório |
| `achado #N`, "auditoria de AAAA-MM-DD" | Rodada de auditoria datada |
| `~/.guaraci_local/` | Pasta local do mantenedor; só é usada com a opção `aplicar_correcoes_locais: true` (ver `MANUAL.md`, correções locais) |
| `_etapa_*` em `pipeline.py` | Etapas do pipeline extraídas de `_executar_impl` (ver `docs/contribuir/arquitetura.md`) |

Regra para código novo: explique a decisão no próprio comentário. Se citar um rótulo, use uma das formas acima.
