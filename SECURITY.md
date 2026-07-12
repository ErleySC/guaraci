# Política de Segurança — GUARACI

## Carregar um modelo `.joblib` de terceiro é execução de código arbitrário

O Guaraci salva modelos treinados no formato `.joblib`, que por baixo usa
**pickle** — o mecanismo de serialização nativo do Python. Pickle não é
apenas dados: um arquivo `.joblib` pode conter instruções que **executam
código arbitrário no seu computador no exato momento em que o arquivo é
carregado**, antes de qualquer validação de conteúdo ser sequer possível.

Isso não é uma falha do Guaraci — é como o pickle funciona para *qualquer*
software Python que o usa (incluindo `scikit-learn`, `joblib`, `pandas`).
Mas significa que **carregar um `.joblib` de origem desconhecida equivale a
rodar um programa desconhecido** no seu computador.

### O que o Guaraci faz a respeito

1. **`guaraci.predicao.carregar_modelo(caminho, confiar=False)`** é o único
   ponto de carregamento de modelo em todo o projeto (CLI e app web). Por
   padrão, **recusa carregar** — é preciso passar `confiar=True`
   explicitamente, uma confirmação humana de que a origem é confiável.
   - Na **CLI**, isso aparece como uma pergunta de confirmação (s/n) antes
     de qualquer leitura do arquivo.
   - No **app web**, é uma caixa de seleção obrigatória ("I trust the
     source of this model file") antes do botão de predição funcionar.
2. **Manifesto de integridade** (`<modelo>.manifest.json`, gerado
   automaticamente junto de todo `modelo_plsda.joblib` exportado pelo
   pipeline): registra o hash SHA-256 do arquivo, versões de
   `guaraci`/`scikit-learn`/`numpy`/Python, timestamp e classes do modelo.
   Quando esse manifesto existe ao lado do `.joblib`, `carregar_modelo`
   **confere o hash antes de chamar `joblib.load`** — se o arquivo foi
   trocado ou corrompido depois que o manifesto foi gerado, o carregamento
   é **bloqueado antes de o pickle executar**, não apenas avisado depois.
3. **Deploy público** (Streamlit Community Cloud ou similar): o operador
   pode definir a variável de ambiente `GUARACI_DISABLE_MODEL_UPLOAD=1`
   para desabilitar completamente o upload de arquivo `.joblib` pela
   interface web, aceitando apenas caminhos locais controlados pelo próprio
   operador do servidor.

### O que isso NÃO resolve

- O manifesto detecta **arquivo trocado depois de gerado** — não valida
  automaticamente que um `.joblib` de terceiro (sem manifesto, ou com
  manifesto de origem desconhecida) é seguro. Não existe verificação
  automática de "isto é seguro" para pickle; a decisão de confiar é sempre
  humana.
- `confiar=True` não é uma prova criptográfica — é uma confirmação
  explícita de que você reconhece o risco e confia na origem (você mesmo
  treinou o modelo, ou a fonte é conhecida e verificada por outro canal).

### Recomendação para quem usa o Guaraci

- **Só carregue modelos `.joblib` que você mesmo treinou**, ou que recebeu
  de uma fonte que você confia plenamente (colega de equipe, repositório
  interno controlado).
- **Nunca** carregue um `.joblib` recebido por e-mail, link público ou
  repositório de terceiros sem verificação, mesmo que o remetente pareça
  confiável — o formato do arquivo não permite inspeção segura do conteúdo
  antes de carregar.
- Prefira compartilhar **dados de entrada** (espectros/CSV) e deixar cada
  pessoa treinar seu próprio modelo, em vez de compartilhar o `.joblib`
  pronto, quando a fonte não for inteiramente confiável.

## Relatar uma vulnerabilidade

Se você encontrar uma vulnerabilidade de segurança no Guaraci (além do
risco inerente ao pickle documentado acima, que é uma limitação conhecida
e não uma falha a ser "corrigida"), abra uma *issue* privada ou entre em
contato diretamente: erleysdacosta@gmail.com.

---
*Ver também `docs/VALIDATION.md` (validação numérica) e `docs/MANUAL.md`
(uso da aba/menu de Predição).*
