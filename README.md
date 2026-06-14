# RAGAS Evaluator para n8n 

Uma ferramenta em Python para avaliação automatizada de pipelines RAG (Retrieval-Augmented Generation) servidos via Webhook no n8n. Este script utiliza a API da OpenAI para calcular as métricas do RAGAS diretamente, sem a necessidade de frameworks intermediários, gerando relatórios detalhados em Excel.

## Funcionalidades

* **Avaliação "LLM-as-a-Judge":** Implementação direta das métricas RAGAS via OpenAI API (`gpt-4o-mini` e `text-embedding-3-small`).
* **Zero Framework Overhead:** Sem dependências pesadas como LangChain ou bibliotecas complexas. Apenas chamadas de API puras.
* **Fluxo em Duas Etapas (Two-Step Evaluation):** Permite coletar dados do n8n primeiro e avaliar depois, ideal para inspeção manual e debug de contextos.
* **Relatórios Excel Ricos:** Geração de planilhas formatadas (`openpyxl`) com design profissional, abas de resumo, legendas e formatação condicional baseada na pontuação.
* **Avisos de Diagnóstico:** Alertas automáticos para ausência de chunks de contexto ou quebra de idioma na resposta do agente no n8n.

##  Métricas Avaliadas

1.  **Faithfulness (Fidelidade):** A resposta gerada é ancorada no contexto recuperado ou o modelo alucinou?
    * Fórmula: $F = \frac{|V|}{|S|}$ (Verificações suportadas / Total de *statements* extraídos).
2.  **Context Relevancy (Relevância do Contexto):** Os *chunks* recuperados pelo banco de dados vetorial são úteis para responder à pergunta?
    * Fórmula: $CR = \frac{|S_{ext}|}{|S_{total}|}$ (Frases relevantes / Total de frases no contexto).
3.  **Answer Relevancy (Relevância da Resposta):** A resposta do modelo realmente responde ao que foi perguntado pelo usuário?
    * Fórmula: Medida através da média de similaridade de cosseno $AR = \text{mean}(\text{sim}(q, q_i))$ usando Embeddings.

## Pré-requisitos e Instalação

1. Clone este repositório.
2. Instale as bibliotecas Python necessárias:
   ```bash
   pip install openai requests pandas openpyxl
