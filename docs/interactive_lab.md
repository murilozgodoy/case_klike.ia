# Klike Interactive Lab

Uma demonstração interativa do case, opcional: uma aplicação web em que um avaliador explora as 500 campanhas, lê as recomendações do engine e compara versões de um criativo, sem abrir notebooks nem instalar Python.

Ela não refaz nenhuma análise. O backend é uma camada fina sobre o pacote `klike`:
- o Klike Score vem do `KlikeScorePredictor`;
- as recomendações, classes, evidências, suporte e ressalvas vêm do `RecommendationEngine`, no formato de `RecommendationReport.to_dict()`;
- a limpeza dos dados é a da Parte 1;
- os indicadores da página inicial vêm da certificação da Parte 2 e da avaliação do engine em `reports/engine_recommendations.csv`.

O frontend só formata e apresenta. Nenhuma estatística é calculada nele.

## O que a aplicação oferece

1. **Campanhas.** Indicadores do projeto, cada um rotulado com a origem; como as quatro partes se conectam; três exemplos guiados; um explorador com filtros por plataforma, objetivo e categoria, busca por ID e paginação; e um formulário para uma campanha nova, que não é guardada.
2. **Análise e recomendações.**
   - Contexto e criativo da campanha.
   - O score previsto, separado do observado nas campanhas da base.
   - A métrica avaliada e os avisos da campanha.
   - O resumo executivo do engine.
   - Cada recomendação com classe, impacto, diferença de score, intervalo do modelo, suporte histórico, ressalvas e, em detalhe, a fundamentação e o próximo passo.
   - Campanhas com zero, uma ou duas recomendações são mostradas como estão, com a explicação.
3. **Laboratório de cenários.**
   - Pré-visualiza uma edição recomendada ou altera livremente um único atributo do criativo.
   - Mostra as duas versões, o score previsto de cada uma, pelo mesmo preditor, e a diferença.
   - Diz se a edição é uma ação do catálogo com evidência. Uma edição livre não herda a evidência de uma ação parecida.
   - Traz os avisos do engine para a versão modificada.
   - O botão de restaurar volta à versão original.

O que a aplicação deixa explícito em cada tela:
- score previsto não é performance;
- associação não é causa;
- suporte histórico não é efeito no recorte;
- a comparação de cenários não é um teste A/B;
- métricas observadas pertencem só à campanha real.

## Arquitetura

```
frontend/   React + TypeScript + Vite (Vercel)
  src/features/   overview (campanhas e formulário), analysis (relatório), lab (cenários)
  src/services/   cliente HTTP com Fetch, tempo limite e aviso de servidor despertando
  src/types/      contratos da API
backend/    FastAPI (Render)
  app/core/       configuração por ambiente e recursos carregados uma vez por processo
  app/services/   campanhas, análise e cenários, chamando o pacote klike
  app/schemas/    contratos Pydantic
  app/api/routes/ rotas enxutas
  tests/          equivalência com o pacote e regras da API
render.yaml       blueprint do serviço no Render
```

É uma arquitetura enxuta por escolha: uma demonstração concentrada na experiência do avaliador e na reutilização do código do case. Ela não tem banco de dados, autenticação nem estado entre sessões, porque nada disso é necessário. O modelo é gerado no build, pela especificação congelada da Parte 2, e carregado uma única vez quando a API inicia. Nenhuma requisição treina modelo ou recalcula evidências.

## API

| Método e rota | O que faz |
| :-- | :-- |
| `GET /api/health` | Estado do serviço e origem do modelo carregado |
| `GET /api/metadata` | Campos e opções lidos do pacote, métricas, classes, indicadores e exemplos guiados |
| `GET /api/campaigns` | Listagem com `platform`, `objective`, `category`, `busca`, `pagina` e `por_pagina` |
| `GET /api/campaigns/{id}` | Atributos de pré-publicação, métricas observadas e o resumo da campanha |
| `GET /api/campaigns/{id}/analysis` | O relatório do engine para a campanha da base |
| `POST /api/analyze` | O relatório do engine para uma campanha nova (os 13 atributos) |
| `POST /api/scenarios/compare` | `campaign_id` ou `campaign`, e `changes` com exatamente um atributo do criativo |

**Erros.**
- A entrada inválida responde 422, com a mensagem do pacote ou da validação, em português.
- Uma campanha inexistente responde 404.
- Um erro interno responde 500 com uma mensagem genérica, sem detalhes internos.
- Valores ausentes saem como `null`, nunca `NaN`.

A documentação interativa fica em `/docs` quando a API está rodando.

## Rodar localmente

Na raiz do repositório, com o ambiente do case já criado:

```bash
pip install -r backend/requirements-dev.txt
python -m klike.train                                      # gera models/, se ainda não existir
uvicorn app.main:app --app-dir backend --reload             # API em http://localhost:8000
pytest backend/tests                                       # testes da API
```

Em outro terminal:

```bash
cd frontend
npm install
npm run dev                                                # http://localhost:5173
```

## Deploy

**Backend no Render (Web Service, plano gratuito).**
1. Em *New → Blueprint*, aponte para o repositório: o `render.yaml` da raiz define o serviço.
   - Se preferir criar à mão, use a raiz do repositório como diretório.
   - O build é `pip install -r backend/requirements.txt && pip install --no-deps -e . && python -m klike.train`.
   - O start é `uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port $PORT`.
   - O health check é `/api/health`, e `PYTHON_VERSION` deve ser `3.13.3`.
2. Defina `CORS_ORIGINS` com a URL do frontend na Vercel. Opcionalmente, defina `CORS_ORIGIN_REGEX` para as URLs de pré-visualização.
3. Confira `https://<serviço>.onrender.com/api/health`.

**Frontend na Vercel.**
1. Importe o repositório e defina *Root Directory* como `frontend`. A Vercel reconhece o Vite: o build é `npm run build`, e a saída, `dist`.
2. Defina `VITE_API_URL` com a URL do serviço no Render, sem barra no fim.
3. Depois do primeiro deploy, volte ao Render e ajuste `CORS_ORIGINS` com a URL final.

**Cold start.** No plano gratuito, o serviço adormece depois de um tempo sem uso e leva cerca de um minuto para voltar. A interface avisa isso enquanto espera e oferece "Tentar novamente" se a primeira chamada falhar.

**Medidas locais.** A API fica pronta em cerca de 5 a 8 segundos e ocupa cerca de 200 MB de memória, dentro dos 512 MB do plano gratuito. Uma análise leva cerca de 0,1 s, e uma comparação de cenários, cerca de 0,2 s.

## Validação

- **Backend:** 33 testes. As respostas da API são comparadas com um engine montado direto do pacote, com a especificação congelada treinada na hora.
  - **Relatórios:** iguais para KLK-0024, KLK-0291 e campanhas novas.
  - **Scores do cenário:** iguais aos do preditor.
  - **Entradas inválidas e ausentes:** campos obrigatórios ausentes, categorias inexistentes, valores fora do intervalo e campos opcionais ausentes.
  - **Música/voz:** a recomendação aparece com o score em queda.
  - **Combinação sem histórico:** a análise vem com o aviso de extrapolação.
  - **Regras do cenário:** recusa duas alterações e alterações de contexto, e não leva métricas observadas para a versão hipotética.
  - **JSON e erros:** JSON estrito, sem `NaN`, e erros internos sem detalhes.
- **Frontend:** 12 testes, com respostas reais da API como fixtures.
  - O cliente HTTP e o aviso de servidor indisponível.
  - A renderização das recomendações e do caso sem recomendação.
  - O formulário de campanha nova.
  - A comparação e a restauração no laboratório.
  - O build, o oxlint e o TypeScript passam.

## Limitações

- **Base simulada:** os dados são os do desafio, simulados, e as recomendações herdam as limitações documentadas nas Partes 1 a 3.
- **Laboratório:** compara previsões do modelo de score, uma alteração por vez. Não estima o efeito de várias edições juntas nem o efeito causal de nenhuma.
- **Sem persistência:** campanhas novas e cenários não são guardados.
- **Cold start:** a primeira visita depois de um período parado espera o serviço gratuito despertar.
