# Guia de uso · Klike Interactive Lab

Como usar a demonstração interativa do case, passo a passo. Não é preciso saber programação nem estatística. A arquitetura, a API e os testes estão no [documento técnico](interactive_lab.md).

## 1. Acessar

- **Online:** pelo link na [seção do Interactive Lab](../README.md#extra--klike-interactive-lab) do README. O servidor gratuito adormece depois de um tempo sem uso, e a primeira visita pode levar cerca de um minuto. A tela avisa enquanto espera e oferece "Tentar novamente" se a primeira chamada falhar.
- **No seu computador:** siga [Como rodar](../README.md#interactive-lab) e abra `http://localhost:5173`.

A barra lateral mostra as três etapas: **Campanhas**, **Análise** e **Laboratório**. No celular, ela vira uma barra no topo. A campanha escolhida aparece em "Campanha em análise", com o atalho "Trocar campanha". Numa campanha da base, o endereço da página guarda a etapa e a campanha, então dá para compartilhar o link de uma análise.

## 2. Escolher uma campanha

- **Indicadores:** o topo da página resume o projeto, e cada número diz de onde vem.
- **Comece por um exemplo:** três campanhas da base que mostram comportamentos diferentes do engine. "Ver análise" abre a análise da campanha.
- **Explorar campanhas:** filtre por plataforma, objetivo e categoria, ou busque pelo número da campanha. Na tabela:
  - **Observado** é o Klike Score real da campanha;
  - **Previsto** é o que o modelo estima só com o que se sabe antes da veiculação;
  - **Recomendações A · B · C** conta as recomendações de cada classe.
- Clique numa linha para ver o contexto e o criativo ao lado. "Ver análise" abre as recomendações; "Testar alterações" vai direto ao laboratório.

## 3. Ler a análise

**O topo** mostra o contexto e o criativo da campanha, o Klike Score previsto e, nas campanhas da base, o observado, numa escala de 0 a 100. Logo abaixo está a **métrica avaliada**, definida pelo objetivo: CTR em tráfego, engajamento em engajamento, retenção em awareness e conversões por mil impressões em conversões e instalação de app. Quando há algo a observar na campanha, como uma legenda sem marcação, aparece um aviso.

**O resumo executivo** diz quantas recomendações há de cada classe e destaca as três primeiras, com as ressalvas em rótulos curtos. Se houver menos de três, a tela explica por quê: o engine não completa a lista com sugestões sem base.

**As classes** dizem a força da evidência na métrica avaliada:

| Classe | Na tela | O que significa |
| :-- | :-- | :-- |
| **A** | evidência robusta | A métrica avaliada tem associação robusta e estável com a edição. |
| **B** | evidência fraca ou sensível | Há um indício, ou a associação é robusta mas muda com as escolhas de análise. |
| **C** | só hipótese de score | O score previsto sobe, mas sem evidência suficiente na métrica avaliada. |

**Recomendações em detalhe.** Cada linha mostra a prioridade, a edição, a classe, o impacto estimado na métrica e a mudança prevista no score. A ordem segue primeiro a classe, então prioridade não é o mesmo que impacto. Clique numa linha para abrir:

- **Impacto estimado:** quanto a métrica foi diferente, no histórico, entre campanhas com e sem a característica.
- **Klike Score previsto:** antes e depois da edição, com o intervalo do modelo.
- **Suporte histórico:** quantas campanhas parecidas existem para comparar.
- **Por quê e próximo passo:** a justificativa e o teste A/B sugerido.
- **Associações históricas** na métrica avaliada e nas outras, com o q-valor.
- O botão **"Testar esta edição no laboratório"**.

No fim da página, "Edições avaliadas e não recomendadas" explica o que ficou de fora e por quê, e "Como ler esta análise" resume os conceitos.

## 4. Usar o laboratório de cenários

O laboratório compara a campanha com uma versão em que um único atributo do criativo muda. O contexto (plataforma, objetivo e público) fica igual.

1. Em "Escolha uma alteração", use **Sugeridas pelo engine** para aplicar uma recomendação, ou **Alteração livre** para escolher o atributo e o valor novo e clicar em "Comparar".
2. A comparação mostra as duas versões lado a lado, o score previsto de cada uma e a diferença em pontos, na mesma escala.
3. O quadro de evidência diz em que situação a edição está:
   - "Edição recomendada pelo engine";
   - "Edição do catálogo, avaliada e não recomendada";
   - "Edição do catálogo, fora do escopo desta métrica";
   - "Sem evidência histórica para esta alteração".

   Uma alteração livre não herda a evidência de uma edição parecida.
4. Abaixo vêm os avisos do engine para a versão modificada e as recomendações que ele faria para ela.
5. "Restaurar cenário original" volta ao início.

## 5. Criar uma campanha hipotética

1. Na etapa Campanhas, clique em **Nova campanha**.
2. Preencha o contexto e o criativo. Duração, legenda e música/voz podem ficar sem valor, como no histórico. Enquanto faltar um campo obrigatório, o botão fica bloqueado e a tela lista o que falta.
3. Clique em **Analisar campanha**. A análise segue as mesmas regras de uma campanha da base, e a campanha também pode ser levada ao laboratório.

A campanha hipotética não é guardada.

## 6. O que os resultados significam

- **Score previsto não é performance.** É uma estimativa do Klike Score, com erro médio de 6,43 pontos. Entre dois criativos, diferenças previstas abaixo de 5 pontos pouco dizem sobre qual terá o score maior.
- **Associação não é causa.** O impacto estimado descreve o histórico, ajustado pelo contexto. Só um teste A/B confirma o efeito de uma edição.
- **O laboratório não é um teste A/B.** Ele compara duas previsões do mesmo modelo.
- **Suporte histórico não é efeito.** Ele diz quanta base existe para comparar, não quanto a edição rende naquele recorte.
- **As métricas observadas são da campanha real** e não valem para a versão modificada nem para uma campanha hipotética.
- **Nem toda campanha recebe três recomendações.** Isso vale para 247 das 500 (49,4%); as outras já seguem a maior parte do catálogo.
- **Os dados são simulados.** As 500 campanhas são as do desafio. O modelo e o engine rodam de verdade, mas os números ilustram o método, não um mercado real.
