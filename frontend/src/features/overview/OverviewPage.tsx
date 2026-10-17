import { ArrowRight, BookOpen, ChartColumn, ChevronRight, Compass, Database, Plus, X } from 'lucide-react'
import { useState } from 'react'
import { BlockHead } from '../../components/ui'
import { integer, number, percent } from '../../lib/format'
import type { Metadata, Selection } from '../../types/api'
import { CampaignExplorer } from './CampaignExplorer'
import { NewCampaignForm } from './NewCampaignForm'

interface Props {
  metadata: Metadata
  onOpen: (selection: Selection, tab: 'analise' | 'laboratorio') => void
}

const PARTS = [
  { icon: Database, title: 'Parte 1 · Dados e EDA', text: 'Associações entre o criativo e a performance, com correção para múltiplos testes.' },
  { icon: ChartColumn, title: 'Parte 2 · Modelo', text: 'Uma Ridge prevê o Klike Score antes da veiculação, com 13 atributos.' },
  { icon: Compass, title: 'Parte 3 · Engine', text: 'Score previsto, associação e força da evidência viram recomendações.' },
  { icon: BookOpen, title: 'Parte 4 · Produto', text: 'Como levar isso a produção, com vídeo e testes A/B reais.' },
]

export function OverviewPage({ metadata, onOpen }: Props) {
  const [creating, setCreating] = useState(false)
  const { modelo, avaliacao_engine: engine } = metadata.indicadores

  return (
    <>
      <section className="intro">
        <div className="intro__text">
          <span className="badge-sim">Dados simulados, fornecidos pelo desafio</span>
          <h1>
            Do criativo à recomendação, <span className="highlight">com a evidência à vista</span>
          </h1>
          <p className="lead">
            Explore as 500 campanhas da base, veja o que o recommendations engine sugere para cada uma
            e teste como o Klike Score previsto muda quando um atributo do criativo muda. Tudo aqui vem
            do mesmo código Python do case.
          </p>
          <ol className="flow" aria-label="Como usar">
            <li>
              <span className="flow__n">1</span>Selecione uma campanha
            </li>
            <li aria-hidden="true">
              <ChevronRight className="icon" size={16} />
            </li>
            <li>
              <span className="flow__n">2</span>Explore as recomendações
            </li>
            <li aria-hidden="true">
              <ChevronRight className="icon" size={16} />
            </li>
            <li>
              <span className="flow__n">3</span>Teste uma alteração
            </li>
          </ol>
        </div>

        <div className="kpis">
          <article className="kpi">
            <p className="kpi__value">{integer(metadata.indicadores.campanhas)}</p>
            <p className="kpi__label">campanhas de vídeo</p>
            <p className="kpi__source">Base do desafio · Meta, TikTok e LinkedIn</p>
          </article>
          <article className="kpi">
            <p className="kpi__value">{number(modelo.mae, 2)} pts</p>
            <p className="kpi__label">
              erro médio do score previsto (de {number(modelo.mae_ic95[0], 2)} a {number(modelo.mae_ic95[1], 2)})
            </p>
            <p className="kpi__source">Parte 2 · certificação no teste</p>
          </article>
          <article className="kpi">
            <p className="kpi__value">{integer(engine.recomendacoes)}</p>
            <p className="kpi__label">
              recomendações: {integer(engine.classes.A)} A, {integer(engine.classes.B)} B e {integer(engine.classes.C)} C
            </p>
            <p className="kpi__source">Parte 3 · avaliação funcional, não mede impacto</p>
          </article>
          <article className="kpi">
            <p className="kpi__value">{percent(engine.parcela_tres_ou_mais)}</p>
            <p className="kpi__label">
              com três ou mais sugestões; {integer(engine.campanhas_tres_ou_mais_com_evidencia)} com três A ou B,{' '}
              {integer(engine.campanhas_sem_recomendacao)} sem nenhuma
            </p>
            <p className="kpi__source">Parte 3 · cobertura do catálogo atual</p>
          </article>
        </div>
      </section>

      <ol className="parts" aria-label="Como as partes se conectam">
        {PARTS.map(({ icon: Icon, title, text }) => (
          <li key={title}>
            <strong>
              <Icon className="icon" size={16} aria-hidden="true" />
              {title}
            </strong>
            {text}
          </li>
        ))}
      </ol>

      <section>
        <BlockHead title="Comece por um exemplo" description="Campanhas reais da base que mostram comportamentos diferentes do engine." />
        <div className="examples">
          {metadata.exemplos.map((example) => (
            <button
              key={example.campaign_id}
              type="button"
              className="example"
              onClick={() => onOpen({ kind: 'historical', id: example.campaign_id }, 'analise')}
            >
              <span className="example__top">
                <span className="example__id">{example.campaign_id}</span>
              </span>
              <span className="example__title">{example.titulo}</span>
              <span className="example__desc">{example.descricao}</span>
              <span className="example__cta">
                Ver análise <ArrowRight size={16} aria-hidden="true" />
              </span>
            </button>
          ))}
        </div>
      </section>

      <section>
        <BlockHead
          title="Explorar campanhas"
          description="Filtre a base e abra uma campanha, ou crie uma campanha hipotética."
          actions={
            <button type="button" className={creating ? 'button button--secondary' : 'button'} onClick={() => setCreating((v) => !v)}>
              {creating ? <X size={16} aria-hidden="true" /> : <Plus size={16} aria-hidden="true" />}
              {creating ? 'Fechar formulário' : 'Nova campanha'}
            </button>
          }
        />
        {creating && (
          <NewCampaignForm fields={metadata.campos} onSubmit={(campaign) => onOpen({ kind: 'new', campaign }, 'analise')} />
        )}
        <CampaignExplorer metadata={metadata} onOpen={onOpen} />
      </section>
    </>
  )
}
