import { useEffect, useState } from 'react'
import { ErrorMessage, Loading } from './components/StatusViews'
import { AnalysisPage } from './features/analysis/AnalysisPage'
import { ScenarioLab, type PresetEdit } from './features/lab/ScenarioLab'
import { OverviewPage } from './features/overview/OverviewPage'
import { useRequest } from './hooks/useRequest'
import { valueLabel } from './lib/format'
import { api } from './services/api'
import type { Recommendation, Selection } from './types/api'

type Tab = 'explorar' | 'analise' | 'laboratorio'

const STEPS: { id: Tab; title: string; desc: string }[] = [
  { id: 'explorar', title: 'Campanhas', desc: 'Escolha uma campanha' },
  { id: 'analise', title: 'Análise', desc: 'Recomendações do engine' },
  { id: 'laboratorio', title: 'Laboratório', desc: 'Compare uma alteração' },
]

// A aba e a campanha da base ficam na URL, para o link poder ser compartilhado.
function fromUrl(): { tab: Tab; selection: Selection | null } {
  const params = new URLSearchParams(window.location.search)
  const tab = STEPS.some((s) => s.id === params.get('aba')) ? (params.get('aba') as Tab) : 'explorar'
  const id = params.get('campanha')
  return { tab, selection: id ? { kind: 'historical', id } : null }
}

function selectionLabel(selection: Selection): { id: string; meta: string } {
  if (selection.kind === 'historical') return { id: selection.id, meta: 'campanha da base' }
  const c = selection.campaign
  return { id: 'Campanha nova', meta: `${c.platform} · ${valueLabel('objective', c.objective)}` }
}

export default function App() {
  const initial = fromUrl()
  const [tab, setTab] = useState<Tab>(initial.tab)
  const [selection, setSelection] = useState<Selection | null>(initial.selection)
  const [preset, setPreset] = useState<PresetEdit | null>(null)
  const metadata = useRequest('metadata', api.metadata)

  useEffect(() => {
    const params = new URLSearchParams()
    if (tab !== 'explorar') params.set('aba', tab)
    if (selection?.kind === 'historical') params.set('campanha', selection.id)
    const query = params.toString()
    window.history.replaceState(null, '', query ? `?${query}` : window.location.pathname)
  }, [tab, selection])

  function go(next: Tab) {
    setTab(next)
    window.scrollTo({ top: 0 })
  }

  function open(next: Selection, nextTab: 'analise' | 'laboratorio') {
    setSelection(next)
    setPreset(null)
    go(nextTab)
  }

  function test(rec: Recommendation) {
    setPreset({ field: rec.campo, value: rec.valor_sugerido })
    go('laboratorio')
  }

  const current = selection ? selectionLabel(selection) : null

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand__mark" aria-hidden="true">
            K
          </span>
          <div>
            <p className="brand__name">Klike Interactive Lab</p>
            <p className="brand__sub">Demonstração do case</p>
          </div>
        </div>

        <nav aria-label="Etapas">
          <ol className="steps">
            {STEPS.map((step, index) => (
              <li key={step.id}>
                <button
                  type="button"
                  className="step"
                  aria-current={tab === step.id ? 'page' : undefined}
                  onClick={() => go(step.id)}
                >
                  <span className="step__num" aria-hidden="true">
                    {index + 1}
                  </span>
                  <span>
                    <span className="step__title">{step.title}</span>
                    <span className="step__desc">{step.desc}</span>
                  </span>
                </button>
              </li>
            ))}
          </ol>
        </nav>

        <div className="current" aria-live="polite">
          <p className="current__label">Campanha em análise</p>
          {current ? (
            <>
              <p className="current__id">{current.id}</p>
              <p className="current__meta">{current.meta}</p>
              <button type="button" className="current__change" onClick={() => go('explorar')}>
                Trocar campanha
              </button>
            </>
          ) : (
            <p className="current__meta">Nenhuma ainda. Comece pela etapa 1.</p>
          )}
        </div>
        <p className="sidebar__note">Dados simulados do desafio. Associação não é causa.</p>
      </aside>

      <div className="main">
        <main className="page">
          {metadata.loading && <Loading label="Conectando ao servidor…" />}
          {metadata.error && <ErrorMessage error={metadata.error} onRetry={metadata.retry} />}
          {metadata.data && tab === 'explorar' && <OverviewPage metadata={metadata.data} onOpen={open} />}
          {metadata.data && tab === 'analise' && (
            <AnalysisPage metadata={metadata.data} selection={selection} onTest={test} onExplore={() => go('explorar')} />
          )}
          {metadata.data && tab === 'laboratorio' && (
            <ScenarioLab
              key={JSON.stringify([selection, preset])}
              metadata={metadata.data}
              selection={selection}
              preset={preset}
              onExplore={() => go('explorar')}
            />
          )}
        </main>
        <footer className="footer">
          Scores, classes, evidências e ressalvas vêm do pacote Python do case, sem cálculo na interface.
          Dados simulados; associação não é causa: toda recomendação é uma hipótese para teste A/B.
        </footer>
      </div>
    </div>
  )
}
