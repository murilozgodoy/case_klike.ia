import { ArrowRight, RotateCcw } from 'lucide-react'
import { useState } from 'react'
import { ErrorMessage, EmptyState, Loading } from '../../components/StatusViews'
import { BlockHead, Callout, Chips, ClassBadge, ScoreGauge } from '../../components/ui'
import { useRequest } from '../../hooks/useRequest'
import { number, signed, valueLabel } from '../../lib/format'
import { api } from '../../services/api'
import type { FieldInfo, Metadata, Report, ScenarioResult, Selection, Value } from '../../types/api'

export interface PresetEdit {
  field: string
  value: Value
}

interface Props {
  metadata: Metadata
  selection: Selection | null
  preset: PresetEdit | null
  onExplore: () => void
}

export function ScenarioLab({ metadata, selection, preset, onExplore }: Props) {
  const base = useRequest(selection && JSON.stringify(selection), () => api.analysis(selection!))

  if (!selection) {
    return (
      <EmptyState title="Nenhuma campanha selecionada">
        <p>O laboratório compara versões do criativo de uma campanha. Escolha uma na etapa 1.</p>
        <button type="button" className="button" style={{ marginTop: '0.75rem' }} onClick={onExplore}>
          Explorar campanhas <ArrowRight size={16} aria-hidden="true" />
        </button>
      </EmptyState>
    )
  }
  if (base.loading) return <Loading label="Carregando a campanha-base…" />
  if (base.error) return <ErrorMessage error={base.error} onRetry={base.retry} />
  if (!base.data) return null
  return (
    <Lab
      key={JSON.stringify(selection)}
      metadata={metadata}
      selection={selection}
      report={base.data}
      preset={preset}
    />
  )
}

function Lab({
  metadata,
  selection,
  report,
  preset,
}: {
  metadata: Metadata
  selection: Selection
  report: Report
  preset: PresetEdit | null
}) {
  const creative = metadata.campos.filter((f) => f.grupo === 'criativo')
  const suggested = report.recomendacoes.length > 0
  const [mode, setMode] = useState<'engine' | 'livre'>(suggested ? 'engine' : 'livre')
  const [field, setField] = useState<string>(preset?.field ?? creative[0].campo)
  const [text, setText] = useState<string>('')
  // A edição em comparação; uma edição vinda da análise já começa comparada.
  const [edit, setEdit] = useState<PresetEdit | null>(preset)
  const comparison = useRequest(edit && JSON.stringify(edit), () =>
    api.compare(selection, edit!.field, edit!.value),
  )
  const running = comparison.loading
  const info = creative.find((f) => f.campo === field) as FieldInfo
  const current = report.campanha[field] ?? null

  const compare = (targetField: string, value: Value) => setEdit({ field: targetField, value })
  const isActive = (targetField: string, value: Value) => edit?.field === targetField && edit.value === value

  function restore() {
    setEdit(null)
    setText('')
  }

  function freeValue(): Value | undefined {
    if (text === '') return undefined
    if (info.tipo === 'booleano') return text === 'true'
    if (info.tipo === 'numero') return Number(text.replace(',', '.'))
    return text
  }

  const options =
    info.tipo === 'booleano'
      ? ['true', 'false'].filter((o) => (o === 'true') !== current)
      : (info.opcoes ?? []).filter((o) => o !== current)

  return (
    <>
      <section className="surface pad lab-head">
        <div className="lab-head__text">
          <p className="kind">Laboratório de cenários</p>
          <h1>{report.campanha_id ?? 'Campanha nova'}</h1>
          <p className="context">
            <span className={`platform platform--${report.campanha.platform}`}>{String(report.campanha.platform)}</span>
            <span>{String(report.campanha.category)}</span>
            <span>{valueLabel('objective', report.objetivo)}</span>
            <span>vídeo {valueLabel('format', report.campanha.format).toLowerCase()}</span>
          </p>
          <p className="muted">
            Uma alteração por vez no criativo, com o mesmo contexto: o Klike Score das duas versões vem
            do mesmo preditor. O contexto da campanha não muda.
          </p>
        </div>
        <div className="score lab-head__score">
          <p className="score__value">{number(report.score_previsto)}</p>
          <p className="score__label">score previsto da versão original</p>
        </div>
      </section>

      <div className="lab-grid">
        <section className="surface pad">
          <h2 className="lab-picker__title">Escolha uma alteração</h2>
          <div className="segmented" role="tablist" aria-label="Tipo de alteração">
            <button type="button" role="tab" aria-selected={mode === 'engine'} onClick={() => setMode('engine')}>
              Sugeridas pelo engine
            </button>
            <button type="button" role="tab" aria-selected={mode === 'livre'} onClick={() => setMode('livre')}>
              Alteração livre
            </button>
          </div>

          {mode === 'engine' ? (
            <div role="tabpanel" aria-label="Sugeridas pelo engine">
              {!suggested ? (
                <p className="muted">O engine não recomenda nenhuma edição para esta campanha. Use a alteração livre.</p>
              ) : (
                <>
                  <p className="muted small lab-picker__hint">Pré-visualize a edição de uma recomendação.</p>
                  <ul className="edit-list">
                    {report.recomendacoes.map((rec) => (
                      <li key={rec.acao}>
                        <button
                          type="button"
                          className={`edit${isActive(rec.campo, rec.valor_sugerido) ? ' is-active' : ''}`}
                          aria-pressed={isActive(rec.campo, rec.valor_sugerido)}
                          disabled={running}
                          onClick={() => compare(rec.campo, rec.valor_sugerido)}
                        >
                          <span className="edit__title">
                            <span className="rank rank--soft">{rec.prioridade}</span>
                            {rec.titulo}
                          </span>
                          <ClassBadge klass={rec.classe} description={rec.classe_descricao} />
                        </button>
                      </li>
                    ))}
                  </ul>
                </>
              )}
            </div>
          ) : (
            <form
              role="tabpanel"
              aria-label="Alteração livre"
              className="free-edit"
              onSubmit={(event) => {
                event.preventDefault()
                const value = freeValue()
                if (value !== undefined) compare(field, value)
              }}
            >
              <p className="muted small">Mude um atributo do criativo e compare.</p>
              <label className="field">
                <span>Atributo do criativo</span>
                <select
                  value={field}
                  onChange={(e) => {
                    setField(e.target.value)
                    setText('')
                  }}
                >
                  {creative.map((f) => (
                    <option key={f.campo} value={f.campo}>
                      {f.rotulo}
                    </option>
                  ))}
                </select>
              </label>
              <p className="muted small">
                Valor atual: <strong>{valueLabel(field, current)}</strong>
              </p>
              <label className="field">
                <span>Valor novo</span>
                {info.tipo === 'numero' ? (
                  <input
                    type="number"
                    inputMode="decimal"
                    min={info.minimo}
                    max={info.maximo}
                    step={info.passo ?? 'any'}
                    value={text}
                    onChange={(e) => setText(e.target.value)}
                    required
                  />
                ) : (
                  <select value={text} onChange={(e) => setText(e.target.value)} required>
                    <option value="">Escolha…</option>
                    {options.map((option) => (
                      <option key={option} value={option}>
                        {info.tipo === 'booleano' ? valueLabel(field, option === 'true') : valueLabel(field, option)}
                      </option>
                    ))}
                  </select>
                )}
              </label>
              <button type="submit" className="button" disabled={running || text === ''}>
                Comparar <ArrowRight size={16} aria-hidden="true" />
              </button>
            </form>
          )}
        </section>

        <div className="lab-result">
          {!edit && (
            <EmptyState title="Nenhuma alteração em comparação">
              Escolha uma edição ao lado: as duas versões aparecem aqui, com o score previsto de cada uma e a
              evidência do engine para a mudança.
            </EmptyState>
          )}
          {running && <Loading label="Comparando as duas versões…" />}
          {comparison.error && <ErrorMessage error={comparison.error} onRetry={comparison.retry} />}
          {comparison.data && <Comparison result={comparison.data} onRestore={restore} />}
        </div>
      </div>
    </>
  )
}

const EVIDENCE_TITLES = {
  recomendada: 'Edição recomendada pelo engine',
  avaliada_nao_recomendada: 'Edição do catálogo, avaliada e não recomendada',
  fora_do_escopo: 'Edição do catálogo, fora do escopo desta métrica',
  sem_evidencia: 'Sem evidência histórica para esta alteração',
} as const

function Comparison({ result, onRestore }: { result: ScenarioResult; onRestore: () => void }) {
  const { alteracao, score, evidencia } = result
  const rec = evidencia.recomendacao
  const historical = result.original.observado !== null
  return (
    <section className="surface pad result">
      <BlockHead
        title="Comparação"
        actions={
          <button type="button" className="button button--secondary" onClick={onRestore}>
            <RotateCcw size={16} aria-hidden="true" /> Restaurar cenário original
          </button>
        }
      />
      <div className="compare">
        <div className="compare__col">
          <p className="compare__tag">Versão original</p>
          <p className="compare__value">{valueLabel(alteracao.campo, alteracao.valor_original)}</p>
          <p className="muted small">{alteracao.rotulo}</p>
          <p className="compare__score">{number(score.original)}</p>
          <p className="muted small">score previsto</p>
        </div>
        <div
          className={`compare__delta${score.diferenca < 0 ? ' compare__delta--neg' : ''}`}
          aria-label="Diferença entre as previsões"
        >
          <strong>{signed(score.diferenca)}</strong>
          <span className="small">pontos previstos</span>
        </div>
        <div className="compare__col compare__col--new">
          <p className="compare__tag">Versão modificada (hipotética)</p>
          <p className="compare__value">{valueLabel(alteracao.campo, alteracao.valor_novo)}</p>
          <p className="muted small">{alteracao.rotulo}</p>
          <p className="compare__score">{number(score.modificado)}</p>
          <p className="muted small">score previsto</p>
        </div>
      </div>
      <ScoreGauge value={score.original} modified={score.modificado} />

      <Callout title="Como ler">{result.nota}</Callout>

      <Callout tone={evidencia.situacao === 'recomendada' ? 'info' : 'warning'} title={EVIDENCE_TITLES[evidencia.situacao]}>
        {rec ? (
          <>
            <p>
              {rec.titulo} <ClassBadge klass={rec.classe} description={rec.classe_descricao} />
            </p>
            <p>{rec.impacto_estimado}</p>
            <Chips items={rec.ressalvas} />
          </>
        ) : (
          <p>{evidencia.motivo}</p>
        )}
      </Callout>

      {result.modificado.avisos.length > 0 && (
        <Callout tone="warning" title="Avisos do engine para a versão modificada">
          <ul>
            {result.modificado.avisos.map((note) => (
              <li key={note}>{note}</li>
            ))}
          </ul>
        </Callout>
      )}
      {historical && (
        <p className="muted small">
          As métricas observadas pertencem à campanha real e não valem para a versão modificada.
        </p>
      )}

      <details className="more more--inset">
        <summary>Recomendações do engine para a versão modificada</summary>
        <p className="muted">{result.modificado.resumo}</p>
        {result.modificado.principais.length > 0 && (
          <ol className="top3">
            {result.modificado.principais.map((entry) => (
              <li key={entry.acao}>
                <span className="top3__head">
                  <span className="rank rank--soft">{entry.prioridade}</span>
                  <ClassBadge klass={entry.classe} />
                </span>
                <span className="top3__title">{entry.titulo}</span>
                <span className="top3__impact">{entry.impacto_estimado}</span>
              </li>
            ))}
          </ol>
        )}
      </details>
    </section>
  )
}
