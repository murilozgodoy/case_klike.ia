import { ArrowRight, Target } from 'lucide-react'
import { ErrorMessage, EmptyState, Loading } from '../../components/StatusViews'
import { AttributeList, BlockHead, Callout, Chips, ClassBadge, ClassBar, ScoreGauge } from '../../components/ui'
import { useRequest } from '../../hooks/useRequest'
import { number, signed, valueLabel } from '../../lib/format'
import { api } from '../../services/api'
import type { Metadata, Recommendation, Report, Selection } from '../../types/api'
import { MethodNote } from './MethodNote'
import { RecommendationCard } from './RecommendationCard'

interface Props {
  metadata: Metadata
  selection: Selection | null
  onTest: (rec: Recommendation) => void
  onExplore: () => void
}

export function AnalysisPage({ metadata, selection, onTest, onExplore }: Props) {
  const report = useRequest(selection && JSON.stringify(selection), () => api.analysis(selection!))

  if (!selection) {
    return (
      <EmptyState title="Nenhuma campanha selecionada">
        <p>Escolha uma campanha ou um exemplo guiado na etapa 1.</p>
        <button type="button" className="button" style={{ marginTop: '0.75rem' }} onClick={onExplore}>
          Explorar campanhas <ArrowRight size={16} aria-hidden="true" />
        </button>
      </EmptyState>
    )
  }
  if (report.loading) return <Loading label="Rodando o recommendations engine…" />
  if (report.error) return <ErrorMessage error={report.error} onRetry={report.retry} />
  if (!report.data) return null
  return <ReportView metadata={metadata} report={report.data} onTest={onTest} />
}

function ReportView({ metadata, report, onTest }: { metadata: Metadata; report: Report; onTest: (rec: Recommendation) => void }) {
  const observed = report.observado?.klike_score ?? null
  const few = report.recomendacoes.length < 3
  return (
    <>
      <section className="surface pad overview-card">
        <div className="overview-card__main">
          <p className="kind">{report.campanha_id ? 'Campanha da base' : 'Campanha nova (hipotética)'}</p>
          <h1>{report.campanha_id ?? 'Campanha nova'}</h1>
          <p className="detail__label">Contexto da campanha</p>
          <AttributeList fields={metadata.campos} values={report.campanha} group="contexto" />
          <p className="detail__label">Criativo</p>
          <AttributeList fields={metadata.campos} values={report.campanha} group="criativo" />
        </div>
        <div className="score-panel">
          <div className="score-panel__values">
            <div className="score">
              <p className="score__value">{number(report.score_previsto)}</p>
              <p className="score__label">Klike Score previsto</p>
              <p className="score__sub">modelo da Parte 2, só com atributos de pré-publicação</p>
            </div>
            {observed !== null && (
              <div className="score">
                <p className="score__value">{number(observed)}</p>
                <p className="score__label">Klike Score observado</p>
                <p className="score__sub">o da campanha real, que o modelo não usa</p>
              </div>
            )}
          </div>
          <ScoreGauge value={report.score_previsto} observed={observed} />
          <p className="metric-line">
            <Target className="icon" size={18} aria-hidden="true" />
            <span>
              Métrica avaliada: <strong>{report.metrica_nome}</strong>{' '}
              {report.metrica_selecionada_manualmente ? (
                <span className="muted">
                  (escolhida manualmente; a principal de {valueLabel('objective', report.objetivo)} é{' '}
                  {metadata.metricas[report.metrica_principal_do_objetivo]})
                </span>
              ) : (
                <span className="muted">(a principal do objetivo {valueLabel('objective', report.objetivo)})</span>
              )}
            </span>
          </p>
        </div>
      </section>

      {report.avisos.length > 0 && (
        <Callout tone="warning" title="Avisos da campanha">
          <ul>
            {report.avisos.map((note) => (
              <li key={note}>{note}</li>
            ))}
          </ul>
        </Callout>
      )}

      <section className="surface pad summary">
        <BlockHead title="Resumo executivo" />
        <p className="summary__text">{report.resumo}</p>
        <ClassBar counts={report.composicao.por_classe} />
        {report.principais.length === 0 ? (
          <Callout title="Nenhuma recomendação para esta campanha">
            O catálogo não tem edição com base suficiente para este criativo, e o engine não completa a lista com
            sugestões artificiais.
          </Callout>
        ) : (
          <ol className="top3" aria-label="Principais recomendações">
            {report.principais.map((entry) => (
              <li key={entry.acao}>
                <span className="top3__head">
                  <span className="rank rank--soft">{entry.prioridade}</span>
                  <ClassBadge klass={entry.classe} />
                </span>
                <span className="top3__title">{entry.titulo}</span>
                <span className="top3__impact">
                  {entry.impacto_estimado}
                  {entry.classe !== 'C' && ` · score ${signed(entry.diferenca_score)}`}
                </span>
                <Chips items={entry.ressalvas} />
              </li>
            ))}
          </ol>
        )}
        {few && report.recomendacoes.length > 0 && (
          <Callout title="Menos de três recomendações">
            O catálogo não encontrou outras alterações com base suficiente para esta campanha. Mostrar menos é uma
            escolha: o engine não preenche a lista com sugestões sem fundamento.
          </Callout>
        )}
      </section>

      {report.recomendacoes.length > 0 && (
        <section>
          <BlockHead
            title="Recomendações em detalhe"
            description="Na ordem do engine: classe primeiro, depois o impacto na métrica avaliada. Clique para abrir a fundamentação."
          />
          <div className="recs">
            {report.recomendacoes.map((rec) => (
              <RecommendationCard key={rec.acao} rec={rec} onTest={onTest} />
            ))}
          </div>
        </section>
      )}

      {report.nao_recomendadas.length > 0 && (
        <details className="surface more">
          <summary>Edições avaliadas e não recomendadas ({report.nao_recomendadas.length})</summary>
          <ul>
            {report.nao_recomendadas.map((item) => (
              <li key={item.acao}>
                <strong>{item.acao}:</strong> {item.motivo}
              </li>
            ))}
          </ul>
        </details>
      )}

      <MethodNote note={report.nota_metodologica} classes={metadata.classes} />
    </>
  )
}
