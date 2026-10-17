import { ChevronDown, FlaskConical } from 'lucide-react'
import { Chips, ClassBadge } from '../../components/ui'
import { integer, number, points, signed } from '../../lib/format'
import type { Association, Recommendation } from '../../types/api'

const pValue = (value: number) => (value < 0.001 ? '< 0,001' : number(value, 3))

const EVIDENCE: Record<string, string> = {
  robusto: 'robusta',
  indício: 'indício',
  inconclusivo: 'inconclusiva',
}

function AssociationRow({ a }: { a: Association }) {
  return (
    <tr>
      <td>{a.metrica_nome}</td>
      <td>
        {EVIDENCE[a.evidencia] ?? a.evidencia}
        {!a.estavel && ', sensível à análise'}
      </td>
      <td className="num">{integer(a.campanhas)}</td>
      <td className="num">{pValue(a.q_valor)}</td>
    </tr>
  )
}

interface Props {
  rec: Recommendation
  onTest?: (rec: Recommendation) => void
}

// Uma recomendação do engine: a linha resume, o detalhe abre ao clicar.
export function RecommendationCard({ rec, onTest }: Props) {
  const model = rec.diferenca_score_modelo
  const support = rec.suporte
  const associations = [...(rec.associacao ? [rec.associacao] : []), ...rec.outras_associacoes]
  return (
    <details className={`rec rec--${rec.classe}`}>
      <summary>
        <span className="rank" aria-label={`Prioridade ${rec.prioridade}`}>
          {rec.prioridade}
        </span>
        <span className="rec__main">
          <span className="rec__title">
            <h3>{rec.titulo}</h3>
            <ClassBadge klass={rec.classe} description={rec.classe_descricao} />
          </span>
          <span className="rec__impact">{rec.impacto_estimado}</span>
          <Chips items={rec.ressalvas} />
        </span>
        <span
          className={`delta${rec.diferenca_score < 0 ? ' delta--neg' : ''}`}
          title="Diferença do Klike Score previsto com esta edição"
        >
          {signed(rec.diferenca_score)} pts
        </span>
        <ChevronDown className="chevron" size={20} aria-hidden="true" />
      </summary>

      <div className="rec__body">
        <dl className="facts">
          <div>
            <dt>Impacto estimado</dt>
            <dd>{rec.impacto_estimado}</dd>
          </div>
          <div>
            <dt>Klike Score previsto</dt>
            <dd>
              {number(rec.score_antes)} → {number(rec.score_depois)}{' '}
              <strong className={rec.diferenca_score < 0 ? 'neg' : undefined}>({signed(rec.diferenca_score)})</strong>
              <span className="muted small block">
                Intervalo do modelo para esta edição: {points(model.ic_inferior)} a {points(model.ic_superior)} pontos.
                Mede a incerteza da estimativa do modelo, não a de uma campanha.
              </span>
            </dd>
          </div>
          <div>
            <dt>Suporte histórico</dt>
            <dd>
              {integer(support.campanhas_valor_atual)} campanhas {support.grupo_valor_atual} e{' '}
              {integer(support.campanhas_valor_sugerido)} {support.grupo_valor_sugerido}
              <span className="muted small block">
                Recorte: {support.recorte}
                {support.aproximacao ? `; ${support.aproximacao}` : ''}. Base de comparação, não uma estimativa do
                efeito nesse recorte.
              </span>
            </dd>
          </div>
        </dl>

        <div className="prose">
          <p>
            <strong>Por quê.</strong> {rec.justificativa}
          </p>
          {rec.nota_de_prioridade && (
            <p>
              <strong>Prioridade.</strong> {rec.nota_de_prioridade}
            </p>
          )}
          {rec.avisos.length > 0 && (
            <ul className="warnings">
              {rec.avisos.map((warning) => (
                <li key={warning}>{warning}</li>
              ))}
            </ul>
          )}
          <p>
            <strong>Próximo passo.</strong> {rec.proximo_passo}
          </p>
        </div>

        {associations.length > 0 && (
          <div className="table-wrap">
            <table className="table table--compact">
              <caption className="muted small">
                Associações históricas, ajustadas pelo contexto; a primeira é a da métrica avaliada. As variações
                estão no impacto estimado e na fundamentação. O q-valor já corrige para múltiplos testes.
              </caption>
              <thead>
                <tr>
                  <th scope="col">Métrica</th>
                  <th scope="col">Evidência</th>
                  <th scope="col" className="num">Campanhas na estimativa</th>
                  <th scope="col" className="num">q-valor</th>
                </tr>
              </thead>
              <tbody>
                {associations.map((a) => (
                  <AssociationRow key={a.metrica} a={a} />
                ))}
              </tbody>
            </table>
          </div>
        )}
        {rec.associacao?.limitacoes && (
          <p className="muted small">Limitação da métrica avaliada: {rec.associacao.limitacoes}.</p>
        )}

        {onTest && (
          <button type="button" className="button button--secondary" style={{ alignSelf: 'flex-start' }} onClick={() => onTest(rec)}>
            <FlaskConical size={16} aria-hidden="true" /> Testar esta edição no laboratório
          </button>
        )}
      </div>
    </details>
  )
}
