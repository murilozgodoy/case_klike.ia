import { Info, TriangleAlert } from 'lucide-react'
import type { ReactNode } from 'react'
import { integer, number, valueLabel } from '../lib/format'
import type { CampaignAttributes, FieldInfo, RecommendationClass } from '../types/api'

const CLASS_SHORT: Record<RecommendationClass, string> = {
  A: 'evidência robusta',
  B: 'evidência fraca ou sensível',
  C: 'só hipótese de score',
}
const CLASS_COLORS: Record<RecommendationClass, string> = {
  A: 'var(--class-a)',
  B: 'var(--class-b)',
  C: 'var(--class-c)',
}

// A classe resume a força da evidência histórica; nenhuma delas é garantia de resultado.
export function ClassBadge({ klass, description }: { klass: RecommendationClass; description?: string }) {
  return (
    <span className={`badge badge--${klass}`} title={description}>
      Classe {klass}
      <span className="badge__hint"> · {CLASS_SHORT[klass]}</span>
    </span>
  )
}

export function Callout({
  tone = 'info',
  title,
  children,
}: {
  tone?: 'info' | 'warning'
  title?: string
  children: ReactNode
}) {
  const Icon = tone === 'warning' ? TriangleAlert : Info
  return (
    <div className={`callout callout--${tone}`}>
      <Icon className="icon" size={18} aria-hidden="true" />
      <div>
        {title && <p className="callout__title">{title}</p>}
        <div>{children}</div>
      </div>
    </div>
  )
}

// Ressalvas curtas do engine; a ressalva geral fica neutra, as demais em destaque.
export function Chips({ items }: { items: string[] }) {
  return (
    <ul className="chips">
      {items.map((item) => (
        <li key={item} className={item.startsWith('associação, não causa') ? 'chip' : 'chip chip--warn'}>
          {item}
        </li>
      ))}
    </ul>
  )
}

export function AttributeList({
  fields,
  values,
  group,
}: {
  fields: FieldInfo[]
  values: CampaignAttributes
  group: 'contexto' | 'criativo'
}) {
  return (
    <dl className="attributes">
      {fields
        .filter((field) => field.grupo === group)
        .map((field) => (
          <div key={field.campo} className="attributes__item">
            <dt>{field.rotulo}</dt>
            <dd>{valueLabel(field.campo, values[field.campo] ?? null)}</dd>
          </div>
        ))}
    </dl>
  )
}

export function PageHead({ title, lead, children }: { title: string; lead?: ReactNode; children?: ReactNode }) {
  return (
    <header className="page-head">
      <div className="page-head__text">
        <h1>{title}</h1>
        {lead && <p className="lead">{lead}</p>}
      </div>
      {children}
    </header>
  )
}

export function BlockHead({ title, description, actions }: { title: string; description?: ReactNode; actions?: ReactNode }) {
  return (
    <div className="block-head">
      <div>
        <h2>{title}</h2>
        {description && <p className="muted small">{description}</p>}
      </div>
      {actions}
    </div>
  )
}

// Escala de 0 a 100 com marcadores nos valores recebidos da API; não calcula nada.
export function ScoreGauge({
  value,
  observed,
  modified,
}: {
  value: number
  observed?: number | null
  modified?: number
}) {
  const at = (x: number) => `${Math.min(100, Math.max(0, x))}%`
  return (
    <div>
      <div className="gauge" aria-hidden="true">
        <div className="gauge__track" />
        {modified === undefined && <div className="gauge__fill" style={{ width: at(value) }} />}
        <div className="gauge__marker" style={{ left: at(value) }} />
        {observed !== undefined && observed !== null && (
          <div className="gauge__marker gauge__marker--observed" style={{ left: at(observed) }} />
        )}
        {modified !== undefined && <div className="gauge__marker gauge__marker--modified" style={{ left: at(modified) }} />}
        <div className="gauge__scale">
          <span>0</span>
          <span>50</span>
          <span>100</span>
        </div>
      </div>
      <p className="legend">
        <span>
          <i style={{ background: 'var(--ink)' }} />
          {modified === undefined ? 'previsto' : `original, ${number(value)}`}
        </span>
        {observed !== undefined && observed !== null && (
          <span>
            <i style={{ background: 'var(--accent-bright)' }} />
            observado
          </span>
        )}
        {modified !== undefined && (
          <span>
            <i style={{ background: 'var(--accent-bright)' }} />
            modificado, {number(modified)}
          </span>
        )}
      </p>
    </div>
  )
}

// Composição da lista por classe, com as contagens do próprio relatório.
export function ClassBar({ counts }: { counts: Record<RecommendationClass, number> }) {
  const total = counts.A + counts.B + counts.C
  if (total === 0) return null
  const classes: RecommendationClass[] = ['A', 'B', 'C']
  return (
    <div>
      <div className="classbar" aria-hidden="true">
        {classes.map((k) =>
          counts[k] > 0 ? <span key={k} style={{ width: `${(counts[k] / total) * 100}%`, background: CLASS_COLORS[k] }} /> : null,
        )}
      </div>
      <p className="legend" style={{ marginTop: '0.45rem' }}>
        {classes.map((k) => (
          <span key={k}>
            <i style={{ background: CLASS_COLORS[k] }} />
            {integer(counts[k])} de classe {k}
          </span>
        ))}
      </p>
    </div>
  )
}
