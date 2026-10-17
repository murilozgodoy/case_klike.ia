import { ArrowRight, FlaskConical, Search } from 'lucide-react'
import { useState } from 'react'
import { ErrorMessage, EmptyState, Loading } from '../../components/StatusViews'
import { AttributeList } from '../../components/ui'
import { useRequest } from '../../hooks/useRequest'
import { integer, number, valueLabel } from '../../lib/format'
import { api, type CampaignFilters } from '../../services/api'
import type { CampaignSummary, Metadata, Selection } from '../../types/api'

interface Props {
  metadata: Metadata
  onOpen: (selection: Selection, tab: 'analise' | 'laboratorio') => void
}

const FILTERS = [
  { key: 'platform', label: 'Plataforma' },
  { key: 'objective', label: 'Objetivo' },
  { key: 'category', label: 'Categoria' },
] as const

function Counts({ campaign }: { campaign: CampaignSummary }) {
  if (campaign.recomendacoes === 0) return <span className="muted">nenhuma</span>
  return (
    <span className="counts" aria-label={`${campaign.A} A, ${campaign.B} B e ${campaign.C} C`}>
      {(['A', 'B', 'C'] as const).map((k) => (
        <span key={k} className={`count count--${k}${campaign[k] === 0 ? ' count--zero' : ''}`}>
          {campaign[k]}
        </span>
      ))}
    </span>
  )
}

export function CampaignExplorer({ metadata, onOpen }: Props) {
  const [filters, setFilters] = useState<CampaignFilters>({ pagina: 1, por_pagina: 12 })
  const [selected, setSelected] = useState<string | null>(null)
  const list = useRequest(JSON.stringify(filters), () => api.campaigns(filters))
  const detail = useRequest(selected, () => api.campaign(selected!))
  const options = (field: string) => metadata.campos.find((f) => f.campo === field)?.opcoes ?? []
  const update = (change: CampaignFilters) => setFilters((current) => ({ ...current, ...change, pagina: 1 }))

  const page = list.data
  const pages = page ? Math.max(1, Math.ceil(page.total / page.por_pagina)) : 1

  return (
    <>
      <div className="filters" role="search">
        {FILTERS.map(({ key, label }) => (
          <label key={key} className="field">
            <span>{label}</span>
            <select value={filters[key] ?? ''} onChange={(e) => update({ [key]: e.target.value || undefined })}>
              <option value="">Todas</option>
              {options(key).map((option) => (
                <option key={option} value={option}>
                  {valueLabel(key, option)}
                </option>
              ))}
            </select>
          </label>
        ))}
        <label className="field">
          <span>Buscar por ID</span>
          <span className="input-icon">
            <Search className="icon" size={16} aria-hidden="true" />
            <input
              type="search"
              placeholder="ex.: 0024"
              value={filters.busca ?? ''}
              onChange={(e) => update({ busca: e.target.value || undefined })}
            />
          </span>
        </label>
      </div>

      <div className="explorer">
        <div className="surface">
          {list.loading && <Loading label="Carregando campanhas…" />}
          {list.error && <ErrorMessage error={list.error} onRetry={list.retry} />}
          {page && page.total === 0 && <EmptyState title="Nenhuma campanha com esses filtros." />}
          {page && page.total > 0 && (
            <>
              <div className="table-wrap">
                <table className="table table--rows">
                  <caption className="muted small">
                    {integer(page.total)} campanhas · score observado é o da campanha real; previsto, o do modelo da
                    Parte 2
                  </caption>
                  <thead>
                    <tr>
                      <th scope="col">Campanha</th>
                      <th scope="col">Plataforma</th>
                      <th scope="col">Objetivo</th>
                      <th scope="col" className="num">Observado</th>
                      <th scope="col" className="num">Previsto</th>
                      <th scope="col">Recomendações A · B · C</th>
                    </tr>
                  </thead>
                  <tbody>
                    {page.campanhas.map((c) => (
                      <tr
                        key={c.campaign_id}
                        className={selected === c.campaign_id ? 'is-selected' : undefined}
                        onClick={() => setSelected(c.campaign_id)}
                      >
                        <td>
                          <button type="button" className="link" onClick={() => setSelected(c.campaign_id)}>
                            {c.campaign_id}
                          </button>
                        </td>
                        <td>
                          <span className={`platform platform--${c.platform}`}>{c.platform}</span>
                        </td>
                        <td>{valueLabel('objective', c.objective)}</td>
                        <td className="num">{number(c.score_observado)}</td>
                        <td className="num">{number(c.score_previsto)}</td>
                        <td>
                          <Counts campaign={c} />
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <nav className="pager" aria-label="Paginação">
                <button
                  type="button"
                  className="button button--secondary"
                  disabled={page.pagina <= 1}
                  onClick={() => setFilters((f) => ({ ...f, pagina: page.pagina - 1 }))}
                >
                  Anterior
                </button>
                <span className="muted small">
                  Página {page.pagina} de {pages}
                </span>
                <button
                  type="button"
                  className="button button--secondary"
                  disabled={page.pagina >= pages}
                  onClick={() => setFilters((f) => ({ ...f, pagina: page.pagina + 1 }))}
                >
                  Próxima
                </button>
              </nav>
            </>
          )}
        </div>

        <aside className="surface pad detail" aria-live="polite">
          {!selected && (
            <EmptyState title="Selecione uma campanha">Clique numa linha para ver o contexto e o criativo.</EmptyState>
          )}
          {selected && detail.loading && <Loading />}
          {selected && detail.error && <ErrorMessage error={detail.error} onRetry={detail.retry} />}
          {detail.data && (
            <>
              <h2>{detail.data.campaign_id}</h2>
              <p className="detail__label">Contexto da campanha</p>
              <AttributeList fields={metadata.campos} values={detail.data.atributos} group="contexto" />
              <p className="detail__label">Criativo</p>
              <AttributeList fields={metadata.campos} values={detail.data.atributos} group="criativo" />
              <div className="detail__actions">
                <button
                  type="button"
                  className="button"
                  onClick={() => onOpen({ kind: 'historical', id: detail.data!.campaign_id }, 'analise')}
                >
                  Ver análise <ArrowRight size={16} aria-hidden="true" />
                </button>
                <button
                  type="button"
                  className="button button--secondary"
                  onClick={() => onOpen({ kind: 'historical', id: detail.data!.campaign_id }, 'laboratorio')}
                >
                  <FlaskConical size={16} aria-hidden="true" /> Testar alterações
                </button>
              </div>
            </>
          )}
        </aside>
      </div>
    </>
  )
}
