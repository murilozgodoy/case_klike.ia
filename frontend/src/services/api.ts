import type {
  CampaignAttributes,
  CampaignDetail,
  CampaignPage,
  Metadata,
  Report,
  ScenarioResult,
  Selection,
  Value,
} from '../types/api'

// Em desenvolvimento, o Vite encaminha /api para o backend local (vite.config.ts).
const BASE_URL = (import.meta.env.VITE_API_URL ?? '').replace(/\/$/, '')
const TIMEOUT_MS = 70_000

export class ApiError extends Error {
  readonly status: number | null
  // Sem resposta, ou 502 a 504: o serviço gratuito pode estar despertando.
  readonly unavailable: boolean

  constructor(message: string, status: number | null, unavailable: boolean) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.unavailable = unavailable
  }
}

const UNAVAILABLE_MESSAGE =
  'Não foi possível falar com o servidor. No plano gratuito, ele pode levar cerca de um minuto para despertar depois de um tempo parado. Tente novamente.'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS)
  let response: Response
  try {
    response = await fetch(`${BASE_URL}${path}`, {
      ...init,
      headers: { 'Content-Type': 'application/json' },
      signal: controller.signal,
    })
  } catch {
    throw new ApiError(UNAVAILABLE_MESSAGE, null, true)
  } finally {
    clearTimeout(timer)
  }
  if (response.ok) return (await response.json()) as T
  if ([502, 503, 504].includes(response.status)) {
    throw new ApiError(UNAVAILABLE_MESSAGE, response.status, true)
  }
  let detail = `Erro ${response.status} ao consultar o servidor.`
  try {
    const body = (await response.json()) as { detail?: unknown }
    if (typeof body.detail === 'string') detail = body.detail
  } catch {
    // corpo sem JSON: fica a mensagem genérica
  }
  throw new ApiError(detail, response.status, false)
}

const post = <T,>(path: string, body: unknown) =>
  request<T>(path, { method: 'POST', body: JSON.stringify(body) })

export interface CampaignFilters {
  platform?: string
  objective?: string
  category?: string
  busca?: string
  pagina?: number
  por_pagina?: number
}

export const api = {
  metadata: () => request<Metadata>('/api/metadata'),
  campaigns: (filters: CampaignFilters) => {
    const params = new URLSearchParams()
    for (const [key, value] of Object.entries(filters)) {
      if (value !== undefined && value !== '') params.set(key, String(value))
    }
    return request<CampaignPage>(`/api/campaigns?${params}`)
  },
  campaign: (id: string) => request<CampaignDetail>(`/api/campaigns/${encodeURIComponent(id)}`),
  analysis: (selection: Selection) =>
    selection.kind === 'historical'
      ? request<Report>(`/api/campaigns/${encodeURIComponent(selection.id)}/analysis`)
      : post<Report>('/api/analyze', selection.campaign),
  compare: (selection: Selection, field: string, value: Value) =>
    post<ScenarioResult>('/api/scenarios/compare', {
      ...(selection.kind === 'historical'
        ? { campaign_id: selection.id }
        : { campaign: selection.campaign as CampaignAttributes }),
      changes: { [field]: value },
    }),
}
