import { afterEach, describe, expect, it, vi } from 'vitest'
import { api, ApiError } from './api'

afterEach(() => vi.unstubAllGlobals())

function respond(status: number, body: unknown) {
  return vi.fn().mockResolvedValue(new Response(JSON.stringify(body), { status }))
}

describe('cliente da API', () => {
  it('trata falha de rede como servidor indisponível, com tentativa sugerida', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')))
    const error = await api.metadata().catch((e: unknown) => e)
    expect(error).toBeInstanceOf(ApiError)
    expect((error as ApiError).unavailable).toBe(true)
    expect((error as ApiError).message).toMatch(/despertar/)
  })

  it('trata 503 como serviço despertando', async () => {
    vi.stubGlobal('fetch', respond(503, {}))
    const error = (await api.metadata().catch((e: unknown) => e)) as ApiError
    expect(error.unavailable).toBe(true)
    expect(error.status).toBe(503)
  })

  it('mostra a mensagem de validação do servidor', async () => {
    vi.stubGlobal('fetch', respond(422, { detail: 'platform: valor desconhecido' }))
    const error = (await api.metadata().catch((e: unknown) => e)) as ApiError
    expect(error.unavailable).toBe(false)
    expect(error.message).toBe('platform: valor desconhecido')
  })

  it('envia uma única alteração na comparação de cenários', async () => {
    const fetchMock = respond(200, {})
    vi.stubGlobal('fetch', fetchMock)
    await api.compare({ kind: 'historical', id: 'KLK-0024' }, 'has_hook', true)
    const [url, init] = fetchMock.mock.calls[0]
    expect(url).toBe('/api/scenarios/compare')
    expect(JSON.parse(init.body)).toEqual({ campaign_id: 'KLK-0024', changes: { has_hook: true } })
  })

  it('monta os filtros da listagem sem parâmetros vazios', async () => {
    const fetchMock = respond(200, { total: 0, pagina: 1, por_pagina: 20, campanhas: [] })
    vi.stubGlobal('fetch', fetchMock)
    await api.campaigns({ platform: 'Meta', objective: '', busca: undefined, pagina: 2 })
    expect(fetchMock.mock.calls[0][0]).toBe('/api/campaigns?platform=Meta&pagina=2')
  })
})
