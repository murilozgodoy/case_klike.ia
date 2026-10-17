import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { fixtures } from '../../test/fixtures'
import type { Report } from '../../types/api'

const analysis = vi.fn<(selection: unknown) => Promise<Report>>()
vi.mock('../../services/api', async (importOriginal) => {
  const original = await importOriginal<typeof import('../../services/api')>()
  return { ...original, api: { ...original.api, analysis: (s: unknown) => analysis(s) } }
})

const { AnalysisPage } = await import('./AnalysisPage')

afterEach(() => {
  cleanup()
  analysis.mockReset()
})

function show(report: Report) {
  analysis.mockResolvedValue(report)
  render(
    <AnalysisPage
      metadata={fixtures.metadata}
      selection={{ kind: 'historical', id: report.campanha_id ?? 'nova' }}
      onTest={() => {}}
      onExplore={() => {}}
    />,
  )
}

describe('análise de uma campanha', () => {
  it('mostra as recomendações na ordem do engine, com classe e ressalvas', async () => {
    show(fixtures.klk0024)
    const titles = await screen.findAllByRole('heading', { level: 3 })
    expect(titles.map((h) => h.textContent)).toEqual(fixtures.klk0024.recomendacoes.map((r) => r.titulo))
    expect(screen.getAllByText(/Classe A/).length).toBeGreaterThan(0)
    expect(screen.getAllByText('classe sensível à análise').length).toBeGreaterThan(0)
    // Score previsto e observado aparecem separados.
    expect(screen.getByText('Klike Score previsto', { selector: '.score__label' })).toBeTruthy()
    expect(screen.getByText('Klike Score observado', { selector: '.score__label' })).toBeTruthy()
  })

  it('não completa a lista quando o engine não tem recomendação', async () => {
    show(fixtures.klk0291)
    expect(await screen.findByText('Nenhuma recomendação para esta campanha')).toBeTruthy()
    expect(screen.queryByText('Recomendações em detalhe')).toBeNull()
  })

  it('explica quando há menos de três recomendações e mostra o score em queda da música', async () => {
    show(fixtures.klk0001)
    expect(await screen.findByText('Menos de três recomendações')).toBeTruthy()
    expect(screen.getAllByText('score previsto cai').length).toBeGreaterThan(0)
  })

  it('pede uma campanha quando nada foi selecionado', () => {
    render(<AnalysisPage metadata={fixtures.metadata} selection={null} onTest={() => {}} onExplore={() => {}} />)
    expect(screen.getByText('Nenhuma campanha selecionada')).toBeTruthy()
    expect(analysis).not.toHaveBeenCalled()
  })
})
