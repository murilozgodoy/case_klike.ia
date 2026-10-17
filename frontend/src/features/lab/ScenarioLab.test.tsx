import { cleanup, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { fixtures } from '../../test/fixtures'

const compare = vi.fn()
vi.mock('../../services/api', async (importOriginal) => {
  const original = await importOriginal<typeof import('../../services/api')>()
  return {
    ...original,
    api: {
      ...original.api,
      analysis: () => Promise.resolve(fixtures.klk0024),
      compare: (...args: unknown[]) => compare(...args),
    },
  }
})

const { ScenarioLab } = await import('./ScenarioLab')

afterEach(() => {
  cleanup()
  compare.mockReset()
})

describe('laboratório de cenários', () => {
  it('compara a edição sugerida pelo engine e restaura o cenário original', async () => {
    compare.mockResolvedValue(fixtures.scenario)
    const user = userEvent.setup()
    render(
      <ScenarioLab
        metadata={fixtures.metadata}
        selection={{ kind: 'historical', id: 'KLK-0024' }}
        preset={null}
        onExplore={() => {}}
      />,
    )
    await user.click(await screen.findByRole('button', { name: /Incluir um hook/ }))
    expect(compare).toHaveBeenCalledWith({ kind: 'historical', id: 'KLK-0024' }, 'has_hook', true)
    expect(await screen.findByText('Comparação')).toBeTruthy()
    expect(screen.getByText(/não é um teste A\/B/)).toBeTruthy()
    expect(screen.getByText('Edição recomendada pelo engine')).toBeTruthy()
    expect(screen.getByText(/As métricas observadas pertencem à campanha real/)).toBeTruthy()

    await user.click(screen.getByRole('button', { name: 'Restaurar cenário original' }))
    expect(screen.queryByText('Comparação')).toBeNull()
  })

  it('abre já comparando a edição escolhida na análise', async () => {
    compare.mockResolvedValue(fixtures.scenario)
    render(
      <ScenarioLab
        metadata={fixtures.metadata}
        selection={{ kind: 'historical', id: 'KLK-0024' }}
        preset={{ field: 'has_hook', value: true }}
        onExplore={() => {}}
      />,
    )
    expect(await screen.findByText('Comparação')).toBeTruthy()
    expect(compare).toHaveBeenCalledTimes(1)
  })
})
