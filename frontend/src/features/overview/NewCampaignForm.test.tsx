import { cleanup, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { fixtures } from '../../test/fixtures'
import { NewCampaignForm } from './NewCampaignForm'

afterEach(cleanup)

const CHOICES: Record<string, string> = {
  Plataforma: 'TikTok',
  Categoria: 'E-commerce',
  Objetivo: 'conversions',
  'Faixa etária': '18-24',
  Retargeting: 'false',
  Formato: 'quadrado',
  'Call-to-action': 'false',
  'Hook nos 3 primeiros segundos': 'false',
  'Rosto humano': 'true',
  'Densidade de texto': 'high',
}

describe('formulário de campanha nova', () => {
  it('só envia com os obrigatórios escolhidos e manda os opcionais vazios como ausentes', async () => {
    const onSubmit = vi.fn()
    const user = userEvent.setup()
    render(<NewCampaignForm fields={fixtures.metadata.campos} onSubmit={onSubmit} />)
    const submit = screen.getByRole('button', { name: 'Analisar campanha' }) as HTMLButtonElement
    expect(submit.disabled).toBe(true)

    for (const [label, value] of Object.entries(CHOICES)) {
      await user.selectOptions(screen.getByLabelText(label), value)
    }
    expect(submit.disabled).toBe(false)
    await user.click(submit)
    expect(onSubmit).toHaveBeenCalledWith({
      platform: 'TikTok',
      category: 'E-commerce',
      objective: 'conversions',
      target_audience_age: '18-24',
      is_retargeting: false,
      video_duration_s: null,
      format: 'quadrado',
      has_subtitle: null,
      has_cta: false,
      has_hook: false,
      has_face: true,
      text_density: 'high',
      music_voice_ratio: null,
    })
  })
})
