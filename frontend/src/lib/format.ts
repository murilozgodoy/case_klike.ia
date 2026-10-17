import type { Value } from '../types/api'

// Só tradução de rótulos e formatação de números: nenhuma regra de interpretação.

const VALUE_LABELS: Record<string, Record<string, string>> = {
  objective: {
    conversions: 'Conversões',
    traffic: 'Tráfego',
    engagement: 'Engajamento',
    awareness: 'Awareness',
    app_install: 'Instalação de app',
  },
  text_density: { low: 'Baixa', medium: 'Média', high: 'Alta' },
  format: { vertical: 'Vertical', quadrado: 'Quadrado', horizontal: 'Horizontal' },
}

export function valueLabel(field: string, value: Value): string {
  if (value === null || value === undefined) {
    return field === 'has_subtitle' ? 'Sem marcação' : 'Não informado'
  }
  if (typeof value === 'boolean') return value ? 'Sim' : 'Não'
  if (typeof value === 'number') {
    if (field === 'music_voice_ratio') return number(value, Math.abs(value * 100 - Math.round(value * 100)) < 1e-6 ? 2 : 3)
    if (field === 'video_duration_s') return number(value, 0)
    return number(value, 2)
  }
  return VALUE_LABELS[field]?.[value] ?? value
}

export function number(value: number, decimals = 1): string {
  return value.toLocaleString('pt-BR', { minimumFractionDigits: decimals, maximumFractionDigits: decimals })
}

// Diferenças com sinal explícito e o sinal de menos tipográfico, como no engine.
export function signed(value: number, decimals = 1): string {
  const text = number(Math.abs(value), decimals)
  if (Number(text.replace(',', '.')) === 0) return text
  return `${value > 0 ? '+' : '−'}${text}`
}

// Pontos de score como o engine os escreve: uma casa; perto de zero, duas, para não parecer zero.
export function points(value: number): string {
  return signed(value, Math.abs(value) > 0 && Math.abs(value) < 0.05 ? 2 : 1)
}

export function percent(value: number, decimals = 1): string {
  return `${number(value * 100, decimals)}%`
}

export function integer(value: number): string {
  return value.toLocaleString('pt-BR')
}
