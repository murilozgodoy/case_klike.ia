import { ArrowRight } from 'lucide-react'
import { useState, type FormEvent } from 'react'
import { valueLabel } from '../../lib/format'
import type { CampaignAttributes, FieldInfo, Value } from '../../types/api'

interface Props {
  fields: FieldInfo[]
  onSubmit: (campaign: CampaignAttributes) => void
}

// Valores iniciais só preenchem o formulário; os obrigatórios precisam ser escolhidos.
const UNSET = ''

function initial(fields: FieldInfo[]): Record<string, string> {
  return Object.fromEntries(fields.map((field) => [field.campo, UNSET]))
}

// Converte o texto do formulário no valor da API; vazio é ausente (só nos opcionais).
function parse(field: FieldInfo, text: string): Value {
  if (text === UNSET) return null
  if (field.tipo === 'booleano') return text === 'true'
  if (field.tipo === 'numero') return Number(text.replace(',', '.'))
  return text
}

export function NewCampaignForm({ fields, onSubmit }: Props) {
  const [values, setValues] = useState<Record<string, string>>(() => initial(fields))
  const missing = fields.filter((f) => f.obrigatorio && values[f.campo] === UNSET)

  function submit(event: FormEvent) {
    event.preventDefault()
    if (missing.length > 0) return
    onSubmit(Object.fromEntries(fields.map((f) => [f.campo, parse(f, values[f.campo])])))
  }

  function renderInput(field: FieldInfo) {
    const value = values[field.campo]
    const set = (next: string) => setValues((current) => ({ ...current, [field.campo]: next }))
    const empty = field.obrigatorio ? 'Escolha…' : field.campo === 'has_subtitle' ? 'Sem marcação' : 'Não informado'
    if (field.tipo === 'numero') {
      return (
        <input
          type="number"
          inputMode="decimal"
          min={field.minimo}
          max={field.maximo}
          step={field.passo ?? 'any'}
          placeholder="Não informado"
          value={value}
          onChange={(e) => set(e.target.value)}
        />
      )
    }
    const options = field.tipo === 'booleano' ? ['true', 'false'] : field.opcoes ?? []
    return (
      <select value={value} onChange={(e) => set(e.target.value)}>
        <option value={UNSET}>{empty}</option>
        {options.map((option) => (
          <option key={option} value={option}>
            {field.tipo === 'booleano' ? valueLabel(field.campo, option === 'true') : valueLabel(field.campo, option)}
          </option>
        ))}
      </select>
    )
  }

  return (
    <form className="surface pad form" onSubmit={submit} aria-label="Nova campanha">
      <p className="muted small">
        Uma campanha hipotética passa pelo mesmo preditor e pelo mesmo engine das campanhas da base. Ela
        não é guardada. Duração, legenda e música/voz podem ficar sem valor, como no histórico; o
        servidor valida o restante.
      </p>
      {(['contexto', 'criativo'] as const).map((group) => (
        <fieldset key={group}>
          <legend>{group === 'contexto' ? 'Contexto da campanha' : 'Criativo'}</legend>
          <div className="form__grid">
            {fields
              .filter((f) => f.grupo === group)
              .map((field) => (
                <label key={field.campo} className="field">
                  <span>
                    {field.rotulo}
                    {!field.obrigatorio && <span className="muted"> (opcional)</span>}
                  </span>
                  {renderInput(field)}
                </label>
              ))}
          </div>
        </fieldset>
      ))}
      <div className="form__footer">
        {missing.length > 0 && (
          <p className="muted small">Faltam: {missing.map((f) => f.rotulo).join(', ')}.</p>
        )}
        <button type="submit" className="button" disabled={missing.length > 0}>
          Analisar campanha <ArrowRight size={16} aria-hidden="true" />
        </button>
      </div>
    </form>
  )
}
