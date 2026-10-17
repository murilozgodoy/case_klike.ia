import { CircleAlert, Inbox, RotateCcw } from 'lucide-react'
import { useEffect, useState, type ReactNode } from 'react'
import type { ApiError } from '../services/api'

export function Loading({ label = 'Carregando…' }: { label?: string }) {
  // Depois de alguns segundos, avisa que o servidor gratuito pode estar despertando.
  const [slow, setSlow] = useState(false)
  useEffect(() => {
    const timer = setTimeout(() => setSlow(true), 4000)
    return () => clearTimeout(timer)
  }, [])
  return (
    <div className="status" role="status" aria-live="polite">
      <span className="spinner" aria-hidden="true" />
      <div>
        <p>{label}</p>
        {slow && (
          <p className="muted small">
            Está demorando: o servidor do plano gratuito pode estar despertando, o que leva cerca de
            um minuto.
          </p>
        )}
      </div>
    </div>
  )
}

export function ErrorMessage({ error, onRetry }: { error: ApiError; onRetry?: () => void }) {
  return (
    <div className="status status--error" role="alert">
      <CircleAlert className="icon" size={20} aria-hidden="true" />
      <div>
        <p>
          <strong>{error.unavailable ? 'Servidor indisponível no momento.' : 'Não foi possível concluir.'}</strong>
        </p>
        <p>{error.message}</p>
        {onRetry && (
          <button type="button" className="button button--secondary" onClick={onRetry}>
            <RotateCcw size={16} aria-hidden="true" />
            Tentar novamente
          </button>
        )}
      </div>
    </div>
  )
}

export function EmptyState({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="empty">
      <p className="empty__title">
        <Inbox size={18} aria-hidden="true" />
        {title}
      </p>
      {children && <div className="muted">{children}</div>}
    </div>
  )
}
