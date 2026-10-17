import { useCallback, useEffect, useRef, useState } from 'react'
import { ApiError } from '../services/api'

export interface RequestState<T> {
  data: T | null
  error: ApiError | null
  loading: boolean
  retry: () => void
}

interface Result<T> {
  id: string
  data: T | null
  error: ApiError | null
}

// Executa `load` sempre que `key` muda; sem chave, não há requisição. O resultado fica
// associado à chave: uma resposta antiga nunca aparece no lugar da atual.
export function useRequest<T>(key: string | null, load: () => Promise<T>): RequestState<T> {
  const [attempt, setAttempt] = useState(0)
  const [result, setResult] = useState<Result<T> | null>(null)
  const loadRef = useRef(load)
  useEffect(() => {
    loadRef.current = load
  })

  const id = key === null ? null : `${key}#${attempt}`
  useEffect(() => {
    if (id === null) return
    let active = true
    loadRef.current().then(
      (data) => {
        if (active) setResult({ id, data, error: null })
      },
      (reason: unknown) => {
        const error = reason instanceof ApiError ? reason : new ApiError(String(reason), null, false)
        if (active) setResult({ id, data: null, error })
      },
    )
    return () => {
      active = false
    }
  }, [id])

  const retry = useCallback(() => setAttempt((n) => n + 1), [])
  const current = result !== null && result.id === id ? result : null
  return {
    data: current?.data ?? null,
    error: current?.error ?? null,
    loading: id !== null && current === null,
    retry,
  }
}
