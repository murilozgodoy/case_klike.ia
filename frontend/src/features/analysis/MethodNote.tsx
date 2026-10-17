import { CircleHelp } from 'lucide-react'

// Como ler os números: curto, e sempre a um clique de distância.
export function MethodNote({ note, classes }: { note: string; classes: Record<string, string> }) {
  return (
    <details className="surface method">
      <summary>
        <CircleHelp size={18} aria-hidden="true" /> Como ler esta análise
      </summary>
      <dl className="method__list">
        <div>
          <dt>Klike Score previsto</dt>
          <dd>
            A previsão de um modelo linear (Parte 2), com erro médio de 6,43 pontos no conjunto de teste. A
            diferença entre duas versões é o que o modelo prevê, não o que acontece de fato.
          </dd>
        </div>
        <div>
          <dt>Associação com performance</dt>
          <dd>
            Quanto a métrica do objetivo difere com e sem a característica, numa regressão sobre a base
            histórica que mantém fixos o contexto e os outros atributos do criativo. É associação, não
            causa.
          </dd>
        </div>
        <div>
          <dt>Classes de evidência</dt>
          <dd>
            A: {classes.A}. B: {classes.B}. C: {classes.C}. Nenhuma classe garante o resultado.
          </dd>
        </div>
        <div>
          <dt>Suporte histórico</dt>
          <dd>
            Quantas campanhas comparáveis existem dos dois lados da edição, no recorte indicado. Descreve
            a base de comparação; não estima o efeito naquele recorte.
          </dd>
        </div>
        <div>
          <dt>O que nada disso mede</dt>
          <dd>
            O efeito causal de editar um vídeo. Só um teste A/B com veiculação real responde isso, e os
            dados aqui são simulados.
          </dd>
        </div>
      </dl>
      <p className="muted small">Nota do engine: {note}</p>
    </details>
  )
}
