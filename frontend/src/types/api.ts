// Contratos da API. O relatório segue RecommendationReport.to_dict() do pacote klike:
// a interface só lê esses campos, sem recalcular nada.

export type Value = string | number | boolean | null

export type CampaignAttributes = Record<string, Value>

export interface FieldInfo {
  campo: string
  rotulo: string
  grupo: 'contexto' | 'criativo'
  tipo: 'categoria' | 'booleano' | 'numero'
  opcoes: string[] | null
  obrigatorio: boolean
  minimo?: number
  maximo?: number
  minimo_exclusivo?: boolean
  passo?: number
  unidade?: string
}

export interface GuidedExample {
  campaign_id: string
  titulo: string
  descricao: string
}

export interface Metadata {
  campos: FieldInfo[]
  atributos_contexto: string[]
  atributos_criativo: string[]
  metricas: Record<string, string>
  metrica_por_objetivo: Record<string, string>
  classes: Record<string, string>
  nota_metodologica: string
  indicadores: {
    campanhas: number
    modelo: { descricao: string; mae: number; mae_ic95: [number, number]; mae_dummy: number; r2: number }
    avaliacao_engine: {
      descricao: string
      recomendacoes: number
      campanhas_tres_ou_mais: number
      parcela_tres_ou_mais: number
      campanhas_com_evidencia: number
      campanhas_tres_ou_mais_com_evidencia: number
      campanhas_sem_recomendacao: number
      classes: Record<string, number>
    }
  }
  exemplos: GuidedExample[]
}

export interface CampaignSummary {
  campaign_id: string
  platform: string
  category: string
  objective: string
  format: string
  score_observado: number
  score_previsto: number
  recomendacoes: number
  A: number
  B: number
  C: number
}

export interface CampaignPage {
  total: number
  pagina: number
  por_pagina: number
  campanhas: CampaignSummary[]
}

export interface CampaignDetail {
  campaign_id: string
  atributos: CampaignAttributes
  observado: Record<string, number | null>
  resumo: CampaignSummary
}

export interface Association {
  metrica: string
  metrica_nome: string
  variacao: number
  ic_inferior: number
  ic_superior: number
  evidencia: string
  estavel: boolean
  p_valor: number
  q_valor: number
  campanhas: number
  sensibilidade: string | null
  limitacoes: string
}

export interface Support {
  recorte: string
  campanhas_valor_atual: number
  campanhas_valor_sugerido: number
  grupo_valor_atual: string
  grupo_valor_sugerido: string
  minimo_por_lado: number
  risco_extrapolacao: string
  observacao: string
  aproximacao?: string
}

export type RecommendationClass = 'A' | 'B' | 'C'

export interface Recommendation {
  prioridade: number
  acao: string
  titulo: string
  campo: string
  valor_atual: Value
  valor_sugerido: Value
  de: string
  para: string
  classe: RecommendationClass
  classe_descricao: string
  impacto_estimado: string
  metrica: string
  metrica_nome: string
  associacao: Association | null
  outras_associacoes: Association[]
  score_antes: number
  score_depois: number
  diferenca_score: number
  diferenca_score_modelo: { estimativa: number; ic_inferior: number; ic_superior: number; observacao: string }
  suporte: Support
  justificativa: string
  nota_de_prioridade: string | null
  avisos: string[]
  ressalvas: string[]
  proximo_passo: string
}

export interface ExecutiveEntry {
  prioridade: number
  acao: string
  titulo: string
  classe: RecommendationClass
  metrica_nome: string
  impacto_estimado: string
  diferenca_score: number
  ressalvas: string[]
}

export interface Report {
  campanha_id: string | null
  campanha: CampaignAttributes
  contexto: CampaignAttributes
  objetivo: string
  metrica: string
  metrica_nome: string
  metrica_principal_do_objetivo: string
  metrica_selecionada_manualmente: boolean
  score_previsto: number
  observado: Record<string, number | null> | null
  campanhas_na_mesma_combinacao: number
  resumo: string
  composicao: {
    recomendacoes: number
    com_evidencia_na_metrica: number
    hipoteses_de_score: number
    por_classe: Record<RecommendationClass, number>
    menos_de_tres: boolean
  }
  principais: ExecutiveEntry[]
  nota_metodologica: string
  avisos: string[]
  recomendacoes: Recommendation[]
  nao_recomendadas: { acao: string; motivo: string }[]
}

export type EvidenceStatus = 'recomendada' | 'avaliada_nao_recomendada' | 'fora_do_escopo' | 'sem_evidencia'

export interface ScenarioResult {
  alteracao: { campo: string; rotulo: string; valor_original: Value; valor_novo: Value }
  score: { original: number; modificado: number; diferenca: number }
  evidencia: { situacao: EvidenceStatus; acao: string | null; recomendacao?: Recommendation; motivo?: string }
  nota: string
  original: Report
  modificado: Report
}

// Campanha em análise: do histórico, pelo id, ou nova, pelos atributos.
export type Selection = { kind: 'historical'; id: string } | { kind: 'new'; campaign: CampaignAttributes }
