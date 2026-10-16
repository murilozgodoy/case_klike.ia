"""Configuração central: caminhos, grupos de colunas e constantes do dataset."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "klike_challenge_dataset.csv"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
# Modelo treinado: gerado por `python -m klike.train`, fora do controle de versão.
MODELS_DIR = ROOT_DIR / "models"
MODEL_PATH = MODELS_DIR / "klike_score_model.joblib"

ID_COL = "campaign_id"
DATE_COL = "date"
TARGET = "klike_score"

# As colunas se dividem pelo MOMENTO em que a informação existe. Essa divisão guia
# tudo o que vem depois: o modelo "pré-veiculação" só pode usar contexto + criativo,
# e o recommendations engine só sugere mudanças em atributos do criativo.

# Contexto da campanha: conhecido antes da veiculação, mas não é decisão criativa.
CONTEXT_COLS = ["platform", "category", "objective", "target_audience_age", "is_retargeting"]

# Atributos do criativo: o que o time de criação controla.
CREATIVE_COLS = [
    "video_duration_s",
    "format",
    "has_subtitle",
    "has_cta",
    "has_hook",
    "has_face",
    "text_density",
    "music_voice_ratio",
]

# Performance: só existe depois que a mídia roda.
PERFORMANCE_COLS = [
    "impressions",
    "clicks",
    "ctr",
    "cpc",
    "spend",
    "conversions",
    "revenue",
    "roas",
    "avg_watch_time_s",
    "engagement_rate",
]

BOOL_COLS = ["is_retargeting", "has_subtitle", "has_cta", "has_hook", "has_face"]

# Marcações que a limpeza acrescenta (data.clean): não são erro, mudam a forma de analisar.
FLAG_COLS = ["flag_ctr_at_cap", "flag_engagement_at_cap", "flag_full_watch", "flag_roas_applicable"]

# Ordem fixa das categorias: mantém tabelas e gráficos consistentes entre notebooks.
CATEGORY_ORDERS = {
    "platform": ["Meta", "TikTok", "LinkedIn"],
    "category": ["E-commerce", "SaaS", "Lead Gen", "App Install", "Branding"],
    "objective": ["conversions", "traffic", "engagement", "awareness", "app_install"],
    "target_audience_age": ["18-24", "25-34", "35-44", "45+"],
    "format": ["vertical", "quadrado", "horizontal"],
    "text_density": ["low", "medium", "high"],
}
ORDERED_CATEGORIES = {"target_audience_age", "text_density"}

# Possíveis tetos de registro: muitos valores "empilhados" exatamente no limite, e
# nenhum acima. Se for censura, o valor real pode ter sido maior. Não é tratado como erro.
CTR_CAP = 0.15
ENGAGEMENT_CAP = 0.10

# Nomes legíveis para gráficos e tabelas voltados ao time de marketing.
COLUMN_LABELS = {
    "platform": "Plataforma",
    "category": "Categoria",
    "objective": "Objetivo",
    "target_audience_age": "Faixa etária",
    "is_retargeting": "Retargeting",
    "video_duration_s": "Duração do vídeo (s)",
    "format": "Formato",
    "has_subtitle": "Legenda",
    "has_cta": "Call-to-action",
    "has_hook": "Hook nos 3 primeiros segundos",
    "has_face": "Rosto humano",
    "text_density": "Densidade de texto",
    "music_voice_ratio": "Proporção música/voz",
    "impressions": "Impressões",
    "clicks": "Cliques",
    "ctr": "CTR",
    "cpc": "CPC (R$)",
    "spend": "Gasto (R$)",
    "conversions": "Conversões",
    "revenue": "Receita (R$)",
    "roas": "ROAS",
    "avg_watch_time_s": "Tempo médio assistido (s)",
    "engagement_rate": "Taxa de engajamento",
    "klike_score": "Klike Score",
}
