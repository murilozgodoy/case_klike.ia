// Respostas reais da API, geradas a partir do backend com as campanhas da base.
import type { CampaignPage, Metadata, Report, ScenarioResult } from '../types/api'
import klk0001 from './fixtures/klk0001.json'
import klk0024 from './fixtures/klk0024.json'
import klk0291 from './fixtures/klk0291.json'
import metadata from './fixtures/metadata.json'
import page from './fixtures/page.json'
import scenario from './fixtures/scenario.json'

export const fixtures = {
  metadata: metadata as unknown as Metadata,
  klk0024: klk0024 as unknown as Report,
  klk0291: klk0291 as unknown as Report,
  klk0001: klk0001 as unknown as Report,
  scenario: scenario as unknown as ScenarioResult,
  page: page as unknown as CampaignPage,
}
