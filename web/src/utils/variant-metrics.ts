import type { GeneratedVariant } from '../types/variant';

export const PRIMARY_METRIC_PRIORITY = [
  'gnina_score',
  'posebusters_filters_passed_count_fast',
  'error_estimate_0',
  'conditioning_score',
  'esp_combo_score',
  'pharm_combo_score',
  'shape_score',
  'esp_score',
  'pharm_score',
  'qed',
  'fsp3',
  'score',
] as const;

const METRIC_LABELS: Record<string, string> = {
  conditioning_score: 'Conditioning',
  esp_combo_score: 'Shape + ESP',
  pharm_combo_score: 'Shape + Pharm',
  shape_score: 'Shape',
  esp_score: 'ESP',
  pharm_score: 'Pharm',
  qed: 'QED',
  fsp3: 'Fsp3',
  sa_score: 'SA',
  logp: 'LogP',
  gnina_score: 'GNINA',
  posebusters_filters_passed_count_fast: 'PB Checks',
  error_estimate_0: 'Predicted Error',
  buried_fraction: 'Buried',
  rmsd: 'RMSD',
  score: 'Score',
};

const PERCENT_METRICS = new Set<string>([
  'conditioning_score',
  'esp_combo_score',
  'pharm_combo_score',
  'shape_score',
  'esp_score',
  'pharm_score',
  'qed',
  'fsp3',
  'score',
]);

export const DETAIL_METRIC_KEYS = [
  'posebusters_filters_passed_count_fast',
  'gnina_score',
  'error_estimate_0',
  'buried_fraction',
  'shape_score',
  'esp_score',
  'pharm_score',
  'qed',
  'sa_score',
  'logp',
  'fsp3',
] as const;

export const metricDisplayLabel = (key: string | null): string =>
  key ? (METRIC_LABELS[key] ?? key) : 'Score';

export const formatMetricValue = (key: string | null, value: number | null): string => {
  if (typeof value !== 'number' || !Number.isFinite(value)) return 'n/a';
  if (!key) return value.toFixed(3);
  if (key === 'rmsd') return `${value.toFixed(2)}A`;
  if (key === 'gnina_score') return value.toFixed(2);
  if (key === 'posebusters_filters_passed_count_fast') return `${Math.round(value)}/4`;
  if (key === 'error_estimate_0') return value.toFixed(2);
  if (key === 'buried_fraction') return `${Math.round(value * 100)}%`;
  if (PERCENT_METRICS.has(key)) return `${Math.round(value * 100)}%`;
  if (key === 'logp' || key === 'sa_score') return value.toFixed(2);
  return value.toFixed(3);
};

export const metricTone = (key: string | null, value: number | null): string => {
  if (typeof value !== 'number' || !Number.isFinite(value)) return 'var(--lp-ink-muted)';
  if (key === 'logp' || key === 'sa_score' || key === 'rmsd' || key === 'error_estimate_0') return 'var(--lp-ink)';
  if (key === 'gnina_score') {
    if (value <= -8) return 'var(--lp-success-ink)';
    if (value <= -6) return 'var(--lp-warning-ink)';
    return 'var(--lp-danger-ink)';
  }
  if (key === 'posebusters_filters_passed_count_fast') {
    if (value >= 4) return 'var(--lp-success-ink)';
    if (value >= 2) return 'var(--lp-warning-ink)';
    return 'var(--lp-danger-ink)';
  }
  if (value >= 0.75) return 'var(--lp-success-ink)';
  if (value >= 0.5) return 'var(--lp-warning-ink)';
  return 'var(--lp-danger-ink)';
};

export const collectDetailMetrics = (variant: GeneratedVariant): Array<{ key: string; value: number | null }> =>
  DETAIL_METRIC_KEYS
    .filter((key) => key !== variant.primaryMetricKey)
    .map((key) => ({ key, value: variant.metricValues[key] ?? null }))
    .filter((entry) => entry.value !== null);

export const summarizeVariantMetrics = (
  variant: Pick<GeneratedVariant, 'primaryMetricKey' | 'primaryMetricValue' | 'rmsd' | 'rank' | 'duplicateCount'>
): string => {
  const primaryLabel = metricDisplayLabel(variant.primaryMetricKey);
  const primaryValue = formatMetricValue(variant.primaryMetricKey, variant.primaryMetricValue);
  const rmsdValue = formatMetricValue('rmsd', variant.rmsd);
  const rankLabel = Number.isFinite(variant.rank) ? `#${Math.round(variant.rank as number)}` : 'n/a';
  const duplicateLabel = Number.isFinite(variant.duplicateCount) ? `${Math.round(variant.duplicateCount as number)}` : '0';
  return `${primaryLabel}: ${primaryValue}, rank: ${rankLabel}, RMSD: ${rmsdValue}, duplicates: ${duplicateLabel}`;
};
