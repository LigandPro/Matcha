export interface GeneratedVariant {
  id: number;
  label: string;
  sdf: string;
  rank: number | null;
  rmsd: number | null;
  score: number | null;
  smiles: string | null;
  duplicateCount: number | null;
  metricValues: Record<string, number | null>;
  primaryMetricKey: string | null;
  primaryMetricValue: number | null;
}
