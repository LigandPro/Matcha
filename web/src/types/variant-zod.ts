import { z } from 'zod';

export const GeneratedVariantSchema = z.object({
  id: z.number().int(),
  label: z.string(),
  sdf: z.string(),
  rank: z.number().nullable(),
  rmsd: z.number().nullable(),
  score: z.number().nullable(),
  smiles: z.string().nullable(),
  duplicateCount: z.number().nullable(),
  metricValues: z.record(z.string(), z.number().nullable()),
  primaryMetricKey: z.string().nullable(),
  primaryMetricValue: z.number().nullable(),
});
