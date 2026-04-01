import { z } from 'zod';
import { GeneratedVariantSchema } from './variant-zod';

export const MatchaHealthSchema = z.object({
  status: z.string(),
  service: z.string().default('matcha-ui-runner'),
}).passthrough();

export const MatchaDefaultFixtureSchema = z.object({
  receptorFilename: z.string().min(1),
  receptorText: z.string().min(1),
  receptorSourcePath: z.string().min(1),
  ligandFilename: z.string().min(1),
  ligandText: z.string().min(1),
  ligandSourcePath: z.string().min(1),
});

export type MatchaDefaultFixture = z.infer<typeof MatchaDefaultFixtureSchema>;

export const MatchaSmilesPreviewRequestSchema = z.object({
  smiles: z.string().min(1),
  name: z.string().min(1).optional(),
});

export type MatchaSmilesPreviewRequest = z.infer<typeof MatchaSmilesPreviewRequestSchema>;

export const MatchaSmilesPreviewSchema = z.object({
  filename: z.string().min(1),
  text: z.string().min(1),
  sourcePath: z.string().min(1),
  smiles: z.string().min(1),
});

export type MatchaSmilesPreview = z.infer<typeof MatchaSmilesPreviewSchema>;

export const MatchaRunParamsSchema = z.object({
  runName: z.string().optional(),
  nSamples: z.number().int().positive(),
  numSteps: z.number().int().positive(),
  device: z.string().nullable().optional(),
  scorer: z.enum(['none', 'gnina']).default('none'),
  scorerMinimize: z.boolean().default(true),
  physicalOnly: z.boolean().default(false),
  bindingSiteMode: z.enum(['protein_center', 'blind', 'manual', 'box_json', 'autobox_ligand']).default('protein_center'),
  centerX: z.number().nullable().optional(),
  centerY: z.number().nullable().optional(),
  centerZ: z.number().nullable().optional(),
  boxJsonFilename: z.string().nullable().optional(),
  boxJsonText: z.string().nullable().optional(),
  autoboxLigandFilename: z.string().nullable().optional(),
  autoboxLigandText: z.string().nullable().optional(),
});

export const MatchaRunRequestSchema = z.object({
  receptorFilename: z.string().min(1),
  receptorText: z.string().min(1),
  ligandFilename: z.string().min(1),
  ligandText: z.string().min(1),
  params: MatchaRunParamsSchema,
});

export type MatchaRunRequest = z.infer<typeof MatchaRunRequestSchema>;

export const MatchaJobSchema = z.object({
  jobId: z.string(),
  state: z.enum(['queued', 'running', 'cancelling', 'completed', 'failed', 'cancelled']),
  message: z.string(),
  createdAt: z.number(),
  updatedAt: z.number(),
  error: z.string().nullable().optional(),
  resultReady: z.boolean(),
  logReady: z.boolean().default(false),
  cancelReady: z.boolean().default(false),
});

export type MatchaJob = z.infer<typeof MatchaJobSchema>;

const NumericVectorSchema = z.union([z.number(), z.array(z.number())]);

export const MatchaTrajectoryFrameSchema = z.object({
  id: z.number().int(),
  label: z.string(),
  sdf: z.string(),
  step: z.number().int(),
  time: z.number().nullable().optional(),
  deltaTranslation: z.array(z.number()).nullable().optional(),
  deltaRotation: z.array(z.number()).nullable().optional(),
  deltaTorsion: NumericVectorSchema.nullable().optional(),
  deltaTranslationNorm: z.number().nullable().optional(),
  deltaRotationNorm: z.number().nullable().optional(),
  deltaTorsionNorm: z.number().nullable().optional(),
  translation: z.array(z.number()).nullable().optional(),
  rotation: z.array(z.array(z.number())).nullable().optional(),
  torsion: NumericVectorSchema.nullable().optional(),
  translationNorm: z.number().nullable().optional(),
  torsionNorm: z.number().nullable().optional(),
});

export type MatchaTrajectoryFrame = z.infer<typeof MatchaTrajectoryFrameSchema>;

export const MatchaVariantSchema = GeneratedVariantSchema.extend({
  sampleIndex: z.number().int(),
  trajectoryFrames: z.array(MatchaTrajectoryFrameSchema).default([]),
});

export type MatchaVariant = z.infer<typeof MatchaVariantSchema>;

export const MatchaJobLogSchema = z.object({
  jobId: z.string(),
  text: z.string(),
  truncated: z.boolean().default(false),
  logPath: z.string().nullable().optional(),
});

export type MatchaJobLog = z.infer<typeof MatchaJobLogSchema>;

export const MatchaWorkspaceSchema = z.object({
  engine: z.literal('matcha'),
  runId: z.string(),
  receptor: z.object({
    filename: z.string(),
    format: z.string(),
    content: z.string(),
  }),
  ligand: z.object({
    filename: z.string(),
    format: z.string(),
    content: z.string(),
  }),
  variants: z.array(MatchaVariantSchema),
});

export type MatchaWorkspace = z.infer<typeof MatchaWorkspaceSchema>;
