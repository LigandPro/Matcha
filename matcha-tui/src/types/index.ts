// Screen types
export type Screen =
  | 'welcome'
  | 'setup'
  | 'setup-files'
  | 'setup-box'
  | 'setup-params'
  | 'setup-review'
  | 'running'
  | 'results'
  | 'history'
  | 'jobs';

// Docking mode
export type DockingMode = 'single' | 'batch';

// Search space mode
export type BoxMode = 'blind' | 'manual' | 'autobox';

// Job status
export type JobStatus = 'queued' | 'running' | 'completed' | 'failed' | 'cancelled';

// Pipeline stage
export type PipelineStage =
  | 'init'
  | 'checkpoints'
  | 'dataset'
  | 'esm'
  | 'stage1'
  | 'stage2'
  | 'stage3'
  | 'scoring'
  | 'posebusters'
  | 'done';

// Stage info for display
export interface StageInfo {
  id: PipelineStage;
  name: string;
  description: string;
}

export const PIPELINE_STAGES: StageInfo[] = [
  { id: 'checkpoints', name: 'Checkpoints', description: 'Loading model checkpoints' },
  { id: 'dataset', name: 'Dataset', description: 'Preparing dataset' },
  { id: 'esm', name: 'ESM Embeddings', description: 'Computing protein embeddings' },
  { id: 'stage1', name: 'Translation (R³)', description: 'Stage 1: Predicting translation' },
  { id: 'stage2', name: 'Rotation (SO(3))', description: 'Stage 2: Predicting rotation' },
  { id: 'stage3', name: 'Torsion (SO(2))', description: 'Stage 3: Predicting torsion angles' },
  { id: 'scoring', name: 'Scoring', description: 'Ranking poses by quality' },
  { id: 'posebusters', name: 'PoseBusters', description: 'Physical validity validation' },
];

// Box configuration
export interface BoxConfig {
  mode: BoxMode;
  centerX?: number;
  centerY?: number;
  centerZ?: number;
  autoboxLigand?: string;
}

// Docking parameters
export interface DockingParams {
  nSamples: number;
  nConfs?: number;
  gpu?: number;
  physicalOnly: boolean;
  runName: string;
  outputDir: string;
  checkpointsDir?: string;
}

// Job configuration
export interface JobConfig {
  mode: DockingMode;
  receptor: string;
  ligand?: string;
  ligandDir?: string;
  box: BoxConfig;
  params: DockingParams;
}

// Active job info (for job queue)
export interface ActiveJob {
  id: string;
  config: JobConfig;
  status: JobStatus;
  startTime?: string;
  endTime?: string;
  progress?: ProgressInfo;
  error?: string;
}

// Ligand status in batch mode
export interface LigandStatus {
  uid: string;
  name: string;
  status: 'pending' | 'running' | 'done' | 'failed';
  pbCount?: number;
  gninaScore?: number;
}

// Progress info
export interface ProgressInfo {
  job_id: string;
  stage: PipelineStage;
  percent: number;
  eta?: number;
  currentLigand?: string;
  ligands?: LigandStatus[];
}

// Pose result (matches backend snake_case format)
export interface PoseResult {
  rank: number;
  pb_count: number;
  checks: {
    not_too_far_away: boolean;
    no_internal_clash: boolean;
    no_clashes: boolean;
    no_volume_clash: boolean;
  };
  buried_fraction: number;
  gnina_score?: number;
}

// Batch ligand status (from backend)
export interface BatchLigandStatus {
  name: string;
  path: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  pb_count?: number;
  gnina_score?: number;
  error_message?: string;
}

// Job results
export interface JobResults {
  runName: string;
  runtime: number;
  totalPoses: number;
  physicalPoses: number;
  poses: PoseResult[];
  bestPosePath: string;
  allPosesPath: string;
  logPath: string;
  receptor?: string;
  ligand?: string;
  // Batch mode fields
  totalLigands?: number;
  ligandStatuses?: BatchLigandStatus[];
}

// Stored job for history
export interface StoredJob {
  id: string;
  config: JobConfig;
  status: JobStatus;
  startTime: string;
  endTime?: string;
  results?: JobResults;
}
