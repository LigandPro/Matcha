import { create } from 'zustand';
import type { Screen, JobConfig, ProgressInfo, JobResults, StoredJob, ActiveJob, JobStatus, PipelineStage, PoseResult, BatchLigandStatus } from '../types/index.js';
import type { ProgressEvent } from '../services/index.js';

type StageRuntimeStatus = {
  status: 'pending' | 'running' | 'done' | 'error';
  progress: number;
  elapsed?: number;
  message?: string;
};

type BatchRuntimeState = {
  isBatch: boolean;
  totalLigands: number;
  currentLigandIndex: number;
  currentLigand: string | null;
  ligandStatuses: BatchLigandStatus[];
};

export type JobRuntimeState = {
  startedAtMs: number | null;
  stages: Record<string, StageRuntimeStatus>;
  poses: PoseResult[];
  batch: BatchRuntimeState;
  completion: {
    outputPath: string;
    poses: PoseResult[];
    totalLigands?: number;
    ligandStatuses?: BatchLigandStatus[];
  } | null;
};

function createInitialStages(): Record<string, StageRuntimeStatus> {
  const stages: Record<string, StageRuntimeStatus> = {
    init: { status: 'pending', progress: 0 },
    checkpoints: { status: 'pending', progress: 0 },
    dataset: { status: 'pending', progress: 0 },
    esm: { status: 'pending', progress: 0 },
    stage1: { status: 'pending', progress: 0 },
    stage2: { status: 'pending', progress: 0 },
    stage3: { status: 'pending', progress: 0 },
    scoring: { status: 'pending', progress: 0 },
    posebusters: { status: 'pending', progress: 0 },
    done: { status: 'pending', progress: 0 },
  };
  return stages;
}

function createInitialJobRuntime(): JobRuntimeState {
  return {
    startedAtMs: null,
    stages: createInitialStages(),
    poses: [],
    batch: {
      isBatch: false,
      totalLigands: 1,
      currentLigandIndex: 0,
      currentLigand: null,
      ligandStatuses: [],
    },
    completion: null,
  };
}

export type LogLevel = 'debug' | 'info' | 'warn' | 'error';
export type ResultsViewMode = 'summary' | 'ligands' | 'ligand-detail';

const STORE_CONFIG = {
  MAX_LOGS: 500,
  MAX_HISTORY: 50,
  MAX_NAV_HISTORY: 100,
  MAX_DEBUG_LOGS: 1000,
} as const;

interface ResultsUIState {
  viewMode: ResultsViewMode;
  selectedTopIndex: number;
  selectedLigandIndex: number;
  activeLigandIndex: number | null;
  detailReturnView: 'summary' | 'ligands';
  runName: string | null;
}

export interface DebugLog {
  timestamp: string;
  level: LogLevel;
  component: string;
  message: string;
  data?: Record<string, unknown>;
}

interface AppState {
  // Navigation
  screen: Screen;
  setScreen: (screen: Screen) => void;
  previousScreen: Screen | null;
  setPreviousScreen: (screen: Screen | null) => void;

  // Current job configuration
  jobConfig: Partial<JobConfig>;
  setJobConfig: (config: Partial<JobConfig>) => void;
  resetJobConfig: () => void;

  // Running job state (legacy - kept for backward compatibility)
  isRunning: boolean;
  progress: ProgressInfo | null;
  logs: string[];
  setRunning: (running: boolean) => void;
  setProgress: (progress: ProgressInfo | null) => void;
  addLog: (log: string) => void;
  clearLogs: () => void;

  // Multi-job queue state
  activeJobs: Map<string, ActiveJob>;
  currentJobId: string | null;
  jobResults: Map<string, JobResults>;
  jobLogs: Map<string, string[]>;
  jobRuntime: Map<string, JobRuntimeState>;
  addJob: (job: ActiveJob) => void;
  updateJobStatus: (id: string, status: JobStatus) => void;
  updateJobProgress: (id: string, progress: ProgressInfo) => void;
  removeJob: (id: string) => void;
  setCurrentJob: (id: string | null) => void;
  getRunningJob: () => ActiveJob | null;
  getQueuedJobs: () => ActiveJob[];
  syncJobsFromBackend: (jobs: Array<Record<string, unknown>>) => void;
  applyProgressEvent: (event: ProgressEvent) => void;

  // Results
  results: JobResults | null;
  setResults: (results: JobResults | null) => void;
  resultsUI: ResultsUIState;
  setResultsUI: (partial: Partial<ResultsUIState>) => void;
  resetResultsUI: (next?: Partial<ResultsUIState>) => void;

  // Job history
  history: StoredJob[];
  addToHistory: (job: StoredJob) => void;
  removeFromHistory: (id: string) => void;
  clearHistory: () => void;
  historySelectedIndex: number;
  setHistorySelectedIndex: (index: number) => void;

  // UI state
  error: string | null;
  setError: (error: string | null) => void;
  notification: string | null;
  setNotification: (notification: string | null) => void;
  modalOpen: boolean;
  setModalOpen: (open: boolean) => void;

  // Debug mode
  debugMode: boolean;
  navigationHistory: string[];
  debugLogs: DebugLog[];
  trackNavigation: (from: Screen, to: Screen) => void;
  addDebugLog: (log: Omit<DebugLog, 'timestamp'>) => void;
  clearDebugLogs: () => void;
}

const initialJobConfig: Partial<JobConfig> = {
  mode: 'single',
  box: {
    mode: 'blind',
  },
  params: {
    nSamples: 40,
    physicalOnly: false,
    runName: '',
    outputDir: './results',
  },
};

const defaultResultsUI: ResultsUIState = {
  viewMode: 'summary',
  selectedTopIndex: 0,
  selectedLigandIndex: 0,
  activeLigandIndex: null,
  detailReturnView: 'summary',
  runName: null,
};

export const useStore = create<AppState>((set, get) => ({
  // Navigation
  screen: 'welcome',
  setScreen: (screen) => set({ screen }),
  previousScreen: null,
  setPreviousScreen: (screen) => set({ previousScreen: screen }),

  // Job configuration
  jobConfig: { ...initialJobConfig },
  setJobConfig: (config) =>
    set((state) => ({
      jobConfig: { ...state.jobConfig, ...config },
    })),
  resetJobConfig: () => set({ jobConfig: { ...initialJobConfig } }),

  // Running state
  isRunning: false,
  progress: null,
  logs: [],
  setRunning: (running) => set({ isRunning: running }),
  setProgress: (progress) => set({ progress }),
  addLog: (log) =>
    set((state) => ({
      logs: [...state.logs.slice(-STORE_CONFIG.MAX_LOGS), log],
    })),
  clearLogs: () => set({ logs: [] }),

  // Multi-job queue
  activeJobs: new Map(),
  currentJobId: null,
  jobResults: new Map(),
  jobLogs: new Map(),
  jobRuntime: new Map(),

  addJob: (job) =>
    set((state) => {
      const jobs = new Map(state.activeJobs);
      jobs.set(job.id, job);
      const runtime = new Map(state.jobRuntime);
      if (!runtime.has(job.id)) {
        runtime.set(job.id, createInitialJobRuntime());
      }
      return { activeJobs: jobs, jobRuntime: runtime };
    }),

  updateJobStatus: (id, status) =>
    set((state) => {
      const jobs = new Map(state.activeJobs);
      const job = jobs.get(id);
      if (job) {
        jobs.set(id, { ...job, status });
      }
      return { activeJobs: jobs };
    }),

  updateJobProgress: (id, progress) =>
    set((state) => {
      const jobs = new Map(state.activeJobs);
      const job = jobs.get(id);
      if (job) {
        jobs.set(id, { ...job, progress });
      }
      return { activeJobs: jobs };
    }),

  removeJob: (id) =>
    set((state) => {
      const jobs = new Map(state.activeJobs);
      jobs.delete(id);
      const runtime = new Map(state.jobRuntime);
      runtime.delete(id);
      return { activeJobs: jobs, jobRuntime: runtime };
    }),

  setCurrentJob: (id) => set({ currentJobId: id }),

  getRunningJob: () => {
    const jobs = Array.from(get().activeJobs.values());
    return jobs.find((j) => j.status === 'running') || null;
  },

  getQueuedJobs: () => {
    const jobs = Array.from(get().activeJobs.values());
    return jobs.filter((j) => j.status === 'queued');
  },

  syncJobsFromBackend: (jobsPayload) =>
    set((state) => {
      const jobs = new Map(state.activeJobs);
      const runtime = new Map(state.jobRuntime);

      for (const raw of jobsPayload) {
        const jobId = String(raw.job_id ?? '');
        if (!jobId) continue;

        const status = (raw.status as JobStatus) ?? 'queued';
        const requestedGpu = typeof raw.requested_gpu === 'number' ? raw.requested_gpu : undefined;
        const assignedGpu = typeof raw.assigned_gpu === 'number' ? raw.assigned_gpu : undefined;
        const externalGpuBusy = raw.external_gpu_busy === true;
        const error = typeof raw.error === 'string' ? raw.error : undefined;
        const startTime = typeof raw.start_time === 'string' ? raw.start_time : undefined;
        const endTime = typeof raw.end_time === 'string' ? raw.end_time : undefined;
        const progressRaw = raw.progress as any;
        const progress =
          progressRaw && typeof progressRaw.stage === 'string' && typeof progressRaw.percent === 'number'
            ? ({
                job_id: jobId,
                stage: progressRaw.stage as PipelineStage,
                percent: progressRaw.percent as number,
              } satisfies ProgressInfo)
            : undefined;

        const existing = jobs.get(jobId);
        jobs.set(jobId, {
          id: jobId,
          config: existing?.config ?? null,
          status,
          startTime: startTime ?? existing?.startTime,
          endTime: endTime ?? existing?.endTime,
          progress: progress ?? existing?.progress,
          error: error ?? existing?.error,
          requestedGpu: requestedGpu ?? existing?.requestedGpu,
          assignedGpu: assignedGpu ?? existing?.assignedGpu,
          externalGpuBusy: externalGpuBusy ?? existing?.externalGpuBusy,
        });

        const parsedStart = startTime ? Date.parse(startTime) : NaN;
        const inferredStart = Number.isFinite(parsedStart) ? parsedStart : Date.now();

        if (!runtime.has(jobId)) {
          const rt = createInitialJobRuntime();
          runtime.set(jobId, status === 'running' ? { ...rt, startedAtMs: inferredStart } : rt);
        } else {
          const rt = runtime.get(jobId)!;
          if (rt.startedAtMs == null && status === 'running') {
            runtime.set(jobId, { ...rt, startedAtMs: inferredStart });
          }
        }
      }

      const anyActive = Array.from(jobs.values()).some((j) => j.status === 'running' || j.status === 'queued');
      return { activeJobs: jobs, jobRuntime: runtime, isRunning: anyActive };
    }),

  applyProgressEvent: (event) =>
    set((state) => {
      const jobs = new Map(state.activeJobs);
      const runtime = new Map(state.jobRuntime);

      const jobId = event.job_id;
      if (!jobId) {
        return state;
      }

      const existingJob = jobs.get(jobId);
      const nextJob: ActiveJob = existingJob ?? {
        id: jobId,
        config: null,
        status: 'running',
      };

      // Initialize runtime state
      const rt = runtime.get(jobId) ?? createInitialJobRuntime();
      const nextRt: JobRuntimeState = {
        ...rt,
        stages: { ...rt.stages },
        batch: { ...rt.batch, ligandStatuses: [...rt.batch.ligandStatuses] },
        poses: [...rt.poses],
      };

      if (nextRt.startedAtMs == null) {
        nextRt.startedAtMs = Date.now();
      }

      // Update based on event type
      if (event.type === 'batch_start') {
        nextRt.batch.isBatch = true;
        nextRt.batch.totalLigands = event.total_ligands ?? nextRt.batch.totalLigands;
        nextRt.batch.ligandStatuses = (event.ligand_statuses as BatchLigandStatus[]) ?? [];
      } else if (event.type === 'batch_progress') {
        nextRt.batch.isBatch = true;
        nextRt.batch.currentLigandIndex = event.ligand_index ?? nextRt.batch.currentLigandIndex;
        nextRt.batch.totalLigands = event.total_ligands ?? nextRt.batch.totalLigands;
        nextRt.batch.ligandStatuses = (event.ligand_statuses as BatchLigandStatus[]) ?? nextRt.batch.ligandStatuses;
      } else if (event.type === 'ligand_start') {
        nextRt.batch.isBatch = true;
        nextRt.batch.currentLigand = event.current_ligand ?? null;
        nextRt.batch.currentLigandIndex = event.ligand_index ?? nextRt.batch.currentLigandIndex;
        nextRt.batch.totalLigands = event.total_ligands ?? nextRt.batch.totalLigands;
        nextRt.batch.ligandStatuses = (event.ligand_statuses as BatchLigandStatus[]) ?? nextRt.batch.ligandStatuses;
        nextRt.stages = createInitialStages();
        nextRt.poses = [];
      } else if (event.type === 'ligand_done') {
        nextRt.batch.isBatch = true;
        nextRt.batch.ligandStatuses = (event.ligand_statuses as BatchLigandStatus[]) ?? nextRt.batch.ligandStatuses;
      } else if (event.type === 'stage_start' && event.stage) {
        nextRt.stages[event.stage] = {
          status: 'running',
          progress: 0,
          message: event.name,
        };
      } else if (event.type === 'stage_progress' && event.stage) {
        const prev = nextRt.stages[event.stage] ?? { status: 'running', progress: 0 };
        nextRt.stages[event.stage] = {
          ...prev,
          status: 'running',
          progress: event.progress ?? prev.progress,
          message: event.message ?? prev.message,
        };
      } else if (event.type === 'stage_done' && event.stage) {
        nextRt.stages[event.stage] = {
          status: 'done',
          progress: 100,
          elapsed: event.elapsed,
        };
      } else if (event.type === 'poses_update' && event.poses) {
        nextRt.poses = event.poses as PoseResult[];
      } else if (event.type === 'job_done') {
        nextJob.status = 'completed';
        nextJob.endTime = new Date().toISOString();
        nextRt.completion = {
          outputPath: event.output_path ?? '',
          poses: (event.poses as PoseResult[]) ?? [],
          totalLigands: event.total_ligands,
          ligandStatuses: event.ligand_statuses as BatchLigandStatus[] | undefined,
        };
      } else if (event.type === 'cancelled') {
        nextJob.status = 'cancelled';
        nextJob.endTime = new Date().toISOString();
      } else if (event.type === 'error') {
        nextJob.status = 'failed';
        nextJob.error = event.message ?? 'Unknown error';
        nextJob.endTime = new Date().toISOString();
      }

      // Progress summary for job list
      if ((event.type === 'stage_start' || event.type === 'stage_progress' || event.type === 'stage_done') && event.stage) {
        const percent =
          event.type === 'stage_done' ? 100 : event.type === 'stage_progress' ? (event.progress ?? 0) : 0;
        nextJob.progress = {
          job_id: jobId,
          stage: event.stage as PipelineStage,
          percent,
        };
      }

      jobs.set(jobId, nextJob);
      runtime.set(jobId, nextRt);

      const anyActive = Array.from(jobs.values()).some((j) => j.status === 'running' || j.status === 'queued');
      return { ...state, activeJobs: jobs, jobRuntime: runtime, isRunning: anyActive };
    }),

  // Results
  results: null,
  setResults: (results) => set({ results }),
  resultsUI: { ...defaultResultsUI },
  setResultsUI: (partial) =>
    set((state) => ({
      resultsUI: { ...state.resultsUI, ...partial },
    })),
  resetResultsUI: (next) =>
    set({
      resultsUI: { ...defaultResultsUI, ...(next ?? {}) },
    }),

  // History
  history: [],
  addToHistory: (job) =>
    set((state) => ({
      history: [job, ...state.history].slice(0, STORE_CONFIG.MAX_HISTORY),
    })),
  removeFromHistory: (id) =>
    set((state) => ({
      history: state.history.filter((j) => j.id !== id),
    })),
  clearHistory: () => set({ history: [] }),
  historySelectedIndex: 0,
  setHistorySelectedIndex: (index) => set({ historySelectedIndex: index }),

  // UI
  error: null,
  setError: (error) => set({ error }),
  notification: null,
  setNotification: (notification) => set({ notification }),
  modalOpen: false,
  setModalOpen: (open) => set({ modalOpen: open }),

  // Debug mode
  debugMode: process.env.MATCHA_DEBUG === '1',
  navigationHistory: [],
  debugLogs: [],
  trackNavigation: (from, to) =>
    set((state) => ({
      navigationHistory: [...state.navigationHistory, `${from} → ${to}`].slice(-STORE_CONFIG.MAX_NAV_HISTORY),
    })),
  addDebugLog: (log) =>
    set((state) => ({
      debugLogs: [
        ...state.debugLogs.slice(-STORE_CONFIG.MAX_DEBUG_LOGS),
        {
          ...log,
          timestamp: new Date().toISOString(),
        },
      ],
    })),
  clearDebugLogs: () => set({ debugLogs: [], navigationHistory: [] }),
}));
