import { create } from 'zustand';
import type { Screen, JobConfig, ProgressInfo, JobResults, StoredJob, ActiveJob, JobStatus } from '../types/index.js';

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
  addJob: (job: ActiveJob) => void;
  updateJobStatus: (id: string, status: JobStatus) => void;
  updateJobProgress: (id: string, progress: ProgressInfo) => void;
  removeJob: (id: string) => void;
  setCurrentJob: (id: string | null) => void;
  getRunningJob: () => ActiveJob | null;
  getQueuedJobs: () => ActiveJob[];

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

  addJob: (job) =>
    set((state) => {
      const jobs = new Map(state.activeJobs);
      jobs.set(job.id, job);
      return { activeJobs: jobs };
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
      return { activeJobs: jobs };
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
