import { create } from 'zustand';
import type { Screen, JobConfig, ProgressInfo, JobResults, StoredJob } from '../types/index.js';

export type LogLevel = 'debug' | 'info' | 'warn' | 'error';

export interface DebugLog {
  timestamp: string;
  level: LogLevel;
  component: string;
  message: string;
  data?: any;
}

interface AppState {
  // Navigation
  screen: Screen;
  setScreen: (screen: Screen) => void;

  // Current job configuration
  jobConfig: Partial<JobConfig>;
  setJobConfig: (config: Partial<JobConfig>) => void;
  resetJobConfig: () => void;

  // Running job state
  isRunning: boolean;
  progress: ProgressInfo | null;
  logs: string[];
  setRunning: (running: boolean) => void;
  setProgress: (progress: ProgressInfo | null) => void;
  addLog: (log: string) => void;
  clearLogs: () => void;

  // Results
  results: JobResults | null;
  setResults: (results: JobResults | null) => void;

  // Job history
  history: StoredJob[];
  addToHistory: (job: StoredJob) => void;
  removeFromHistory: (id: string) => void;
  clearHistory: () => void;

  // UI state
  error: string | null;
  setError: (error: string | null) => void;

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

export const useStore = create<AppState>((set) => ({
  // Navigation
  screen: 'welcome',
  setScreen: (screen) => set({ screen }),

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
      logs: [...state.logs.slice(-500), log], // Keep last 500 logs
    })),
  clearLogs: () => set({ logs: [] }),

  // Results
  results: null,
  setResults: (results) => set({ results }),

  // History
  history: [],
  addToHistory: (job) =>
    set((state) => ({
      history: [job, ...state.history].slice(0, 50), // Keep last 50 jobs
    })),
  removeFromHistory: (id) =>
    set((state) => ({
      history: state.history.filter((j) => j.id !== id),
    })),
  clearHistory: () => set({ history: [] }),

  // UI
  error: null,
  setError: (error) => set({ error }),

  // Debug mode
  debugMode: process.env.MATCHA_DEBUG === '1',
  navigationHistory: [],
  debugLogs: [],
  trackNavigation: (from, to) =>
    set((state) => ({
      navigationHistory: [...state.navigationHistory, `${from} → ${to}`].slice(-100), // Keep last 100 transitions
    })),
  addDebugLog: (log) =>
    set((state) => ({
      debugLogs: [
        ...state.debugLogs.slice(-1000), // Keep last 1000 debug logs
        {
          ...log,
          timestamp: new Date().toISOString(),
        },
      ],
    })),
  clearDebugLogs: () => set({ debugLogs: [], navigationHistory: [] }),
}));
