/**
 * Running Screen - shows docking progress with batch monitoring support.
 */

import React, { useEffect, useState, useRef, useReducer, useCallback } from 'react';
import { Box, Text, useInput } from 'ink';
import path from 'path';
import { useStore } from '../store/index.js';
import { getBridge, initBridge, closeBridge, type ProgressEvent } from '../services/index.js';
import { ProgressBar } from '../components/ProgressBar.js';
import { icons, getStageColor, getStatusColor } from '../utils/colors.js';
import { PIPELINE_STAGES, type PipelineStage, type PoseResult, type JobConfig } from '../types/index.js';
import { formatDuration, generateRunName } from '../utils/format.js';
import { isKey } from '../utils/keyboard.js';
import { logger } from '../utils/logger.js';

interface StageStatus {
  status: 'pending' | 'running' | 'done' | 'error';
  progress: number;
  elapsed?: number;
  message?: string;
}

interface LigandStatus {
  name: string;
  path: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  pb_count?: number;
  gnina_score?: number;
  error_message?: string;
}

interface BatchState {
  isBatch: boolean;
  totalLigands: number;
  currentLigandIndex: number;
  currentLigand: string | null;
  ligandStatuses: LigandStatus[];
}

type BatchAction =
  | { type: 'BATCH_START'; totalLigands: number; ligandStatuses: LigandStatus[] }
  | { type: 'BATCH_PROGRESS'; ligandIndex: number; ligandStatuses: LigandStatus[] }
  | { type: 'LIGAND_START'; currentLigand: string; ligandIndex: number; totalLigands: number; ligandStatuses: LigandStatus[] }
  | { type: 'LIGAND_DONE'; ligandStatuses: LigandStatus[] }
  | { type: 'RESET' };

const initialBatchState: BatchState = {
  isBatch: false,
  totalLigands: 1,
  currentLigandIndex: 0,
  currentLigand: null,
  ligandStatuses: [],
};

function batchReducer(state: BatchState, action: BatchAction): BatchState {
  switch (action.type) {
    case 'BATCH_START':
      return {
        ...state,
        isBatch: true,
        totalLigands: action.totalLigands,
        ligandStatuses: action.ligandStatuses,
      };
    case 'BATCH_PROGRESS':
      return {
        ...state,
        currentLigandIndex: action.ligandIndex,
        ligandStatuses: action.ligandStatuses,
      };
    case 'LIGAND_START':
      return {
        ...state,
        currentLigand: action.currentLigand,
        currentLigandIndex: action.ligandIndex,
        totalLigands: action.totalLigands,
        ligandStatuses: action.ligandStatuses,
      };
    case 'LIGAND_DONE':
      return {
        ...state,
        ligandStatuses: action.ligandStatuses,
      };
    case 'RESET':
      return initialBatchState;
    default:
      return state;
  }
}

export function RunningScreen(): React.ReactElement {
  const screen = useStore((s) => s.screen);
  const setScreen = useStore((s) => s.setScreen);
  const setRunning = useStore((s) => s.setRunning);
  const setResults = useStore((s) => s.setResults);
  const setError = useStore((s) => s.setError);
  const setNotification = useStore((s) => s.setNotification);
  const addLog = useStore((s) => s.addLog);
  const jobConfig = useStore((s) => s.jobConfig);
  const debugMode = useStore((s) => s.debugMode);
  const addDebugLog = useStore((s) => s.addDebugLog);

  // Multi-job support
  const currentJobId = useStore((s) => s.currentJobId);
  const setCurrentJob = useStore((s) => s.setCurrentJob);
  const activeJobs = useStore((s) => s.activeJobs);
  const addJob = useStore((s) => s.addJob);
  const updateJobStatus = useStore((s) => s.updateJobStatus);
  const updateJobProgress = useStore((s) => s.updateJobProgress);
  const jobIdRef = useRef<string | null>(currentJobId);

  // Initialize all stages with pending status to avoid race conditions
  const [stages, setStages] = useState<Record<string, StageStatus>>(() => {
    const initial: Record<string, StageStatus> = {};
    PIPELINE_STAGES.forEach(stage => {
      initial[stage.id] = { status: 'pending', progress: 0 };
    });
    return initial;
  });
  const [poses, setPoses] = useState<PoseResult[]>([]);
  const [startTime] = useState(Date.now());
  const [elapsedTime, setElapsedTime] = useState(0);
  const jobStartedRef = useRef(false);
  const [bridgeReady, setBridgeReady] = useState(false);

  // Batch mode state
  const [batchState, dispatchBatch] = useReducer(batchReducer, initialBatchState);
  const { isBatch, totalLigands, currentLigandIndex, currentLigand, ligandStatuses } = batchState;
  const [batchProgress, setBatchProgress] = useState(0);
  const stageOrder = ['init', 'checkpoints', 'dataset', 'esm', 'stage1', 'stage2', 'stage3', 'scoring', 'posebusters'] as const;
  const stageWeights: Record<string, number> = {
    init: 1,
    checkpoints: 1,
    dataset: 1,
    esm: 1,
    stage1: 1,
    stage2: 1,
    stage3: 1,
    scoring: 1,
    posebusters: 1,
  };

  useEffect(() => {
    const timer = setInterval(() => {
      setElapsedTime(Math.floor((Date.now() - startTime) / 1000));
    }, 1000);
    return () => clearInterval(timer);
  }, [startTime]);

  // Compute overall batch progress from stage progress
  useEffect(() => {
    if (!isBatch) return;
    let totalWeight = 0;
    let weightedProgress = 0;

    for (const stageId of stageOrder) {
      const weight = stageWeights[stageId] ?? 1;
      totalWeight += weight;
      const status = stages[stageId];
      if (!status) {
        continue;
      }
      let percent = 0;
      if (status.status === 'done') {
        percent = 100;
      } else if (status.status === 'running') {
        percent = status.progress ?? 0;
      }
      weightedProgress += weight * percent;
    }

    if (totalWeight > 0) {
      setBatchProgress(Math.round(weightedProgress / totalWeight));
    }
  }, [isBatch, stages]);

  // Progress event handler - shared between new job and resume
  const handleProgress = useCallback((event: ProgressEvent) => {
    // Filter events by job_id - only process events for current job
    const targetJobId = jobIdRef.current;
    if (targetJobId && event.job_id && event.job_id !== targetJobId) {
      return;
    }

    addLog(JSON.stringify(event));

    // Batch events
    if (event.type === 'batch_start') {
      dispatchBatch({
        type: 'BATCH_START',
        totalLigands: event.total_ligands ?? 1,
        ligandStatuses: (event.ligand_statuses as LigandStatus[]) ?? [],
      });
    } else if (event.type === 'batch_progress') {
      dispatchBatch({
        type: 'BATCH_PROGRESS',
        ligandIndex: event.ligand_index ?? 0,
        ligandStatuses: (event.ligand_statuses as LigandStatus[]) ?? [],
      });
    } else if (event.type === 'ligand_start') {
      dispatchBatch({
        type: 'LIGAND_START',
        currentLigand: event.current_ligand ?? '',
        ligandIndex: event.ligand_index ?? 0,
        totalLigands: event.total_ligands ?? 1,
        ligandStatuses: (event.ligand_statuses as LigandStatus[]) ?? [],
      });
      // Reset stages for new ligand
      setStages({});
    } else if (event.type === 'ligand_done') {
      dispatchBatch({
        type: 'LIGAND_DONE',
        ligandStatuses: (event.ligand_statuses as LigandStatus[]) ?? [],
      });
    } else if (event.type === 'stage_start' && event.stage) {
      setStages((prev) => ({
        ...prev,
        [event.stage!]: { status: 'running', progress: 0, message: event.name },
      }));
    } else if (event.type === 'stage_progress' && event.stage) {
      setStages((prev) => ({
        ...prev,
        [event.stage!]: { ...prev[event.stage!], progress: event.progress ?? 0, message: event.message },
      }));
    } else if (event.type === 'stage_done' && event.stage) {
      setStages((prev) => ({
        ...prev,
        [event.stage!]: { status: 'done', progress: 100, elapsed: event.elapsed },
      }));
    } else if (event.type === 'poses_update' && event.poses) {
      setPoses(event.poses as PoseResult[]);
    } else if (event.type === 'job_done') {
      setRunning(false);
      const jid = jobIdRef.current;
      if (jid) updateJobStatus(jid, 'completed');

      // Use event.poses from backend (aggregated data for all ligands)
      const finalPoses = (event.poses ?? []) as PoseResult[];
      const runDir = event.output_path ?? '';
      const runName = jobConfig.params?.runName ?? 'matcha_tui_run';
      const isBatchRun =
        (event.total_ligands ?? 0) > 1 || jobConfig.mode === 'batch';

      // Check if all ligands failed
      const evLigandStatuses = event.ligand_statuses as LigandStatus[] | undefined;
      const completedCount = evLigandStatuses?.filter(l => l.status === 'completed').length ?? 0;
      const failedCount = evLigandStatuses?.filter(l => l.status === 'failed').length ?? 0;
      const allFailed = failedCount > 0 && completedCount === 0;

      // If all failed, show error and go to welcome
      if (allFailed) {
        const firstError = evLigandStatuses?.find(l => l.error_message)?.error_message;
        setError(`Docking failed: ${firstError ?? 'All ligands failed'}`);
        setScreen('welcome');
        return;
      }

      const bestPosePath = isBatchRun
        ? path.join(runDir, 'best_poses')
        : path.join(runDir, `${runName}_best.sdf`);
      const allPosesPath = isBatchRun
        ? path.join(runDir, 'all_poses')
        : path.join(runDir, `${runName}_poses.sdf`);
      const logPath = path.join(runDir, `${runName}.log`);

      const results = {
        runName,
        runtime: (Date.now() - startTime) / 1000,
        totalPoses: finalPoses.length,
        physicalPoses: finalPoses.filter((p) => p.pb_count === 4).length,
        poses: finalPoses,
        bestPosePath,
        allPosesPath,
        logPath,
        receptor: jobConfig.receptor,
        ligand: jobConfig.mode === 'batch' ? jobConfig.ligandDir : jobConfig.ligand,
        totalLigands: event.total_ligands,
        ligandStatuses: evLigandStatuses,
      };

      setResults(results);

      // Check if user is still on running screen
      const currentScreen = useStore.getState().screen;
      if (currentScreen === 'running') {
        setScreen('results');
      } else {
        setNotification('Docking completed! View results in History.');
      }
    } else if (event.type === 'error') {
      setError(event.message ?? 'Unknown error');
      setRunning(false);
      const jid = jobIdRef.current;
      if (jid) updateJobStatus(jid, 'failed');
    } else if (event.type === 'cancelled') {
      setRunning(false);
      setScreen('welcome');
      const jid = jobIdRef.current;
      if (jid) updateJobStatus(jid, 'failed');
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Start docking job on mount, or resume if returning to existing job
  useEffect(() => {
    if (jobStartedRef.current) return;
    jobStartedRef.current = true;

    // Check if we're resuming an existing job (e.g. user pressed 'r' to return)
    const storeState = useStore.getState();
    const existingJobId = storeState.currentJobId;
    const existingJob = existingJobId ? storeState.activeJobs.get(existingJobId) : null;
    const isResuming = existingJob != null && existingJob.status === 'running';

    if (isResuming) {
      // Resume: just re-subscribe to bridge progress events
      jobIdRef.current = existingJobId;
      setRunning(true);
      try {
        const bridge = getBridge();
        if (bridge.isReady()) {
          bridge.on('progress', handleProgress);
          setBridgeReady(true);
          addLog(`Resumed monitoring job ${existingJobId}`);
        }
      } catch {
        addLog('Bridge not available for resume');
      }
      return;
    }

    // New job: initialize bridge and start docking
    setRunning(true);

    // Build config from jobConfig
    const config = {
      receptor: jobConfig.receptor || '',
      ligand: jobConfig.mode === 'single' ? jobConfig.ligand : undefined,
      ligand_dir: jobConfig.mode === 'batch' ? jobConfig.ligandDir : undefined,
      output_dir: jobConfig.params?.outputDir || './results',
      run_name: jobConfig.params?.runName || generateRunName(),
      n_samples: jobConfig.params?.nSamples || 40,
      n_confs: jobConfig.params?.nConfs,
      gpu: jobConfig.params?.gpu,
      checkpoints: jobConfig.params?.checkpointsDir,
      physical_only: jobConfig.params?.physicalOnly || false,
      box_mode: jobConfig.box?.mode || 'blind',
      center_x: jobConfig.box?.centerX,
      center_y: jobConfig.box?.centerY,
      center_z: jobConfig.box?.centerZ,
      autobox_ligand: jobConfig.box?.autoboxLigand,
    };

    addLog(`Starting docking with config: ${JSON.stringify(config)}`);

    const startJob = async () => {
      try {
        addLog('Initializing Python backend...');
        const projectRoot = process.env.MATCHA_ROOT || path.join(process.cwd(), '..');
        addLog(`Project root: ${projectRoot}`);

        const bridge = await initBridge({
          projectRoot,
          useUv: true,
        });

        // Listen for stderr to help with debugging
        bridge.on('stderr', (data: string) => {
          addLog(`[stderr] ${data}`);
        });

        // Listen for debug events from backend
        bridge.on('debug', (event: any) => {
          if (debugMode) {
            addDebugLog({
              level: event.level,
              component: event.component,
              message: event.message,
              data: event.data,
            });
            const logMethod = event.level as 'debug' | 'info' | 'warn' | 'error';
            logger[logMethod](event.component, event.message, event.data);
            addLog(`[${event.level}] [${event.component}] ${event.message}`);
          }
        });

        bridge.on('error', (err: Error) => {
          addLog(`[bridge error] ${err.message}`);
        });

        bridge.on('exit', (code: number) => {
          addLog(`[backend exit] code ${code}`);
        });

        // Subscribe to progress BEFORE starting docking
        bridge.on('progress', handleProgress);

        addLog('Backend initialized, starting docking...');
        setBridgeReady(true);

        // Start docking
        const { job_id } = await bridge.startDocking(config);

        // Save job_id for tracking
        jobIdRef.current = job_id;
        setCurrentJob(job_id);
        addJob({
          id: job_id,
          config: jobConfig as JobConfig,
          status: 'running',
          startTime: new Date().toISOString(),
        });
        addLog(`Job started with ID: ${job_id}`);
      } catch (err) {
        const error = err as Error;
        addLog(`Error: ${error.message}`);
        setError(`Failed to start docking: ${error.message}`);
        setRunning(false);
        setScreen('welcome');
      }
    };

    startJob();

    // Cleanup: remove progress listener on unmount (job continues in background)
    return () => {
      try {
        const bridge = getBridge();
        bridge.removeListener('progress', handleProgress);
      } catch {
        // Bridge may already be closed
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useInput((input, key) => {
    if (isKey(input, 'c') && bridgeReady && currentJobId) {
      getBridge().cancelJob(currentJobId).catch(() => {});
    } else if (isKey(input, 'h') || key.escape || key.leftArrow) {
      // Allow user to leave running screen and go back to home
      // Job continues in background
      setScreen('welcome');
      return;
    }
  });

  return (
    <Box flexDirection="column" gap={1}>
      {/* Compact status line */}
      <Box>
        <Text color="yellow">{icons.running} </Text>
        <Text color="white" bold>Docking</Text>
        <Text color="gray" dimColor> • </Text>
        <Text color="gray" dimColor>{formatDuration(elapsedTime)}</Text>
        {isBatch && (
          <>
            <Text color="gray" dimColor> • </Text>
            <Text color="magenta">Ligand </Text>
            <Text color="white">{currentLigandIndex + 1}/{totalLigands}</Text>
          </>
        )}
        <Text color="gray" dimColor> • </Text>
        <Text color="#D0D1FA" dimColor>h/Esc</Text>
        <Text color="gray" dimColor> run in background</Text>
      </Box>

      {/* Batch progress (if batch mode) */}
      {isBatch && (
        <Box flexDirection="column" marginTop={1}>
          <Text bold color="white">Batch Progress</Text>
          <Box marginTop={1}>
            <ProgressBar percent={batchProgress} width={40} showPercent />
          </Box>
        </Box>
      )}

      {/* Current ligand info (if batch mode) */}
      {isBatch && currentLigand && (
        <Box marginTop={1}>
          <Text color="magenta">Current: </Text>
          <Text color="yellow">{currentLigand}</Text>
        </Box>
      )}

      {/* Pipeline stages */}
      <Box flexDirection="column" marginTop={1}>
        <Text bold color="white">Pipeline Progress</Text>
        <Box flexDirection="column" marginTop={1}>
          {PIPELINE_STAGES.map((stageInfo) => (
            <StageRow key={stageInfo.id} stage={stageInfo} status={stages[stageInfo.id]} />
          ))}
        </Box>
      </Box>

      {/* Current poses for this ligand */}
      {poses.length > 0 && (
        <Box flexDirection="column" marginTop={1}>
          <Text bold color="white">Top Poses {currentLigand ? `(${currentLigand})` : ''}</Text>
          <Box flexDirection="column" marginTop={1}>
            {poses.slice(0, 5).map((pose) => (
              <Box key={pose.rank}>
                <Text color="white">{String(pose.rank).padEnd(6)}</Text>
                <Text color={pose.pb_count === 4 ? 'green' : 'yellow'}>{`${pose.pb_count ?? 0}/4`.padEnd(8)}</Text>
                {pose.gnina_score != null && (
                  <Text color="#D0D1FA">{pose.gnina_score.toFixed(2)}</Text>
                )}
              </Box>
            ))}
          </Box>
        </Box>
      )}

      {/* Ligand status list (if batch mode, scrollable) */}
      {isBatch && ligandStatuses.length > 0 && (
        <Box flexDirection="column" marginTop={1}>
          <Text bold color="white">Ligand Status</Text>
          <Box flexDirection="column" marginTop={1} height={Math.min(8, ligandStatuses.length)}>
            {ligandStatuses.slice(0, 8).map((lig, idx) => (
              <LigandRow key={idx} ligand={lig} isCurrent={idx === currentLigandIndex} />
            ))}
            {ligandStatuses.length > 8 && (
              <Text color="gray">... and {ligandStatuses.length - 8} more</Text>
            )}
          </Box>
        </Box>
      )}

    </Box>
  );
}

function StageRow({ stage, status }: { stage: { id: PipelineStage; name: string }; status?: StageStatus }): React.ReactElement {
  const icon = status?.status === 'done' ? icons.check : status?.status === 'running' ? icons.running : icons.pending;
  const color = status?.status ? getStageColor(status.status as 'pending' | 'running' | 'done') : 'gray';

  return (
    <Box>
      <Text color={color}>{icon} </Text>
      <Text color={color} bold={status?.status === 'running'}>{stage.name.padEnd(20)}</Text>
      {status?.status === 'running' && (
        <Box marginLeft={1} width={30}>
          <ProgressBar percent={status.progress} width={20} showPercent />
        </Box>
      )}
      {status?.status === 'done' && status.elapsed !== undefined && (
        <Text color="gray"> {formatDuration(Math.round(status.elapsed))}</Text>
      )}
    </Box>
  );
}

function LigandRow({ ligand, isCurrent }: { ligand: LigandStatus; isCurrent: boolean }): React.ReactElement {
  const statusIcon = {
    pending: icons.pending,
    running: icons.running,
    completed: icons.check,
    failed: icons.error,
  }[ligand.status];

  const statusColor = getStatusColor(ligand.status);

  return (
    <Box>
      <Text color={statusColor}>{statusIcon} </Text>
      <Text color={isCurrent ? 'magenta' : 'white'} bold={isCurrent}>
        {ligand.name.substring(0, 25).padEnd(25)}
      </Text>
      {ligand.status === 'completed' && ligand.pb_count !== undefined && (
        <>
          <Text color="gray"> pb: </Text>
          <Text color={ligand.pb_count === 4 ? 'green' : 'yellow'}>{ligand.pb_count}/4</Text>
          {ligand.gnina_score != null && (
            <>
              <Text color="gray"> aff: </Text>
              <Text color="#D0D1FA">{ligand.gnina_score.toFixed(2)}</Text>
            </>
          )}
        </>
      )}
      {ligand.status === 'failed' && ligand.error_message && (
        <Text color="red"> {ligand.error_message.substring(0, 30)}</Text>
      )}
    </Box>
  );
}
