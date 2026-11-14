/**
 * Running Screen - shows docking progress with batch monitoring support.
 */

import React, { useEffect, useState, useRef, useReducer } from 'react';
import { Box, Text, useInput } from 'ink';
import path from 'path';
import { useStore } from '../store/index.js';
import { getBridge, initBridge, closeBridge, type ProgressEvent } from '../services/index.js';
import { ProgressBar } from '../components/ProgressBar.js';
import { icons, getStageColor, getStatusColor } from '../utils/colors.js';
import { PIPELINE_STAGES, type PipelineStage, type PoseResult, type ProgressInfo } from '../types/index.js';
import { formatDuration, generateRunName } from '../utils/format.js';
import { isKey } from '../utils/keyboard.js';
import { logger, type LogLevel } from '../utils/logger.js';
import { TABLE_COLUMN_WIDTHS, TEXT_LIMITS, UI_DIMENSIONS } from '../constants/ui.js';

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
  error_estimate?: number;
  pb_count?: number;
  error_message?: string;
}

interface DebugEventData {
  level: LogLevel;
  component: string;
  message: string;
  data?: Record<string, unknown>;
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
  const updateJobProgress = useStore((s) => s.updateJobProgress);
  const updateJobStatus = useStore((s) => s.updateJobStatus);
  const pendingJobConfig = useStore((s) => s.pendingJobConfig);
  const setPendingJobConfig = useStore((s) => s.setPendingJobConfig);

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

  // Restore state when resuming a job
  useEffect(() => {
    if (currentJobId && activeJobs.has(currentJobId)) {
      const job = activeJobs.get(currentJobId)!;

      // Restore running state
      if (job.status === 'running') {
        setRunning(true);
      }

      // Restore progress if available
      if (job.progress) {
        // Note: stages are local to this component and will be updated by incoming events
        // We could restore them from somewhere else if needed
        addLog(`Restored job ${currentJobId}, status: ${job.status}`);
      }
    }
  }, []); // Run once on mount

  // Start docking job on mount
  useEffect(() => {
    // If there's already a currentJobId in store, we're resuming an existing job
    // Don't start a new one, just re-subscribe to events
    const isResumingJob = !!currentJobId;

    if (jobStartedRef.current) return;
    jobStartedRef.current = true;

    if (!isResumingJob) {
      setRunning(true);
    }

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

    // Initialize bridge first, then start docking
    const startOrResumeJob = async () => {
      try {
        // Get existing bridge or initialize if needed
        let bridge;
        let stderrBuffer = ''; // Capture all stderr

        try {
          bridge = getBridge();
          if (!bridge.isReady()) {
            throw new Error('Bridge not ready');
          }
          addLog('Resuming existing job...');
        } catch {
          // Initialize Python backend
          addLog('Initializing Python backend...');
          // TUI is in matcha-tui/ subdirectory, but Python package is in parent directory
          // Use parent directory as project root so 'uv run python -m matcha.tui.backend' can find the package
          const projectRoot = process.env.MATCHA_ROOT || path.join(process.cwd(), '..');
          addLog(`Project root: ${projectRoot}`);

          bridge = await initBridge({
            projectRoot,
            useUv: true,
          });
        }

        // Listen for stderr to help with debugging
        bridge.on('stderr', (data: string) => {
          stderrBuffer += data;
          addLog(`[stderr] ${data}`);
        });

        // Listen for debug events from backend
        bridge.on('debug', (event: DebugEventData) => {
          if (debugMode) {
            // Add to store for UI display
            addDebugLog({
              level: event.level,
              component: event.component,
              message: event.message,
              data: event.data,
            });

            // Write to file via logger
            const logMethod = event.level as 'debug' | 'info' | 'warn' | 'error';
            logger[logMethod](event.component, event.message, event.data);

            // Also add to running logs display
            addLog(`[${event.level}] [${event.component}] ${event.message}`);
          }
        });

        bridge.on('error', (err: Error) => {
          addLog(`[bridge error] ${err.message}`);
        });

        bridge.on('exit', (code: number) => {
          addLog(`[backend exit] code ${code}`);
        });

        // IMPORTANT: Subscribe to progress BEFORE starting docking!
        bridge.on('progress', (event: ProgressEvent) => {
          // Filter events by job_id - only process events for current job
          if (currentJobId && event.job_id !== currentJobId) {
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

            // Update progress in activeJobs
            if (event.job_id && event.stage && event.progress !== undefined) {
              const progressInfo: ProgressInfo = {
                job_id: event.job_id,
                stage: event.stage as PipelineStage,
                percent: event.progress,
                currentLigand: event.current_ligand,
              };
              updateJobProgress(event.job_id, progressInfo);
            }
          } else if (event.type === 'stage_done' && event.stage) {
            setStages((prev) => ({
              ...prev,
              [event.stage!]: { status: 'done', progress: 100, elapsed: event.elapsed },
            }));
          } else if (event.type === 'poses_update' && event.poses) {
            setPoses(event.poses as PoseResult[]);
          } else if (event.type === 'job_done') {
            setRunning(false);
            if (isBatch) {
              setBatchProgress(100);
            }

            // Update job status in activeJobs
            if (event.job_id) {
              updateJobStatus(event.job_id, 'completed');
            }
            // Use event.poses from backend (aggregated data for all ligands)
            // Local 'poses' state is only for displaying current ligand during execution
            const finalPoses = (event.poses ?? []) as PoseResult[];
            const runDir = event.output_path ?? '';
            const runName = jobConfig.params?.runName ?? 'matcha_tui_run';
            const isBatchRun =
              (event.total_ligands ?? 0) > 1 || jobConfig.mode === 'batch';
            const bestPosePath = isBatchRun
              ? path.join(runDir, 'best_poses')
              : path.join(runDir, `${runName}_best.sdf`);
            const allPosesPath = isBatchRun
              ? path.join(runDir, 'all_poses')
              : path.join(runDir, `${runName}_poses.sdf`);
            const logPath = path.join(runDir, `${runName}.log`);

            // DEBUG: Log received poses
            console.error('[DEBUG] job_done event.poses count:', finalPoses.length);
            if (finalPoses.length > 0) {
              console.error('[DEBUG] First pose:', JSON.stringify(finalPoses[0], null, 2));
            }

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
              ligand: jobConfig.ligand,
              totalLigands: event.total_ligands,
              ligandStatuses: event.ligand_statuses as LigandStatus[] | undefined,
            };

            setResults(results);

            // Check if user is still on running screen
            // If they navigated away, show notification instead of auto-navigating
            if (screen === 'running') {
              setScreen('results');
            } else {
              setNotification('✓ Docking completed! View results in History.');
            }
          } else if (event.type === 'error') {
            // Update job status
            if (event.job_id) {
              updateJobStatus(event.job_id, 'failed');
            }

            setError(event.message ?? 'Unknown error');
            setRunning(false);
          } else if (event.type === 'cancelled') {
            // Update job status
            if (event.job_id) {
              updateJobStatus(event.job_id, 'cancelled');
            }

            setRunning(false);
            setScreen('welcome');
          }
        });

        addLog('Backend initialized...');
        setBridgeReady(true);

        // Only start new docking job if not resuming
        if (!isResumingJob) {
          addLog('Starting new docking job...');
          const { job_id } = await bridge.startDocking(config);
          addLog(`Received job_id: ${job_id}`);

          // Add job to activeJobs Map with snapshot config
          if (pendingJobConfig) {
            addLog(`Adding job to activeJobs with config: ${JSON.stringify(pendingJobConfig)}`);
            addJob({
              id: job_id,
              config: pendingJobConfig,
              status: 'running',
              startTime: new Date().toISOString(),
            });
            addLog(`Job added to activeJobs, total jobs: ${activeJobs.size + 1}`);

            // Clear pending config
            setPendingJobConfig(null);
          } else {
            addLog(`WARNING: pendingJobConfig is null, job NOT added to activeJobs!`);
          }

          // Save job_id for tracking
          setCurrentJob(job_id);
          addLog(`Job started with ID: ${job_id}`);
        } else {
          addLog(`Resuming job ${currentJobId}, not starting new docking job`);
          setRunning(true);
        }
      } catch (err) {
        const error = err as Error;
        addLog(`Error: ${error.message}`);
        setError(`Failed to start docking: ${error.message}`);
        setRunning(false);
        setScreen('welcome');
      }
    };

    startOrResumeJob();

    // No cleanup on unmount - allow background job to continue
    // Backend will be cleaned up when app exits
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Progress events are now handled in the first useEffect (before startDocking)

  useInput((input, key) => {
    if (isKey(input, 'c') && bridgeReady && currentJobId) {
      getBridge().cancelJob(currentJobId).catch(() => {});
    } else if (isKey(input, 'h') || key.escape) {
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
        <Text color="cyan" dimColor>h/Esc</Text>
        <Text color="gray" dimColor> run in background</Text>
      </Box>

      {/* Batch progress (if batch mode) */}
      {isBatch && (
        <Box flexDirection="column" marginTop={1}>
          <Text bold color="white">Batch Progress</Text>
          <Box marginTop={1}>
            <ProgressBar percent={batchProgress} width={UI_DIMENSIONS.PROGRESS_BAR_WIDTH} showPercent />
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
                <Text color="white">{String(pose.rank).padEnd(TABLE_COLUMN_WIDTHS.RANK)}</Text>
                <Text color="magenta">{(pose.error_estimate ?? 0).toFixed(3).padEnd(TABLE_COLUMN_WIDTHS.ERROR_ESTIMATE)}</Text>
                <Text color={pose.pb_count === 4 ? 'green' : 'yellow'}>{pose.pb_count ?? 0}/4</Text>
              </Box>
            ))}
          </Box>
        </Box>
      )}

      {/* Ligand status list (if batch mode, scrollable) */}
      {isBatch && ligandStatuses.length > 0 && (
        <Box flexDirection="column" marginTop={1}>
          <Text bold color="white">Ligand Status</Text>
          <Box flexDirection="column" marginTop={1} height={Math.min(UI_DIMENSIONS.MAX_VISIBLE_LIGANDS, ligandStatuses.length)}>
            {ligandStatuses.slice(0, UI_DIMENSIONS.MAX_VISIBLE_LIGANDS).map((lig, idx) => (
              <LigandRow key={idx} ligand={lig} isCurrent={idx === currentLigandIndex} />
            ))}
            {ligandStatuses.length > UI_DIMENSIONS.MAX_VISIBLE_LIGANDS && (
              <Text color="gray">... and {ligandStatuses.length - UI_DIMENSIONS.MAX_VISIBLE_LIGANDS} more</Text>
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
      <Text color={color} bold={status?.status === 'running'}>{stage.name.padEnd(TABLE_COLUMN_WIDTHS.STAGE_NAME)}</Text>
      {status?.status === 'running' && (
        <Box marginLeft={1} width={UI_DIMENSIONS.PROGRESS_BOX_WIDTH}>
          <ProgressBar percent={status.progress} width={UI_DIMENSIONS.PROGRESS_BAR_COMPACT_WIDTH} showPercent />
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
        {ligand.name.substring(0, TEXT_LIMITS.LIGAND_NAME).padEnd(TEXT_LIMITS.LIGAND_NAME)}
      </Text>
      {ligand.status === 'completed' && ligand.error_estimate !== undefined && (
        <>
          <Text color="gray"> err: </Text>
          <Text color="magenta">{ligand.error_estimate.toFixed(3)}</Text>
          <Text color="gray"> pb: </Text>
          <Text color={ligand.pb_count === 4 ? 'green' : 'yellow'}>{ligand.pb_count}/4</Text>
        </>
      )}
      {ligand.status === 'failed' && ligand.error_message && (
        <Text color="red"> {ligand.error_message.substring(0, TEXT_LIMITS.ERROR_MESSAGE_RUNNING)}</Text>
      )}
    </Box>
  );
}
