/**
 * Running Screen - shows docking progress with batch monitoring support.
 */

import React, { useEffect, useState, useRef } from 'react';
import { Box, Text, useInput } from 'ink';
import path from 'path';
import { useStore } from '../store/index.js';
import { getBridge, initBridge, closeBridge, type ProgressEvent } from '../services/index.js';
import { ProgressBar } from '../components/ProgressBar.js';
import { icons } from '../utils/colors.js';
import { PIPELINE_STAGES, type PipelineStage, type PoseResult } from '../types/index.js';
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
  error_estimate?: number;
  pb_count?: number;
  error_message?: string;
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
  const [isBatch, setIsBatch] = useState(false);
  const [totalLigands, setTotalLigands] = useState(1);
  const [currentLigandIndex, setCurrentLigandIndex] = useState(0);
  const [currentLigand, setCurrentLigand] = useState<string | null>(null);
  const [ligandStatuses, setLigandStatuses] = useState<LigandStatus[]>([]);
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

  // Start docking job on mount
  useEffect(() => {
    if (jobStartedRef.current) return;
    jobStartedRef.current = true;
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

    // Initialize bridge first, then start docking
    const startJob = async () => {
      try {
        // Initialize Python backend
        addLog('Initializing Python backend...');
        // TUI is in matcha-tui/ subdirectory, but Python package is in parent directory
        // Use parent directory as project root so 'uv run python -m matcha.tui.backend' can find the package
        const projectRoot = process.env.MATCHA_ROOT || path.join(process.cwd(), '..');
        addLog(`Project root: ${projectRoot}`);

        // Capture all stderr before bridge starts
        let stderrBuffer = '';

        const bridge = await initBridge({
          projectRoot,
          useUv: true,
        });

        // Listen for stderr to help with debugging
        bridge.on('stderr', (data: string) => {
          stderrBuffer += data;
          addLog(`[stderr] ${data}`);
        });

        // Listen for debug events from backend
        bridge.on('debug', (event: any) => {
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
          addLog(JSON.stringify(event));

          // Batch events
          if (event.type === 'batch_start') {
            setIsBatch(true);
            setTotalLigands(event.total_ligands ?? 1);
            if (event.ligand_statuses) {
              setLigandStatuses(event.ligand_statuses as LigandStatus[]);
            }
          } else if (event.type === 'batch_progress') {
            setCurrentLigandIndex(event.ligand_index ?? 0);
            if (event.ligand_statuses) {
              setLigandStatuses(event.ligand_statuses as LigandStatus[]);
            }
          } else if (event.type === 'ligand_start') {
            setCurrentLigand(event.current_ligand ?? null);
            setCurrentLigandIndex(event.ligand_index ?? 0);
            setTotalLigands(event.total_ligands ?? 1);
            // Reset stages for new ligand
            setStages({});
            if (event.ligand_statuses) {
              setLigandStatuses(event.ligand_statuses as LigandStatus[]);
            }
          } else if (event.type === 'ligand_done') {
            if (event.ligand_statuses) {
              setLigandStatuses(event.ligand_statuses as LigandStatus[]);
            }
          } else if (event.type === 'stage_start' && event.stage) {
            // Update current ligand if provided
            if (event.current_ligand) {
              setCurrentLigand(event.current_ligand);
            }
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
            if (isBatch) {
              setBatchProgress(100);
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
            setError(event.message ?? 'Unknown error');
            setRunning(false);
          } else if (event.type === 'cancelled') {
            setRunning(false);
            setScreen('welcome');
          }
        });

        addLog('Backend initialized, starting docking...');
        setBridgeReady(true);

        // Start docking
        await bridge.startDocking(config);
      } catch (err) {
        const error = err as Error;
        addLog(`Error: ${error.message}`);
        setError(`Failed to start docking: ${error.message}`);
        setRunning(false);
        setScreen('welcome');
      }
    };

    startJob();

    // No cleanup on unmount - allow background job to continue
    // Backend will be cleaned up when app exits
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Progress events are now handled in the first useEffect (before startDocking)

  useInput((input, key) => {
    if (isKey(input, 'c') && bridgeReady) {
      getBridge().cancelJob().catch(() => {});
    } else if (isKey(input, 'h') || key.escape) {
      // Allow user to leave running screen and go back to home
      // Job continues in background
      setScreen('welcome');
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
        <Text color="cyan" dimColor>Esc</Text>
        <Text color="gray" dimColor> to home</Text>
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
                <Text color="magenta">{(pose.error_estimate ?? 0).toFixed(3).padEnd(12)}</Text>
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
  const color = status?.status === 'done' ? 'green' : status?.status === 'running' ? 'yellow' : 'gray';

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

  const statusColor = {
    pending: 'gray',
    running: 'yellow',
    completed: 'green',
    failed: 'red',
  }[ligand.status] as 'gray' | 'yellow' | 'green' | 'red';

  return (
    <Box>
      <Text color={statusColor}>{statusIcon} </Text>
      <Text color={isCurrent ? 'magenta' : 'white'} bold={isCurrent}>
        {ligand.name.substring(0, 25).padEnd(25)}
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
        <Text color="red"> {ligand.error_message.substring(0, 30)}</Text>
      )}
    </Box>
  );
}
