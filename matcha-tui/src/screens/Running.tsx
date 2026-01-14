/**
 * Running Screen - shows docking progress with batch monitoring support.
 */

import React, { useEffect, useState, useRef } from 'react';
import { Box, Text, useInput } from 'ink';
import { useStore } from '../store/index.js';
import { getBridge, initBridge, closeBridge, type ProgressEvent } from '../services/index.js';
import { ProgressBar } from '../components/ProgressBar.js';
import { icons } from '../utils/colors.js';
import { PIPELINE_STAGES, type PipelineStage, type PoseResult } from '../types/index.js';
import { formatDuration } from '../utils/format.js';
import * as path from 'path';
import { fileURLToPath } from 'url';

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
  const setScreen = useStore((s) => s.setScreen);
  const setRunning = useStore((s) => s.setRunning);
  const setResults = useStore((s) => s.setResults);
  const setError = useStore((s) => s.setError);
  const addLog = useStore((s) => s.addLog);
  const jobConfig = useStore((s) => s.jobConfig);

  const [stages, setStages] = useState<Record<string, StageStatus>>({});
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

  useEffect(() => {
    const timer = setInterval(() => {
      setElapsedTime(Math.floor((Date.now() - startTime) / 1000));
    }, 1000);
    return () => clearInterval(timer);
  }, [startTime]);

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
      run_name: jobConfig.params?.runName || `matcha_${Date.now()}`,
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
        // Get project root from env (set in index.tsx) or calculate from current file
        const __filename = fileURLToPath(import.meta.url);
        const __dirname = path.dirname(__filename);
        const projectRoot = process.env.MATCHA_ROOT || path.resolve(__dirname, '..', '..', '..');
        addLog(`Project root: ${projectRoot}`);
        addLog(`MATCHA_ROOT env: ${process.env.MATCHA_ROOT || 'not set'}`);
        addLog(`__dirname: ${__dirname}`);

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

        bridge.on('error', (err: Error) => {
          addLog(`[bridge error] ${err.message}`);
        });

        bridge.on('exit', (code: number) => {
          addLog(`[backend exit] code ${code}`);
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

    // Cleanup on unmount
    return () => {
      closeBridge().catch(() => {});
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!bridgeReady) return;

    const bridge = getBridge();

    const handleProgress = (event: ProgressEvent) => {
      addLog(JSON.stringify(event));

      // Batch events
      if (event.type === 'batch_start') {
        setIsBatch(true);
        setTotalLigands(event.total_ligands ?? 1);
        if (event.ligand_statuses) {
          setLigandStatuses(event.ligand_statuses as LigandStatus[]);
        }
      } else if (event.type === 'batch_progress') {
        setBatchProgress(event.progress ?? 0);
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
        setResults({
          runName: 'completed',
          runtime: (Date.now() - startTime) / 1000,
          totalPoses: poses.length,
          physicalPoses: poses.filter((p) => p.pbCount === 4).length,
          poses,
          bestPosePath: event.output_path ?? '',
          allPosesPath: '',
          logPath: '',
          totalLigands: event.total_ligands,
          ligandStatuses: event.ligand_statuses as LigandStatus[] | undefined,
        });
        setScreen('results');
      } else if (event.type === 'error') {
        setError(event.message ?? 'Unknown error');
        setRunning(false);
      } else if (event.type === 'cancelled') {
        setRunning(false);
        setScreen('welcome');
      }
    };

    bridge.on('progress', handleProgress);
    return () => { bridge.off('progress', handleProgress); };
  }, [bridgeReady, addLog, setScreen, setRunning, setResults, setError, startTime, poses]);

  useInput((input) => {
    if (input === 'c' && bridgeReady) {
      getBridge().cancelJob().catch(() => {});
    }
  });

  return (
    <Box flexDirection="column" gap={1}>
      {/* Header with elapsed time */}
      <Box>
        <Text color="magenta">Elapsed: </Text>
        <Text color="white">{formatDuration(elapsedTime)}</Text>
        {isBatch && (
          <>
            <Text color="gray"> │ </Text>
            <Text color="magenta">Ligand: </Text>
            <Text color="white">{currentLigandIndex + 1}/{totalLigands}</Text>
          </>
        )}
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
                <Text color="magenta">{pose.errorEstimate.toFixed(3).padEnd(12)}</Text>
                <Text color={pose.pbCount === 4 ? 'green' : 'yellow'}>{pose.pbCount}/4</Text>
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

      <Box marginTop={2}>
        <Text color="gray">Press [c] to cancel</Text>
      </Box>
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
