/**
 * Running Screen - shows docking progress for the selected job.
 *
 * Progress events are handled globally in App.tsx and stored in Zustand.
 * This screen is a pure view over the store state, plus job start/cancel actions.
 */

import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Box, Text, useInput } from 'ink';
import path from 'path';
import { useStore } from '../store/index.js';
import { getBridge, initBridge } from '../services/index.js';
import { ProgressBar } from '../components/ProgressBar.js';
import { icons, getStageColor, getStatusColor } from '../utils/colors.js';
import { PIPELINE_STAGES, type PipelineStage, type PoseResult, type JobConfig } from '../types/index.js';
import { formatDuration, generateRunName } from '../utils/format.js';
import { isKey } from '../utils/keyboard.js';

type StageRuntimeStatus = {
  status: 'pending' | 'running' | 'done' | 'error';
  progress: number;
  elapsed?: number;
  message?: string;
};

const STAGE_ORDER = ['init', 'checkpoints', 'dataset', 'esm', 'stage1', 'stage2', 'stage3', 'scoring', 'posebusters'] as const;

function getInitialStages(): Record<string, StageRuntimeStatus> {
  const initial: Record<string, StageRuntimeStatus> = {};
  for (const id of STAGE_ORDER) {
    initial[id] = { status: 'pending', progress: 0 };
  }
  return initial;
}

export function RunningScreen(): React.ReactElement {
  const screen = useStore((s) => s.screen);
  const setScreen = useStore((s) => s.setScreen);
  const setResults = useStore((s) => s.setResults);
  const setError = useStore((s) => s.setError);
  const setNotification = useStore((s) => s.setNotification);

  const jobConfig = useStore((s) => s.jobConfig);

  // Multi-job support
  const currentJobId = useStore((s) => s.currentJobId);
  const setCurrentJob = useStore((s) => s.setCurrentJob);
  const addJob = useStore((s) => s.addJob);
  const syncJobsFromBackend = useStore((s) => s.syncJobsFromBackend);
  const activeJobs = useStore((s) => s.activeJobs);
  const jobRuntime = useStore((s) => (s.currentJobId ? s.jobRuntime.get(s.currentJobId) ?? null : null));

  const job = currentJobId ? activeJobs.get(currentJobId) ?? null : null;

  const jobStartedRef = useRef(false);
  const resultsShownForJobRef = useRef<string | null>(null);

  // Elapsed time (based on runtime timestamp, if present)
  const [elapsedTime, setElapsedTime] = useState(0);
  useEffect(() => {
    const timer = setInterval(() => {
      const startMs = jobRuntime?.startedAtMs ?? null;
      if (!startMs) return;
      setElapsedTime(Math.floor((Date.now() - startMs) / 1000));
    }, 1000);
    return () => clearInterval(timer);
  }, [jobRuntime?.startedAtMs]);

  const stages = jobRuntime?.stages ?? getInitialStages();
  const poses = jobRuntime?.poses ?? [];
  const batch = jobRuntime?.batch ?? {
    isBatch: false,
    totalLigands: 1,
    currentLigandIndex: 0,
    currentLigand: null,
    ligandStatuses: [],
  };

  const batchProgress = useMemo(() => {
    if (!batch.isBatch) return 0;
    let totalWeight = 0;
    let weighted = 0;
    for (const stageId of STAGE_ORDER) {
      totalWeight += 1;
      const st = stages[stageId];
      if (!st) continue;
      const percent = st.status === 'done' ? 100 : st.status === 'running' ? st.progress : 0;
      weighted += percent;
    }
    return totalWeight > 0 ? Math.round(weighted / totalWeight) : 0;
  }, [batch.isBatch, stages]);

  // Start docking job on mount (if no current job is set).
  useEffect(() => {
    if (jobStartedRef.current) return;
    jobStartedRef.current = true;

    const startJob = async () => {
      try {
        // If we're resuming an existing job, do nothing.
        if (currentJobId && job && (job.status === 'running' || job.status === 'queued')) {
          return;
        }

        // Build config from jobConfig (backend expects snake_case keys).
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

        const projectRoot =
          process.env.MATCHA_ROOT || path.join(process.cwd(), '..');

        const bridge = getBridge();
        if (!bridge.isReady()) {
          await initBridge({ projectRoot, useUv: true });
        }

        const { job_id, status } = await getBridge().startDocking(config);

        setCurrentJob(job_id);
        addJob({
          id: job_id,
          config: jobConfig as JobConfig,
          status: status === 'queued' ? 'queued' : 'running',
          startTime: new Date().toISOString(),
          requestedGpu: jobConfig.params?.gpu,
        });

        // Sync jobs list (get assigned GPU, etc.)
        try {
          const { jobs } = await getBridge().listJobs();
          syncJobsFromBackend(jobs as any);
        } catch {
          // Ignore sync errors
        }
      } catch (err) {
        const error = err as Error;
        setError(`Failed to start docking: ${error.message}`);
        setScreen('welcome');
      }
    };

    startJob();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Auto-show results when the selected job completes and we have completion payload.
  useEffect(() => {
    if (!currentJobId || !jobRuntime?.completion) return;
    if (job?.status !== 'completed') return;
    if (resultsShownForJobRef.current === currentJobId) return;
    if (!jobRuntime.completion.outputPath) return;

    resultsShownForJobRef.current = currentJobId;

    const runDir = jobRuntime.completion.outputPath;
    const runName = jobConfig.params?.runName ?? 'matcha_tui_run';
    const isBatchRun = batch.isBatch || jobConfig.mode === 'batch' || (jobRuntime.completion.totalLigands ?? 0) > 1;

    const bestPosePath = isBatchRun
      ? path.join(runDir, 'best_poses')
      : path.join(runDir, `${runName}_best.sdf`);
    const allPosesPath = isBatchRun
      ? path.join(runDir, 'all_poses')
      : path.join(runDir, `${runName}_poses.sdf`);
    const logPath = path.join(runDir, `${runName}.log`);

    const finalPoses = jobRuntime.completion.poses ?? [];

    const results = {
      runName,
      runtime: (jobRuntime.startedAtMs ? (Date.now() - jobRuntime.startedAtMs) / 1000 : 0),
      totalPoses: finalPoses.length,
      physicalPoses: finalPoses.filter((p) => p.pb_count === 4).length,
      poses: finalPoses,
      bestPosePath,
      allPosesPath,
      logPath,
      receptor: jobConfig.receptor,
      ligand: jobConfig.mode === 'batch' ? jobConfig.ligandDir : jobConfig.ligand,
      totalLigands: jobRuntime.completion.totalLigands,
      ligandStatuses: jobRuntime.completion.ligandStatuses,
    };

    setResults(results);

    const currentScreen = useStore.getState().screen;
    if (currentScreen === 'running') {
      setScreen('results');
    } else {
      setNotification('Docking completed! View results in History.');
    }
  }, [batch.isBatch, currentJobId, job?.status, jobConfig, jobRuntime?.completion, jobRuntime?.startedAtMs, setNotification, setResults, setScreen]);

  useInput((input, key) => {
    if (!currentJobId) return;

    if (isKey(input, 'c')) {
      getBridge().cancelJob(currentJobId).catch(() => {});
      return;
    }

    if (isKey(input, 'h') || key.escape || key.leftArrow) {
      // Allow leaving the running screen. Job continues in background.
      setScreen('welcome');
      return;
    }
  });

  const isQueued = job?.status === 'queued';

  return (
    <Box flexDirection="column" gap={1}>
      {/* Compact status line */}
      <Box>
        <Text color={isQueued ? 'gray' : 'yellow'}>{isQueued ? icons.pending : icons.running} </Text>
        <Text color="white" bold>{isQueued ? 'Queued' : 'Docking'}</Text>
        <Text color="gray" dimColor> • </Text>
        <Text color="gray" dimColor>{formatDuration(elapsedTime)}</Text>
        {batch.isBatch && (
          <>
            <Text color="gray" dimColor> • </Text>
            <Text color="magenta">Ligand </Text>
            <Text color="white">{batch.currentLigandIndex + 1}/{batch.totalLigands}</Text>
          </>
        )}
        <Text color="gray" dimColor> • </Text>
        <Text color="#D0D1FA" dimColor>h/Esc</Text>
        <Text color="gray" dimColor> run in background</Text>
      </Box>

      {isQueued && (
        <Box>
          <Text color="gray" dimColor>Waiting for an available GPU...</Text>
        </Box>
      )}

      {/* Batch progress (if batch mode) */}
      {batch.isBatch && (
        <Box flexDirection="column" marginTop={1}>
          <Text bold color="white">Batch Progress</Text>
          <Box marginTop={1}>
            <ProgressBar percent={batchProgress} width={40} showPercent />
          </Box>
        </Box>
      )}

      {/* Current ligand info (if batch mode) */}
      {batch.isBatch && batch.currentLigand && (
        <Box marginTop={1}>
          <Text color="magenta">Current: </Text>
          <Text color="yellow">{batch.currentLigand}</Text>
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

      {/* Current poses */}
      {poses.length > 0 && (
        <Box flexDirection="column" marginTop={1}>
          <Text bold color="white">Top Poses {batch.currentLigand ? `(${batch.currentLigand})` : ''}</Text>
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

      {/* Ligand status list (if batch mode) */}
      {batch.isBatch && batch.ligandStatuses.length > 0 && (
        <Box flexDirection="column" marginTop={1}>
          <Text bold color="white">Ligand Status</Text>
          <Box flexDirection="column" marginTop={1} height={Math.min(8, batch.ligandStatuses.length)}>
            {batch.ligandStatuses.slice(0, 8).map((lig, idx) => (
              <LigandRow key={idx} ligand={lig} isCurrent={idx === batch.currentLigandIndex} />
            ))}
            {batch.ligandStatuses.length > 8 && (
              <Text color="gray">... and {batch.ligandStatuses.length - 8} more</Text>
            )}
          </Box>
        </Box>
      )}
    </Box>
  );
}

function StageRow({ stage, status }: { stage: { id: PipelineStage; name: string }; status?: StageRuntimeStatus }): React.ReactElement {
  const icon =
    status?.status === 'done'
      ? icons.check
      : status?.status === 'running'
      ? icons.running
      : status?.status === 'error'
      ? icons.error
      : icons.pending;

  const color =
    status?.status === 'error'
      ? 'red'
      : status?.status
      ? getStageColor(status.status as 'pending' | 'running' | 'done')
      : 'gray';

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

function LigandRow({ ligand, isCurrent }: { ligand: any; isCurrent: boolean }): React.ReactElement {
  const status = ligand.status as 'pending' | 'running' | 'completed' | 'failed';
  const statusIcon = {
    pending: icons.pending,
    running: icons.running,
    completed: icons.check,
    failed: icons.error,
  }[status] ?? icons.pending;

  const statusColor = getStatusColor(status);

  return (
    <Box>
      <Text color={statusColor}>{statusIcon} </Text>
      <Text color={isCurrent ? 'magenta' : 'white'} bold={isCurrent}>
        {String(ligand.name ?? '').substring(0, 25).padEnd(25)}
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
        <Text color="red"> {String(ligand.error_message).substring(0, 30)}</Text>
      )}
    </Box>
  );
}
