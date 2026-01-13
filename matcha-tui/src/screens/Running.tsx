/**
 * Running Screen - shows docking progress.
 */

import React, { useEffect, useState } from 'react';
import { Box, Text, useInput } from 'ink';
import { useStore } from '../store/index.js';
import { getBridge, type ProgressEvent } from '../services/index.js';
import { ProgressBar } from '../components/ProgressBar.js';
import { icons } from '../utils/colors.js';
import { PIPELINE_STAGES, type PipelineStage, type PoseResult } from '../types/index.js';
import { formatDuration } from '../utils/format.js';

interface StageStatus {
  status: 'pending' | 'running' | 'done' | 'error';
  progress: number;
  elapsed?: number;
  message?: string;
}

export function RunningScreen(): React.ReactElement {
  const setScreen = useStore((s) => s.setScreen);
  const setRunning = useStore((s) => s.setRunning);
  const setResults = useStore((s) => s.setResults);
  const setError = useStore((s) => s.setError);
  const addLog = useStore((s) => s.addLog);

  const [stages, setStages] = useState<Record<string, StageStatus>>({});
  const [poses, setPoses] = useState<PoseResult[]>([]);
  const [startTime] = useState(Date.now());
  const [elapsedTime, setElapsedTime] = useState(0);

  useEffect(() => {
    const timer = setInterval(() => {
      setElapsedTime(Math.floor((Date.now() - startTime) / 1000));
    }, 1000);
    return () => clearInterval(timer);
  }, [startTime]);

  useEffect(() => {
    const bridge = getBridge();

    const handleProgress = (event: ProgressEvent) => {
      addLog(JSON.stringify(event));

      if (event.type === 'stage_start' && event.stage) {
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
  }, [addLog, setScreen, setRunning, setResults, setError, startTime, poses]);

  useInput((input) => {
    if (input === 'c') {
      getBridge().cancelJob().catch(() => {});
    }
  });

  return (
    <Box flexDirection="column" gap={1}>
      <Box>
        <Text color="cyan">Elapsed: </Text>
        <Text color="white">{formatDuration(elapsedTime)}</Text>
      </Box>

      <Box flexDirection="column" marginTop={1}>
        <Text bold color="white">Pipeline Progress</Text>
        <Box flexDirection="column" marginTop={1}>
          {PIPELINE_STAGES.map((stageInfo) => (
            <StageRow key={stageInfo.id} stage={stageInfo} status={stages[stageInfo.id]} />
          ))}
        </Box>
      </Box>

      {poses.length > 0 && (
        <Box flexDirection="column" marginTop={1}>
          <Text bold color="white">Top Poses</Text>
          <Box flexDirection="column" marginTop={1}>
            {poses.slice(0, 5).map((pose) => (
              <Box key={pose.rank}>
                <Text color="white">{String(pose.rank).padEnd(6)}</Text>
                <Text color="cyan">{pose.errorEstimate.toFixed(3).padEnd(12)}</Text>
                <Text color={pose.pbCount === 4 ? 'green' : 'yellow'}>{pose.pbCount}/4</Text>
              </Box>
            ))}
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
