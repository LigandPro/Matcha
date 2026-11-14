/**
 * History Screen - view previous docking runs.
 */

import React, { useEffect, useState } from 'react';
import { Box, Text, useInput } from 'ink';
import path from 'path';
import * as fs from 'fs';
import { useStore } from '../store/index.js';
import { getBridge, initBridge } from '../services/index.js';
import { icons, getStatusColor } from '../utils/colors.js';
import { logger } from '../utils/logger.js';
import { isKey } from '../utils/keyboard.js';
import { parseRuntimeFromLog } from '../utils/runtime.js';
import type { BatchLigandStatus, JobResults, PoseResult } from '../types/index.js';
import { LoadingState, EmptyState } from '../components/index.js';
import { TABLE_COLUMN_WIDTHS, TEXT_LIMITS } from '../constants/ui.js';

interface ApiError {
  message: string;
  code?: string;
}

type DeleteState =
  | { status: 'idle' }
  | { status: 'confirming' }
  | { status: 'deleting' }
  | { status: 'error'; message: string };

interface RunInfo {
  name: string;
  path: string;
  date: string;
  status: string;
  receptor?: string;
  ligand?: string;
}

interface RunDetails {
  name: string;
  path: string;
  files: Record<string, string>;
  receptor?: string;
  ligand?: string;
  is_batch?: boolean;
}

export function HistoryScreen(): React.ReactElement {
  const setScreen = useStore((s) => s.setScreen);
  const setPreviousScreen = useStore((s) => s.setPreviousScreen);
  const setError = useStore((s) => s.setError);
  const setResults = useStore((s) => s.setResults);
  const jobConfig = useStore((s) => s.jobConfig);
  const setModalOpen = useStore((s) => s.setModalOpen);
  const debugMode = useStore((s) => s.debugMode);

  const [runs, setRuns] = useState<RunInfo[]>([]);
  const selectedIndex = useStore((s) => s.historySelectedIndex);
  const setHistorySelectedIndex = useStore((s) => s.setHistorySelectedIndex);
  const [selectedDetails, setSelectedDetails] = useState<RunDetails | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadingResults, setLoadingResults] = useState(false);
  const [deleteState, setDeleteState] = useState<DeleteState>({ status: 'idle' });

  // Update modal state when deleteState changes
  useEffect(() => {
    const isConfirming = deleteState.status === 'confirming' || deleteState.status === 'deleting' || deleteState.status === 'error';
    setModalOpen(isConfirming);
    return () => setModalOpen(false); // Clean up on unmount
  }, [deleteState, setModalOpen]);

  useEffect(() => {
    const loadHistory = async () => {
      try {
        setLoading(true);

        // Initialize backend if not ready
        const bridge = getBridge();
        if (!bridge.isReady()) {
          const projectRoot = process.env.MATCHA_ROOT || path.join(process.cwd(), '..');
          await initBridge({
            projectRoot,
            useUv: true,
          });
        }

        const outputDir = jobConfig.params?.outputDir ?? './results';
        const result = await getBridge().listRuns(outputDir);

        // Backend already sorts by date (newest first), no need to sort again
        setRuns(result);
      } catch (err: unknown) {
        const error = err as ApiError;
        setError(`Failed to load history: ${error.message}`);
      } finally {
        setLoading(false);
      }
    };

    loadHistory();
  }, [jobConfig.params?.outputDir, setError]);

  useEffect(() => {
    if (runs.length === 0) {
      if (selectedIndex !== 0) {
        setHistorySelectedIndex(0);
      }
      return;
    }
    if (selectedIndex > runs.length - 1) {
      setHistorySelectedIndex(runs.length - 1);
    }
  }, [runs.length, selectedIndex, setHistorySelectedIndex]);

  // Load details for selected run
  useEffect(() => {
    const loadDetails = async () => {
      if (runs.length === 0 || !runs[selectedIndex]) {
        setSelectedDetails(null);
        return;
      }

      try {
        const details = await getBridge().getRunDetails(runs[selectedIndex].path);
        setSelectedDetails(details as RunDetails);
      } catch (err) {
        setSelectedDetails(null);
      }
    };

    loadDetails();
  }, [runs, selectedIndex]);

  // Load and view results for selected run
  const viewResults = async () => {
    if (loadingResults || runs.length === 0) return;

    const selectedRun = runs[selectedIndex];
    if (!selectedRun) return;

    try {
      setLoadingResults(true);

      const details = (await getBridge().getRunDetails(selectedRun.path)) as RunDetails;
      const isBatchRun =
        Boolean(details.is_batch) ||
        Boolean(details.files?.best_poses) ||
        Boolean(details.files?.all_poses_dir);

      let poses: unknown[] = [];
      if (!isBatchRun) {
        poses = await getBridge().getPoses(selectedRun.path);
      }

      if (debugMode) {
        logger.debug('History', 'Loaded poses', { count: poses.length });
        logger.debug('History', 'Run details', { details });
      }

      // Transform poses to match expected format with validation
      const transformedPoses: PoseResult[] = poses
        .filter((p: unknown): p is Record<string, unknown> => p != null && typeof (p as Record<string, unknown>).error_estimate === 'number')
        .map((p: Record<string, unknown>) => ({
          rank: (p.rank as number) ?? 0,
          error_estimate: (p.error_estimate as number) ?? 0,
          pb_count: (p.pb_count as number) ?? 0,
          checks: {
            not_too_far_away: (p.not_too_far_away as boolean) ?? false,
            no_internal_clash: (p.no_internal_clash as boolean) ?? false,
            no_clashes: (p.no_clashes as boolean) ?? false,
            no_volume_clash: (p.no_volume_clash as boolean) ?? false,
          },
          buried_fraction: (p.buried_fraction as number) ?? 0,
        }));

      // Calculate statistics
      const totalPoses = transformedPoses.length;
      const physicalPoses = transformedPoses.filter((p) => p.pb_count === 4).length;

      // Check for empty data
      if (poses.length === 0) {
        logger.warn('History', 'No poses found for run', { path: selectedRun.path });
      }

      // Try to extract runtime from log file
      let runtime = 0;
      let logContent: string | null = null;
      if (details.files?.log && fs.existsSync(details.files.log)) {
        try {
          logContent = fs.readFileSync(details.files.log, 'utf-8');
          runtime = parseRuntimeFromLog(logContent);
        } catch (err) {
          // Log parsing failed, keep runtime as 0
        }
      }

      let ligandStatuses: BatchLigandStatus[] | undefined;
      let totalLigandsFromLog: number | undefined;
      if (isBatchRun) {
        const summaryLog = details.files?.log;
        const bestDir = details.files?.best_poses;
        const parsed: BatchLigandStatus[] = [];

        if (summaryLog && fs.existsSync(summaryLog)) {
          if (!logContent) {
            logContent = fs.readFileSync(summaryLog, 'utf-8');
          }
          const lines = logContent.split('\n');
          const bestRegex = /^\s*(\S+):\s*Best err=([-\d.]+),\s*pb=(\d)\/4/;
          const noResRegex = /^\s*(\S+):\s*No results/;
          const totalRegex = /Total molecules\s*:\s*(\d+)/i;
          const totalMatch = logContent.match(totalRegex);
          if (totalMatch) {
            totalLigandsFromLog = parseInt(totalMatch[1], 10);
          }

          for (const line of lines) {
            const bestMatch = line.match(bestRegex);
            if (bestMatch) {
              const name = bestMatch[1];
              const ligandPath = bestDir ? path.join(bestDir, `${name}.sdf`) : '';
              parsed.push({
                name,
                path: ligandPath,
                status: 'completed',
                error_estimate: parseFloat(bestMatch[2]),
                pb_count: parseInt(bestMatch[3], 10),
              });
              continue;
            }

            const noResMatch = line.match(noResRegex);
            if (noResMatch) {
              const name = noResMatch[1];
              parsed.push({
                name,
                path: bestDir ? path.join(bestDir, `${name}.sdf`) : '',
                status: 'failed',
                error_message: 'No results',
              });
            }
          }
        }

        if (parsed.length > 0) {
          ligandStatuses = parsed;
        } else if (bestDir && fs.existsSync(bestDir)) {
          const files = fs.readdirSync(bestDir).filter((f) => f.endsWith('.sdf'));
          ligandStatuses = files.map((file) => ({
            name: path.basename(file, '.sdf'),
            path: path.join(bestDir, file),
            status: 'completed',
          }));
          if (!totalLigandsFromLog) {
            totalLigandsFromLog = files.length;
          }
        }
      }

      // Build results object
      const results: JobResults = {
        runName: selectedRun.name,
        runtime,
        totalPoses,
        physicalPoses,
        poses: transformedPoses,
        bestPosePath: details.files?.best_pose || details.files?.best_poses || selectedRun.path,
        allPosesPath: details.files?.all_poses || details.files?.all_poses_dir || '',
        logPath: details.files?.log || '',
        receptor: selectedRun.receptor,
        ligand: selectedRun.ligand,
        totalLigands: ligandStatuses?.length ?? totalLigandsFromLog,
        ligandStatuses,
      };

      // Set results and navigate to results screen
      setResults(results);
      setPreviousScreen('history');
      setScreen('results');
    } catch (err: unknown) {
      const error = err as ApiError;
      logger.error('History', 'Failed to load results', { error: error.message });
      setError(`Failed to load results: ${error.message}`);
    } finally {
      setLoadingResults(false);
    }
  };

  // Delete selected run
  const deleteRun = async () => {
    if (deleteState.status === 'deleting' || runs.length === 0) return;

    const selectedRun = runs[selectedIndex];
    if (!selectedRun) return;

    try {
      setDeleteState({ status: 'deleting' });

      // Call backend to delete run
      const result = await getBridge().deleteRun(selectedRun.path);

      if (result.success) {
        // Remove from local list
        const newRuns = runs.filter((_, idx) => idx !== selectedIndex);
        setRuns(newRuns);

        // Adjust selected index if needed
        if (selectedIndex >= newRuns.length && newRuns.length > 0) {
          setHistorySelectedIndex(newRuns.length - 1);
        }

        setDeleteState({ status: 'idle' });
      } else {
        setDeleteState({ status: 'error', message: result.error || 'Failed to delete run' });
      }
    } catch (err: unknown) {
      const error = err as ApiError;
      setDeleteState({ status: 'error', message: `Failed to delete run: ${error.message}` });
    }
  };

  useInput((input, key) => {
    if (loadingResults || deleteState.status === 'deleting') return; // Disable navigation while loading

    // Confirmation mode
    if (deleteState.status === 'confirming' || deleteState.status === 'error') {
      if (isKey(input, 'y')) {
        deleteRun();
      } else if (isKey(input, 'n') || key.escape) {
        setDeleteState({ status: 'idle' });
      }
      return;
    }

    // Normal mode
    if (key.upArrow) setHistorySelectedIndex(Math.max(0, selectedIndex - 1));
    else if (key.downArrow) setHistorySelectedIndex(Math.min(runs.length - 1, selectedIndex + 1));
    else if (key.return) viewResults();
    else if (isKey(input, 'd') && runs.length > 0) setDeleteState({ status: 'confirming' });
    else if (key.escape || key.leftArrow) setScreen('welcome');
  });

  if (loading) {
    return <LoadingState message="Loading history..." />;
  }

  if (loadingResults) {
    return (
      <Box flexDirection="column" gap={1}>
        <Text color="yellow">Loading results for {runs[selectedIndex]?.name}...</Text>
        <Text color="gray">Please wait...</Text>
      </Box>
    );
  }

  if (deleteState.status === 'deleting') {
    return (
      <Box flexDirection="column" gap={1}>
        <Text color="yellow">Deleting {runs[selectedIndex]?.name}...</Text>
        <Text color="gray">Please wait...</Text>
      </Box>
    );
  }

  // Confirmation dialog
  if ((deleteState.status === 'confirming' || deleteState.status === 'error') && runs[selectedIndex]) {
    return (
      <Box flexDirection="column" gap={1}>
        <Box borderStyle="round" borderColor="gray" paddingX={1}>
          <Box flexDirection="column">
            <Text bold color="white">Delete run?</Text>
            <Text color="gray" dimColor>This action cannot be undone.</Text>
            <Text color="yellow">{runs[selectedIndex].name}</Text>
            {deleteState.status === 'error' && (
              <Text color="red" dimColor>Error: {deleteState.message}</Text>
            )}
          </Box>
        </Box>

        <Box paddingX={1}>
          <Text color="cyan" dimColor>[y]</Text>
          <Text color="gray" dimColor> delete </Text>
          <Text color="gray" dimColor>· </Text>
          <Text color="cyan" dimColor>[n]</Text>
          <Text color="gray" dimColor> cancel </Text>
          <Text color="gray" dimColor>· </Text>
          <Text color="cyan" dimColor>[esc]</Text>
          <Text color="gray" dimColor> close</Text>
        </Box>
      </Box>
    );
  }

  if (runs.length === 0) {
    return (
      <Box flexDirection="column" gap={1}>
        <EmptyState message="No previous runs found" hint="Run a docking job to see it here" />
        <Box marginTop={2}><Text color="cyan">[Esc] Back to home</Text></Box>
      </Box>
    );
  }

  return (
    <Box flexDirection="column" gap={1}>
      <Text bold color="white">Previous Runs ({runs.length})</Text>

      <Box marginTop={1}>
        <Text color="gray" bold>{'  '}{'Name'.padEnd(TABLE_COLUMN_WIDTHS.HISTORY_NAME)}{'Date'.padEnd(TABLE_COLUMN_WIDTHS.DATE)}Status</Text>
      </Box>

      <Box flexDirection="column">
        {runs.map((run, index) => {
          const isSelected = index === selectedIndex;
          const statusIcon = run.status === 'completed' ? icons.check : icons.pending;
          const statusColor = run.status === 'completed' ? getStatusColor('completed') : getStatusColor('pending');

          return (
            <Box key={run.path}>
              <Text color={isSelected ? 'cyan' : 'white'}>{isSelected ? '▶ ' : '  '}</Text>
              <Text color={isSelected ? 'cyan' : 'white'} bold={isSelected}>{run.name.slice(0, TEXT_LIMITS.RUN_NAME).padEnd(TABLE_COLUMN_WIDTHS.HISTORY_NAME)}</Text>
              <Text color="gray">{formatDate(run.date).padEnd(TABLE_COLUMN_WIDTHS.DATE)}</Text>
              <Text color={statusColor}>{statusIcon} {run.status}</Text>
            </Box>
          );
        })}
      </Box>

      {runs[selectedIndex] && selectedDetails && (
        <Box flexDirection="column" marginTop={2} borderStyle="single" borderColor="gray" paddingX={1}>
          <Text bold color="white">{runs[selectedIndex].name}</Text>
          <Box><Text color="gray">Date: </Text><Text color="white">{formatDate(runs[selectedIndex].date)}</Text></Box>
          {selectedDetails.receptor && (
            <Box>
              <Text color="gray">Receptor: </Text>
              <Text color="green">{path.basename(selectedDetails.receptor)}</Text>
            </Box>
          )}
          {selectedDetails.ligand && (
            <Box>
              <Text color="gray">Ligand: </Text>
              <Text color="green">{path.basename(selectedDetails.ligand)}</Text>
            </Box>
          )}
          <Box><Text color="gray">Path: </Text><Text color="blue">{runs[selectedIndex].path}</Text></Box>
        </Box>
      )}

      <Box marginTop={2} gap={2}>
        <Text color="cyan">[Enter] View results</Text>
        <Text color="red">[d] Delete</Text>
        <Text color="gray">[↑/↓] Navigate</Text>
        <Text color="gray">[Esc] Back</Text>
        <Text color="gray">[←] Back</Text>
      </Box>
    </Box>
  );
}

function formatDate(isoDate: string): string {
  try {
    const date = new Date(isoDate);
    return date.toLocaleString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
  } catch {
    return isoDate.slice(0, TABLE_COLUMN_WIDTHS.DATE_TRUNCATE);
  }
}
