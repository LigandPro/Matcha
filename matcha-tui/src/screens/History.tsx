/**
 * History Screen - view previous docking runs.
 */

import React, { useEffect, useState } from 'react';
import { Box, Text, useInput } from 'ink';
import path from 'path';
import * as fs from 'fs';
import { useStore } from '../store/index.js';
import { getBridge, initBridge } from '../services/index.js';
import { icons } from '../utils/colors.js';
import { logger } from '../utils/logger.js';
import type { JobResults, PoseResult } from '../types/index.js';

interface RunInfo {
  name: string;
  path: string;
  date: string;
  status: string;
}

export function HistoryScreen(): React.ReactElement {
  const setScreen = useStore((s) => s.setScreen);
  const setError = useStore((s) => s.setError);
  const setResults = useStore((s) => s.setResults);
  const jobConfig = useStore((s) => s.jobConfig);
  const setModalOpen = useStore((s) => s.setModalOpen);
  const debugMode = useStore((s) => s.debugMode);

  const [runs, setRuns] = useState<RunInfo[]>([]);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [loading, setLoading] = useState(true);
  const [loadingResults, setLoadingResults] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [deleting, setDeleting] = useState(false);

  // Update modal state when confirmDelete changes
  useEffect(() => {
    setModalOpen(confirmDelete);
    return () => setModalOpen(false); // Clean up on unmount
  }, [confirmDelete, setModalOpen]);

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
      } catch (err: any) {
        setError(`Failed to load history: ${err.message}`);
      } finally {
        setLoading(false);
      }
    };

    loadHistory();
  }, [jobConfig.params?.outputDir, setError]);

  // Load and view results for selected run
  const viewResults = async () => {
    if (loadingResults || runs.length === 0) return;

    const selectedRun = runs[selectedIndex];
    if (!selectedRun) return;

    try {
      setLoadingResults(true);

      // Load poses and details from backend
      const [poses, details] = await Promise.all([
        getBridge().getPoses(selectedRun.path),
        getBridge().getRunDetails(selectedRun.path),
      ]);

      if (debugMode) {
        logger.debug('History', 'Loaded poses', { count: poses.length });
        logger.debug('History', 'Run details', details);
      }

      // Transform poses to match expected format with validation
      const transformedPoses: PoseResult[] = poses
        .filter((p: any) => p != null && typeof p.error_estimate === 'number')
        .map((p: any) => ({
          rank: p.rank ?? 0,
          error_estimate: p.error_estimate ?? 0,
          pb_count: p.pb_count ?? 0,
          checks: {
            not_too_far_away: p.not_too_far_away ?? false,
            no_internal_clash: p.no_internal_clash ?? false,
            no_clashes: p.no_clashes ?? false,
            no_volume_clash: p.no_volume_clash ?? false,
          },
          buried_fraction: p.buried_fraction ?? 0,
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
      if (details.files?.log && fs.existsSync(details.files.log)) {
        try {
          const logContent = fs.readFileSync(details.files.log, 'utf-8');
          const match = logContent.match(/Total runtime: ([\d.]+)/);
          if (match) {
            runtime = parseFloat(match[1]);
          }
        } catch (err) {
          // Log parsing failed, keep runtime as 0
        }
      }

      // Build results object
      const results: JobResults = {
        runName: selectedRun.name,
        runtime,
        totalPoses,
        physicalPoses,
        poses: transformedPoses,
        bestPosePath: details.files?.best_pose || selectedRun.path,
        allPosesPath: details.files?.all_poses || '',
        logPath: details.files?.log || '',
      };

      // Set results and navigate to results screen
      setResults(results);
      setScreen('results');
    } catch (err: any) {
      logger.error('History', 'Failed to load results', { error: err });
      setError(`Failed to load results: ${err.message}`);
    } finally {
      setLoadingResults(false);
    }
  };

  // Delete selected run
  const deleteRun = async () => {
    if (deleting || runs.length === 0) return;

    const selectedRun = runs[selectedIndex];
    if (!selectedRun) return;

    try {
      setDeleting(true);

      // Call backend to delete run
      const result = await getBridge().deleteRun(selectedRun.path);

      if (result.success) {
        // Remove from local list
        const newRuns = runs.filter((_, idx) => idx !== selectedIndex);
        setRuns(newRuns);

        // Adjust selected index if needed
        if (selectedIndex >= newRuns.length && newRuns.length > 0) {
          setSelectedIndex(newRuns.length - 1);
        }

        setConfirmDelete(false);
      } else {
        setError(result.error || 'Failed to delete run');
      }
    } catch (err: any) {
      setError(`Failed to delete run: ${err.message}`);
    } finally {
      setDeleting(false);
    }
  };

  useInput((input, key) => {
    if (loadingResults || deleting) return; // Disable navigation while loading

    // Confirmation mode
    if (confirmDelete) {
      if (input === 'y') {
        deleteRun();
      } else if (input === 'n' || key.escape) {
        setConfirmDelete(false);
      }
      return;
    }

    // Normal mode
    if (key.upArrow) setSelectedIndex((i) => Math.max(0, i - 1));
    else if (key.downArrow) setSelectedIndex((i) => Math.min(runs.length - 1, i + 1));
    else if (key.return) viewResults();
    else if (input === 'd' && runs.length > 0) setConfirmDelete(true);
    else if (key.escape) setScreen('welcome');
  });

  if (loading) {
    return <Box><Text color="yellow">Loading history...</Text></Box>;
  }

  if (loadingResults) {
    return (
      <Box flexDirection="column" gap={1}>
        <Text color="yellow">Loading results for {runs[selectedIndex]?.name}...</Text>
        <Text color="gray">Please wait...</Text>
      </Box>
    );
  }

  if (deleting) {
    return (
      <Box flexDirection="column" gap={1}>
        <Text color="yellow">Deleting {runs[selectedIndex]?.name}...</Text>
        <Text color="gray">Please wait...</Text>
      </Box>
    );
  }

  // Confirmation dialog
  if (confirmDelete && runs[selectedIndex]) {
    return (
      <Box flexDirection="column" gap={1}>
        <Box borderStyle="double" borderColor="red" paddingX={2} paddingY={1}>
          <Box flexDirection="column">
            <Text bold color="red">Delete Run?</Text>
            <Box marginTop={1}>
              <Text color="white">Are you sure you want to delete this run?</Text>
            </Box>
            <Box marginTop={1}>
              <Text color="yellow" bold>{runs[selectedIndex].name}</Text>
            </Box>
            <Box marginTop={1}>
              <Text color="gray">This action cannot be undone.</Text>
            </Box>
          </Box>
        </Box>

        <Box marginTop={2} gap={4}>
          <Text color="red" bold>[y] Yes, delete</Text>
          <Text color="cyan">[n] No, cancel</Text>
          <Text color="gray">[Esc] Cancel</Text>
        </Box>
      </Box>
    );
  }

  if (runs.length === 0) {
    return (
      <Box flexDirection="column" gap={1}>
        <Text color="gray">No previous runs found.</Text>
        <Box marginTop={2}><Text color="cyan">[Esc] Back to home</Text></Box>
      </Box>
    );
  }

  return (
    <Box flexDirection="column" gap={1}>
      <Text bold color="white">Previous Runs ({runs.length})</Text>

      <Box marginTop={1}>
        <Text color="gray" bold>{'  '}{'Name'.padEnd(25)}{'Date'.padEnd(20)}Status</Text>
      </Box>

      <Box flexDirection="column">
        {runs.map((run, index) => {
          const isSelected = index === selectedIndex;
          const statusIcon = run.status === 'completed' ? icons.check : icons.pending;
          const statusColor = run.status === 'completed' ? 'green' : 'yellow';

          return (
            <Box key={run.path}>
              <Text color={isSelected ? 'cyan' : 'white'}>{isSelected ? '▶ ' : '  '}</Text>
              <Text color={isSelected ? 'cyan' : 'white'} bold={isSelected}>{run.name.slice(0, 23).padEnd(25)}</Text>
              <Text color="gray">{formatDate(run.date).padEnd(20)}</Text>
              <Text color={statusColor}>{statusIcon} {run.status}</Text>
            </Box>
          );
        })}
      </Box>

      {runs[selectedIndex] && (
        <Box flexDirection="column" marginTop={2} borderStyle="single" borderColor="gray" paddingX={1}>
          <Text bold color="white">{runs[selectedIndex].name}</Text>
          <Box><Text color="gray">Path: </Text><Text color="blue">{runs[selectedIndex].path}</Text></Box>
        </Box>
      )}

      <Box marginTop={2} gap={2}>
        <Text color="cyan">[Enter] View results</Text>
        <Text color="red">[d] Delete</Text>
        <Text color="gray">[↑/↓] Navigate</Text>
        <Text color="gray">[Esc] Back</Text>
      </Box>
    </Box>
  );
}

function formatDate(isoDate: string): string {
  try {
    const date = new Date(isoDate);
    return date.toLocaleString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
  } catch {
    return isoDate.slice(0, 16);
  }
}
