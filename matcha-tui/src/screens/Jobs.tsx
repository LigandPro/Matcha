/**
 * Jobs Screen - displays all active jobs (running and queued).
 */

import React, { useState, useEffect } from 'react';
import { Box, Text, useInput } from 'ink';
import { useStore } from '../store/index.js';
import { getBridge } from '../services/python-bridge.js';
import { icons, getStatusColor } from '../utils/colors.js';
import type { ActiveJob } from '../types/index.js';

export function JobsScreen(): React.ReactElement {
  const setScreen = useStore((s) => s.setScreen);
  const setCurrentJob = useStore((s) => s.setCurrentJob);
  const activeJobs = useStore((s) => s.activeJobs);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [refreshing, setRefreshing] = useState(false);

  const jobsList = Array.from(activeJobs.values());

  // Poll for job updates every 2 seconds
  useEffect(() => {
    const pollJobs = async () => {
      try {
        setRefreshing(true);
        const bridge = getBridge();
        if (bridge.isReady()) {
          await bridge.listJobs();
          // Jobs will be updated via store by the caller
        }
      } catch (err) {
        // Ignore errors during polling
      } finally {
        setRefreshing(false);
      }
    };

    // Initial poll
    pollJobs();

    // Poll every 2 seconds
    const interval = setInterval(pollJobs, 2000);

    return () => clearInterval(interval);
  }, []);

  useInput((input, key) => {
    if (key.upArrow) {
      setSelectedIndex((prev) => Math.max(0, prev - 1));
    } else if (key.downArrow) {
      setSelectedIndex((prev) => Math.min(jobsList.length - 1, prev + 1));
    } else if ((key.return || key.rightArrow) && jobsList.length > 0) {
      // Navigate to running screen for selected job
      const job = jobsList[selectedIndex];
      setCurrentJob(job.id);
      setScreen('running');
    } else if (key.escape || input === 'b' || key.leftArrow) {
      setScreen('welcome');
    }
  });

  if (jobsList.length === 0) {
    return (
      <Box flexDirection="column" paddingX={1}>
        <Box marginBottom={1}>
          <Text color="gray" dimColor>No active jobs</Text>
        </Box>
        <Box>
          <Text color="#D0D1FA" dimColor>[Esc]</Text>
          <Text color="gray" dimColor> Back to home</Text>
        </Box>
      </Box>
    );
  }

  return (
    <Box flexDirection="column" paddingX={1}>
      {/* Header */}
      <Box marginBottom={1}>
        <Text color="#D0D1FA" bold>Active Jobs</Text>
        <Text color="gray" dimColor> ({jobsList.length} total)</Text>
        {refreshing && (
          <>
            <Text color="gray" dimColor> • </Text>
            <Text color="yellow" dimColor>Refreshing...</Text>
          </>
        )}
      </Box>

      {/* Table header */}
      <Box>
        <Text color="gray" dimColor>  Status  </Text>
        <Text color="gray" dimColor>Job ID           </Text>
        <Text color="gray" dimColor>Progress  </Text>
        <Text color="gray" dimColor>Details</Text>
      </Box>

      {/* Jobs list */}
      {jobsList.map((job, idx) => {
        const isSelected = idx === selectedIndex;
        const statusIcon = job.status === 'running' ? '●' : '○';
        const statusColor = job.status === 'running' ? getStatusColor('running') : getStatusColor('pending');
        const progress = job.progress?.percent ?? 0;
        const stage = job.progress?.stage ?? 'init';

        return (
          <Box key={job.id}>
            <Text color={isSelected ? '#D0D1FA' : 'gray'}>
              {isSelected ? '▸ ' : '  '}
            </Text>
            <Text color={statusColor}>{statusIcon}</Text>
            <Text> {job.status.padEnd(8)}</Text>
            <Text> {job.id.substring(0, 16).padEnd(17)}</Text>
            <Text color="white"> {progress.toString().padStart(3)}%</Text>
            <Text color="gray" dimColor> {stage}</Text>
          </Box>
        );
      })}

      {/* Help */}
      <Box marginTop={2} gap={2}>
        <Box>
          <Text color="#D0D1FA" dimColor>[↑↓]</Text>
          <Text color="gray" dimColor> Navigate</Text>
        </Box>
        <Box>
          <Text color="#D0D1FA" dimColor>[Enter]</Text>
          <Text color="gray" dimColor> View progress</Text>
        </Box>
        <Box>
          <Text color="#D0D1FA" dimColor>[Esc]</Text>
          <Text color="gray" dimColor> Back</Text>
        </Box>
      </Box>
    </Box>
  );
}
