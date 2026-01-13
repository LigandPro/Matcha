/**
 * History Screen - view previous docking runs.
 */

import React, { useEffect, useState } from 'react';
import { Box, Text, useInput } from 'ink';
import { useStore } from '../store/index.js';
import { getBridge } from '../services/index.js';
import { icons } from '../utils/colors.js';

interface RunInfo {
  name: string;
  path: string;
  date: string;
  status: string;
}

export function HistoryScreen(): React.ReactElement {
  const setScreen = useStore((s) => s.setScreen);
  const setError = useStore((s) => s.setError);
  const jobConfig = useStore((s) => s.jobConfig);

  const [runs, setRuns] = useState<RunInfo[]>([]);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const outputDir = jobConfig.params?.outputDir ?? './results';
    setLoading(true);
    getBridge()
      .listRuns(outputDir)
      .then((result) => setRuns(result))
      .catch((err) => setError(`Failed to load history: ${err.message}`))
      .finally(() => setLoading(false));
  }, [jobConfig.params?.outputDir, setError]);

  useInput((input, key) => {
    if (key.upArrow) setSelectedIndex((i) => Math.max(0, i - 1));
    else if (key.downArrow) setSelectedIndex((i) => Math.min(runs.length - 1, i + 1));
    else if (key.escape) setScreen('welcome');
  });

  if (loading) {
    return <Box><Text color="yellow">Loading history...</Text></Box>;
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

      <Box marginTop={2}><Text color="gray">Use arrows to navigate, [Esc] to go back</Text></Box>
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
